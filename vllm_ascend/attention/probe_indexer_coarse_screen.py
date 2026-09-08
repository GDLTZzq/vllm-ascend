# -*- coding: utf-8 -*-
"""NPU probe for npu_indexer_coarse_screen (op1) vs the torch reference.

Run ON the A3 (910B/910_93) box AFTER the op has been compiled in
(vllm_ascend wheel rebuilt once so the torch binding + aclnn op are live):

    cd <repo-root>
    python vllm_ascend/attention/probe_indexer_coarse_screen.py

Contract under test:
  For each request r, with seq = actual_seq_lengths_key[r] (= L + g), the op
  must return min(seq, sparse_count) valid positions that are exactly the
  top-`min(seq, sparse_count)` of the proxy score
      score[p] = sum_h w_bar[r,h] * relu(q_bar[r,h] . k[r,p]),  p in [0, seq)
  (positions beyond `seq` masked to -inf, never returned), plus
  `sparse_count - min(seq, sparse_count)` trailing -1 slots.
  q_bar/w_bar are the g-row mean proxies the op computes internally from the
  raw TND query rows and the caller's row_weights (here all ones).

Two verdicts per case:
  * STRICT   : op positions equal torch.topk positions (set) from a golden
               score computed with host-side `.mean(dim=1)` proxies.
  * TIE-SAFE : same, but a position set differing from torch.topk is still a
               PASS if every swap pair carries an exactly equal score (a
               valid tie alternative). Catches real bugs (wrong key read /
               wrong pooling / wrong masking / boundary leak) while tolerating
               benign equal-score order differences.

Real failures we look for are order-independent; STRICT order mismatch on
near-boundary scores caused by a 1-ulp pooling round difference shows up as a
non-tie swap and is reported as a FAIL with the score gap printed.
"""

import torch

import vllm_ascend  # noqa: F401  (registers torch_npu + enables custom ops)
from vllm_ascend.utils import enable_custom_op

COARSE = 4096
BLOCK = 128
HEAD_DIM = 128


def enable():
    enable_custom_op()
    op = torch.ops._C_ascend.npu_indexer_coarse_screen
    torch.npu.synchronize()
    return op


def _relu(x):
    return x.clamp(min=0)


def golden_proxies(q_dq, w, r, g):
    """Host-side proxies, byte-identical to pivot_indexer.select_topk."""
    q_bar = q_dq.view(r, g, -1, HEAD_DIM).mean(dim=1)  # [R,H,Dh]
    w_bar = w.view(r, g, -1).mean(dim=1)               # [R,H]
    return q_bar, w_bar


def golden_scores(q_bar, w_bar, key_flat, block_table, seq_lens):
    """score[r,p] over p in [0, seq_lens[r]) in fp32, -inf beyond."""
    dev = q_bar.device
    r = q_bar.shape[0]
    l_max = int(seq_lens.max())
    pos = torch.arange(l_max, dtype=torch.int64, device=dev)
    slots = block_table[:, pos // BLOCK] * BLOCK + pos % BLOCK  # [R,L]
    k_all = key_flat[slots.reshape(-1)].reshape(r, l_max, -1)   # [R,L,D] bf16
    score = torch.relu(torch.bmm(q_bar, k_all.transpose(1, 2)))  # [R,H,L] fp32
    score = (score * w_bar.unsqueeze(-1)).sum(dim=1)             # [R,L] fp32
    beyond = pos.unsqueeze(0) >= seq_lens.unsqueeze(1).to(torch.int64)
    score = score.masked_fill(beyond, float("-inf"))
    return score


def topk_positions(score, r, seq_lens):
    """Expected position set: top-`min(seq,COARSE)` of each row, sorted."""
    expected = []
    for i in range(r):
        k = int(min(int(seq_lens[i]), COARSE))
        row = score[i, : int(seq_lens[i])]
        topk = torch.topk(row, k).indices.sort().values  # position values
        expected.append(topk.tolist())
    return expected


def check_case(op, tag, r, g, h, seq_lens, dtype=torch.bfloat16):
    dev = torch.npu.current_device()
    seq_lens = torch.as_tensor(seq_lens, dtype=torch.int64, device=dev)
    d = r * g  # total pooled query rows
    # --- raw query rows [D,H,Dh] bf16; production decode head g rows/request.
    torch.manual_seed(0x5EED + r * 131 + h)
    q = (torch.randn(d, h, HEAD_DIM, dtype=torch.float32, device=dev) * 0.08).to(dtype)
    # --- per-row weights [D,H] positive (as in the score formula).
    w = (torch.rand(d, h, dtype=torch.float32, device=dev) * 0.5 + 0.5).to(dtype)
    # --- PA_BSND key cache: give every request contiguous physical blocks and
    # --- write garbage into every slot (masking must hide slots >= seq).
    seq_int = [int(s) for s in seq_lens]
    nblk = [max(1, -(-s // BLOCK)) for s in seq_int]
    starts = torch.cumsum(torch.tensor(nblk, dtype=torch.int64), 0) - nblk[0]
    total_blk = int(starts[-1] + nblk[-1])
    # kv_cache[2] layout: [num_blocks, block_size, 1, head_dim] (PA_BSND). Every
    # physical slot is garbage; only slots < seq_lens[r] may influence results.
    key = (torch.randn(total_blk, BLOCK, 1, HEAD_DIM, dtype=torch.float32, device=dev) * 0.3).to(dtype)
    bt = torch.zeros(r, max(nblk), dtype=torch.int32, device=dev)
    for i in range(r):
        bt[i, : nblk[i]] = starts[i] + torch.arange(nblk[i], device=dev)
    # --- cumulative query-row ends [R] and key seq lens [R].
    cum = torch.cumsum(torch.full((r,), g, dtype=torch.int32, device=dev), 0)
    # --- golden proxies + scores.
    q_bar, w_bar = golden_proxies(q, w, r, g)
    score = golden_scores(q_bar, w_bar, key.reshape(-1, HEAD_DIM), bt, seq_lens)
    expected = topk_positions(score, r, seq_lens)
    # --- invoke the op (row_weights all ones => plain mean, like the torch path).
    row_weights = torch.ones(r, g, dtype=dtype, device=dev)
    out = op(
        q, key, w, row_weights,
        actual_seq_lengths_query=cum,
        actual_seq_lengths_key=seq_lens.to(torch.int32),
        block_table=bt,
        layout_query="TND", layout_key="PA_BSND", sparse_count=COARSE,
    )  # [R,1,COARSE] int32
    out = out.view(r, COARSE).to(torch.int64)

    strict = True
    tie_ok = True
    fails = []
    for i in range(r):
        got = sorted(int(v) for v in out[i] if v >= 0)
        k = min(int(seq_int[i]), COARSE)
        # structural: exactly k valid slots and the rest -1.
        nminus = int((out[i] < 0).sum())
        if len(got) != k or nminus != COARSE - k or (got and max(got) >= seq_int[i]):
            tie_ok = False
            fails.append(f"  r{i}: struct seq={seq_int[i]} nvalid={len(got)} nminus={nminus} "
                         f"oob={max(got) >= seq_int[i] if got else False}")
            continue
        want = expected[i]
        if got == want:
            continue
        strict = False
        # tie-safe: every swap pair must have exactly equal score.
        set_got, set_want = set(got), set(want)
        gap = []
        for a in got:
            if a not in set_want:
                for b in want:
                    if b not in set_got:
                        gap.append(float(score[i, a]) - float(score[i, b]))
        if not gap or any(abs(d) > 0.0 for d in gap):
            tie_ok = False
            min_gap = min(gap) if gap else float("inf")
            fails.append(f"  r{i}: positions differ (min swap-gap={min_gap:.3e}) "
                         f"got[:8]={got[:8]} want[:8]={want[:8]}")
    verdict = "PASS(strict)" if strict else ("PASS(tie-safe)" if tie_ok else "FAIL")
    print(f"[{tag}] r={r} g={g} H={h} seq={seq_int[:6]}... -> {verdict}")
    for f in fails:
        print(f)
    torch.npu.synchronize()
    return strict or tie_ok


def main():
    op = enable()
    ok = True
    # C1 short prefixes (< COARSE), uniform decode head.
    ok &= check_case(op, "C1 uniform-short", r=4, g=4, h=8, seq_lens=[100, 200, 300, 40])
    # C2 short prefixes (< COARSE), varied lengths, no ties in practice.
    ok &= check_case(op, "C2 varied-short", r=6, g=2, h=16,
                     seq_lens=[1500, 2000, 2500, 800, 1200, 3000])
    # C3 long prefixes (> COARSE): top-4096 fully valid, no -1.
    ok &= check_case(op, "C3 over-4k", r=3, g=8, h=16, seq_lens=[4600, 5120, 4200])
    # C4 key domain not a multiple of 128 (tail garbage must be masked out).
    ok &= check_case(op, "C4 non-mult-128", r=3, g=4, h=8, seq_lens=[500, 900, 1300])
    print("\n=== coarse_screen NPU probe (P1 bf16 gate):",
          "ALL PASS" if ok else "SOME FAIL", "===")
    # C5 fp16 instantiation is informational only (PIVOT is BF16-only), so it
    # does not affect the gate's exit code.
    check_case(op, "C5 fp16(info)", r=4, g=4, h=8, seq_lens=[120, 300, 60, 200],
               dtype=torch.float16)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
