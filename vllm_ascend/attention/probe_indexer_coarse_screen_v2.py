# -*- coding: utf-8 -*-
"""NPU probe for npu_indexer_coarse_screen (ranked coarse screen + window union).

Run ON the A3 (910B/910_93) box AFTER the op has been compiled in (the
vllm_ascend wheel was rebuilt once so the torch binding + aclnn op are live):

    cd <repo-root>
    python vllm_ascend/attention/probe_indexer_coarse_screen_v2.py

Contract under test (window union fused into the op): the op emits the WHOLE
step-2 candidate row -- the ranked coarse screen AND the verbatim window union.
For each decode request r with group width g (query rows [r*g, r*g+g)) and key
prefix length aslk = seq_lens[r] = L + g, let lo = max(0, aslk - 2g + 1) (the
lower bound of the group's local window [L-g+1, L+g)). The op must produce ONE
row of width ``Align8(COARSE + 2g - 1)`` (COARSE = 4096) whose valid front is
exactly

  [ ranked top-min(lo, COARSE) of the domain-[0, lo) proxy score ]
  [ the window union [lo, aslk) verbatim, ascending ]
  [ -1 pad to Align8(COARSE + 2g - 1) ]

so valid COUNT == min(lo, COARSE) + (aslk - lo). The two segments are disjoint
by construction (lo IS the window's own lower bound), which is why the window
needs no dedup and no membership test. Rows with aslk <= 2g-1 (first token)
have lo == 0, an empty ranked segment, and are ``[0 .. aslk-1] + -1 pad``:

  score[r, p] = sum_h w_bar[r,h] * relu(q_bar[r,h] . k[r,p]),  p in [0, lo)
  q_bar/w_bar = the request's g-row mean proxy (row_weights all ones here =>
  plain mean, byte-identical to python's q_bar/w_bar)

The ranked domain EXCLUDES both the request's own g tokens [L, L+g) and the
window's left half [lo, L) (the paper recalls the pool at the group's FIRST
position). The op's row is exactly what the torch path produces --
``_coarse_screen(dom=lo)`` followed by ``_inject_local_window`` -- so the probe
validates both ends:
  (1) op row valid SET == that torch row's valid set; count ==
      min(lo, COARSE) + (aslk - lo); structure = valid front (unique,
      < aslk), then -1 tail;
  (2) ``(op_row >= 0).sum(dim=1)`` == the torch row's valid count -- that count
      is the aslk pivot_indexer now reads off the op row (the op path no longer
      runs _inject_local_window).

Two verdicts per case (same scheme as the refine op probe):
  * STRICT   : op row valid SET equals the golden reference SET (order within
               a row is irrelevant -- the refine op re-sorts by score), the
               valid COUNT equals the reference's, and (2) holds.
  * TIE-SAFE : a differing set is still a PASS only if every differing pair
               lies in the ranked domain [0, lo) and carries an EXACTLY equal
               proxy score (a legitimate tie alternative). Any real bug --
               wrong key read / wrong pooling / domain leak / window-tail loss
               / count mismatch -- fails loudly (window-union positions are
               >= lo, so a mismatch there always trips the domain guard).

bf16 is the PIVOT dtype and the gate; the fp16 case is informational only.
"""

import torch

import vllm_ascend  # noqa: F401  (registers torch_npu + enables custom ops)
from vllm_ascend.attention.pivot_indexer import (
    _coarse_screen,
    _inject_local_window,
)
from vllm_ascend.utils import enable_custom_op

COARSE = 4096
BLOCK = 128
HEAD_DIM = 128


def enable():
    enable_custom_op()
    op = torch.ops._C_ascend.npu_indexer_coarse_screen
    torch.npu.synchronize()
    return op


def _row_width(g: int) -> int:
    """The op's fixed per-row output width = Align8(COARSE + 2g - 1)."""
    return (COARSE + 2 * g - 1 + 7) & ~7


def golden_row(torch_row_entry):
    """Expected op row valid entries (order-insensitive): the torch row's valid front.

    The golden row is that of the torch path -- ``_coarse_screen(dom=lo)``
    followed by ``_inject_local_window`` -- so its valid entries are exactly
    [ranked top-min(lo, COARSE) of domain [0, lo)] ++ [window union [lo, aslk)].
    """
    return [int(x) for x in torch_row_entry if x >= 0]


def golden_proxies(q_dq, w, r, g):
    """Host-side proxies, byte-identical to pivot_indexer.select_topk's mean."""
    q_bar = q_dq.view(r, g, -1, HEAD_DIM).mean(dim=1)  # [R,H,Dh]
    w_bar = w.view(r, g, -1).mean(dim=1)               # [R,H]
    return q_bar, w_bar


def domain_scores(q_bar, w_bar, key_flat, block_table, dom_lens):
    """fp32 score[r, p] over p in [0, dom_lens[r]) -- mirrors _coarse_screen's
    formula exactly. Consumed only to license tie-safe swaps on the domain."""
    dev = q_bar.device
    r = q_bar.shape[0]
    lmax = int(dom_lens.max())
    pos = torch.arange(lmax, dtype=torch.int64, device=dev)
    slots = block_table[:, pos // BLOCK] * BLOCK + pos % BLOCK  # [R,L]
    k_all = key_flat[slots.reshape(-1)].view(r, lmax, -1)       # [R,L,Dh]
    q32 = q_bar.to(torch.float32)
    w32 = w_bar.to(torch.float32)
    k32 = k_all.to(torch.float32)
    score = torch.relu(torch.bmm(q32, k32.transpose(1, 2)))     # [R,H,L]
    score = (score * w32.unsqueeze(-1)).sum(dim=1)              # [R,L] fp32
    return score


def check_case(op, tag, r, g, h, seq_lens, dtype=torch.bfloat16):
    dev = torch.npu.current_device()
    seq_int = [int(s) for s in seq_lens]
    d = r * g  # total pooled query rows (decode head: g rows per request)
    # Generate EVERYTHING on the host once: the golden reference runs on the
    # host tensors and the NPU op receives byte-identical copies. (a) NPU's
    # aclnnAdd/arange reject int64 elementwise math, which _coarse_screen and
    # _inject_local_window rely on; (b) CPU/NPU RNG streams differ, so the two
    # sides cannot be generated independently.
    torch.manual_seed(0x5EED + r * 131 + h)
    q_c = (torch.randn(d, h, HEAD_DIM, dtype=torch.float32) * 0.08).to(dtype)
    w_c = (torch.rand(d, h, dtype=torch.float32) * 0.5 + 0.5).to(dtype)
    nblk = [max(1, -(-s // BLOCK)) for s in seq_int]
    # Contiguous physical blocks: request i owns blocks [starts[i], starts[i+1]).
    blk_cum = torch.cumsum(torch.tensor(nblk, dtype=torch.int64), 0)
    starts = torch.cat([torch.zeros(1, dtype=torch.int64), blk_cum[:-1]])
    total_blk = int(blk_cum[-1])
    key_c = (torch.randn(total_blk, BLOCK, 1, HEAD_DIM,
                         dtype=torch.float32) * 0.3).to(dtype)
    bt_c = torch.zeros(r, max(nblk), dtype=torch.int32)
    for i in range(r):
        bt_c[i, :nblk[i]] = starts[i] + torch.arange(nblk[i], dtype=torch.int64)
    seq_c = torch.tensor(seq_int, dtype=torch.int64)  # [R] natural seq = L + g
    # ranked domain [0, lo), lo = max(0, aslk - 2g + 1) (the window's own lower bound)
    dom_c = (seq_c - 2 * g + 1).clamp(min=0)  # [R]
    # cumsum promotes int32 -> int64 (op's actual_seq_lengths_query must be
    # int32), so cast back explicitly.
    cum_c = torch.cumsum(torch.full((r,), g, dtype=torch.int32), 0).to(torch.int32)  # [R] cum ends

    # --- golden reference (host): the exact torch row the op must reproduce --
    # coarse ranking over the shrunk domain [0, lo), then the window union pass.
    q_bar, w_bar = golden_proxies(q_c, w_c, r, g)
    C_ref = _coarse_screen(q_bar, w_bar, (None, None, key_c), bt_c, BLOCK, dom_c)
    gold_row, _aslk_win = _inject_local_window(C_ref, seq_c, g)
    score = domain_scores(q_bar, w_bar, key_c.reshape(-1, HEAD_DIM), bt_c, dom_c)

    # --- invoke the op (byte-identical NPU copies; row_weights ones => mean).
    W = _row_width(g)
    q, w, key = q_c.to(dev), w_c.to(dev), key_c.to(dev)
    bt = bt_c.to(dev)
    row_weights = torch.ones(r, g, dtype=dtype, device=dev)
    out = op(
        q, key, w, row_weights,
        actual_seq_lengths_query=cum_c.to(dev),
        actual_seq_lengths_key=seq_c.to(torch.int32).to(dev),
        block_table=bt,
        layout_query="TND", layout_key="PA_BSND", sparse_count=COARSE,
    )  # [R, 1, W] int32
    out = out.reshape(r, -1).cpu()  # structural compare runs on the host
    if out.shape[1] != W:
        print(f"[{tag}] FATAL: op row width {out.shape[1]} != expected {W}")
        torch.npu.synchronize()
        return False
    # (1)+(2) live in the per-row loop below: op valid SET == the torch row's
    # valid SET, and op valid COUNT == the torch row's valid count -- that count
    # is the aslk pivot_indexer now reads off the op row (the op path no longer
    # runs _inject_local_window). Guard the expectation itself first: the torch
    # count must equal the closed form min(lo, COARSE) + (aslk - lo).
    nref = [len(golden_row(gold_row[i])) for i in range(r)]
    exp_cnt = [min(int(dom_c[i]), COARSE) + (seq_int[i] - int(dom_c[i]))
               for i in range(r)]
    cnt_ok = nref == exp_cnt

    strict = True
    tie_ok = True
    fails = []
    for i in range(r):
        row = out[i]
        Lg = seq_int[i]
        thr = int(dom_c[i])  # ranked domain [0, lo); positions >= lo are window tail
        n = int((row >= 0).sum().item())
        head = row[:n]
        tail = row[n:]
        got = sorted(int(x) for x in head)
        want = sorted(golden_row(gold_row[i]))  # torch row valid front
        nwant = len(want)
        # structural: head all valid < Lg and unique; tail all exactly -1;
        # valid count == the reference's (aslk').
        oob = (head.clamp(min=0).max().item() >= Lg) if n else False
        struct_ok = (
            (tail == -1).all().item()
            and int(head.unique().numel()) == n
            and n == nwant
            and not oob
        )
        if not struct_ok:
            tie_ok = False
            fails.append(f"  r{i}: STRUCT seq={Lg} nvalid={n} ref={nwant} "
                         f"nminus={int((row < 0).sum().item())} oob={oob} "
                         f"dup={n - int(head.unique().numel())}")
            continue
        if got == want:
            continue
        strict = False
        # tie-safe: differences must be ranked-domain-only and every swap exactly tied.
        set_got, set_want = set(got), set(want)
        extra_g = [a for a in got if a not in set_want]
        extra_w = [b for b in want if b not in set_got]
        # STRUCTURE DUMP: the run-length encoding of each side's dropped domain
        # positions. A contiguous run (e.g. "10-511") = a whole chunk lost; a
        # scattered pattern = a score/ranking divergence. Cheap (O(L), pure host).
        def _rle(vals):
            runs = []
            for x in vals:
                if runs and x == runs[-1][1] + 1:
                    runs[-1][1] = x
                else:
                    runs.append([x, x])
            return " ".join(f"{a}-{b}" if b > a else str(a) for a, b in runs)

        miss_g = [p for p in range(thr) if p not in set_got]
        miss_w = [p for p in range(thr) if p not in set_want]
        fails.append(f"     op drops {len(miss_g)}/{thr}: {_rle(miss_g)[:400]}")
        fails.append(f"     ref drops {len(miss_w)}/{thr}: {_rle(miss_w)[:400]}")
        bad = [a for a in extra_g if a >= thr] + [b for b in extra_w if b >= thr]
        gap = []
        for a in extra_g:
            if a >= thr:
                continue
            for b in extra_w:
                if b >= thr:
                    continue
                gap.append(float(score[i, a]) - float(score[i, b]))
        if bad or not gap or any(abs(dd) > 0.0 for dd in gap):
            tie_ok = False
            fails.append(f"  r{i}: set differs (extra_g={extra_g[:6]} extra_w="
                         f"{extra_w[:6]} tail_diff={bad}) got[:8]={got[:8]} "
                         f"want[:8]={want[:8]}")
    if not cnt_ok:
        strict = False
        fails.append(f"  count closed-form mismatch: torch counts {nref} != "
                     f"min(lo, COARSE) + (aslk - lo) = {exp_cnt}")
    verdict = "PASS(strict)" if strict else ("PASS(tie-safe)" if tie_ok else "FAIL")
    print(f"[{tag}] r={r} g={g} H={h} W={W} cnt={'ok' if cnt_ok else 'DIFF'} "
          f"seq={seq_int[:6]} -> {verdict}")
    for f in fails:
        print(f)
    torch.npu.synchronize()
    return strict or tie_ok


def main():
    op = enable()
    ok = True
    # C1 uniform short prefixes (L + g <= 4096 => lo < COARSE): the whole ranked
    # domain is recalled, so the row valid set is the entire prefix [0, aslk),
    # then -1 to Align8.
    ok &= check_case(op, "C1 uniform-short", r=4, g=4, h=8,
                     seq_lens=[100, 200, 300, 40])
    # C2 one truncated (lo > 4096) request mixed with short rows in one batch:
    # the long row runs a real top-4096 over [0, lo); short rows are whole-tail.
    ok &= check_case(op, "C2 mixed-trunc", r=5, g=2, h=16,
                     seq_lens=[4600, 200, 1500, 40, 300])
    # C3 all over 4k: every row a genuine score top-4096 over [0, lo).
    ok &= check_case(op, "C3 over-4k", r=3, g=8, h=16,
                     seq_lens=[4600, 5120, 4200])
    # C4 seq 4100 with g=4 => L == 4096 exactly, lo = 4093 < COARSE: the whole
    # ranked domain is recalled (no coarse -1 pad) -- and non-128-multiple
    # ranked domains (L = 4696, lo = 4693; L = 4296, lo = 4293).
    ok &= check_case(op, "C4 non128-dom", r=3, g=4, h=8,
                     seq_lens=[4100, 4700, 4300])
    # C5 first-token requests (aslk == g, lo == 0, empty ranked domain): the op
    # row must be [0, g) + -1 pad (window tail only, no ranked segment).
    ok &= check_case(op, "C5 first-token", r=4, g=2, h=8,
                     seq_lens=[2, 300, 2, 1500])
    # C6 many requests in one batch: spills across cores and covers both AIV
    # parities (odd/even) at once, with lo==0 divert, whole-tail short rows
    # and several truncated rows mixed. The row set/count must hold per request
    # regardless of which (odd/even, which core) owned it.
    ok &= check_case(op, "C6 multi-core-mix", r=14, g=4, h=8,
                     seq_lens=[4, 4600, 300, 4, 200, 4700, 4, 900, 1500, 4,
                               4100, 60, 4, 2500])
    # C7 g == 16 (host's max group): widest local window (2g-1 = 31).
    ok &= check_case(op, "C7 g16-maxwin", r=3, g=16, h=8,
                     seq_lens=[4600, 400, 16])
    print("\n=== coarse_screen v2 NPU probe (single-op bf16 gate):",
          "ALL PASS" if ok else "SOME FAIL", "===")
    # C8 fp16 instantiation is informational only (PIVOT is BF16-only), so it
    # does not affect the gate's exit code.
    check_case(op, "C8 fp16(info)", r=4, g=4, h=8,
               seq_lens=[120, 300, 60, 200], dtype=torch.float16)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
