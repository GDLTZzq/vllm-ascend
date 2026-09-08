# -*- coding: utf-8 -*-
"""NPU probe for npu_indexer_coarse_screen v2 (fused coarse + local window).

Run ON the A3 (910B/910_93) box AFTER the op has been compiled in (the
vllm_ascend wheel was rebuilt once so the torch binding + aclnn op are live):

    cd <repo-root>
    python vllm_ascend/attention/probe_indexer_coarse_screen_v2.py

Contract under test (v2 fused semantics, replaces the v1 op entirely).
For each decode request r with group width g (query rows [r*g, r*g+g)) and
key prefix length seq_lens[r] = L + g, the op must produce ONE candidate row
of width W8 = Align8(COARSE + 2g - 1) whose valid front is exactly:

  (top-`min(L, COARSE)` of the domain-[0, L) proxy score)  union  window
  window = the group's local-window union [L-g+1, L+g), deduped against the
  coarse set, own tokens [L, L+g) always present (they never enter the pool)

where the proxy score per request is the g-row mean proxy of the raw query
rows (row_weights all ones here => plain mean, byte-identical to python's
q_bar/w_bar). The valid entries are FRONT-COMPACTED (coarse first, then
window new entries ascending), everything past the valid count is exactly -1
pad to W8. This is precisely what pivot_indexer's torch reference
``_coarse_screen(q_bar, w_bar, ..., dom)`` + ``_inject_local_window(C, ...)``
produce, so that reference (imported from the module under test) is the
golden.

Two verdicts per case (same scheme as the refine op probe):
  * STRICT   : op row valid SET equals the golden reference SET (order within
               a row is irrelevant -- the refine op re-sorts by score) and
               the valid COUNT equals the reference aslk'.
  * TIE-SAFE : a differing set is still a PASS only if every differing pair
               lies in the score domain [0, L) and carries an EXACTLY equal
               proxy score (a legitimate tie alternative). Any real bug --
               wrong key read / wrong pooling / domain leak / own-token loss
               / window misplacement / count mismatch -- fails loudly.

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
    """W8 = Align8(sparse_count + 2g - 1), the op's fixed per-row output width."""
    return (COARSE + 2 * g - 1 + 7) // 8 * 8


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
    seq_lens = torch.as_tensor(seq_lens, dtype=torch.int64, device=dev)
    dom = seq_lens - g  # [K] domain prefix length L (pool is over [0, L))
    d = r * g  # total pooled query rows (decode head: g rows per request)
    torch.manual_seed(0x5EED + r * 131 + h)
    q = (torch.randn(d, h, HEAD_DIM, dtype=torch.float32, device=dev) * 0.08).to(dtype)
    w = (torch.rand(d, h, dtype=torch.float32, device=dev) * 0.5 + 0.5).to(dtype)
    # --- PA_BSND key cache: contiguous physical blocks per request, garbage in
    # --- every slot (masking must hide slots >= dom).
    seq_int = [int(s) for s in seq_lens]
    nblk = [max(1, -(-s // BLOCK)) for s in seq_int]
    # Contiguous physical blocks: request i owns blocks [starts[i], starts[i+1]).
    blk_cum = torch.cumsum(torch.tensor(nblk, dtype=torch.int64), 0)
    starts = torch.cat([torch.zeros(1, dtype=torch.int64), blk_cum[:-1]])
    total_blk = int(blk_cum[-1])
    key = (torch.randn(total_blk, BLOCK, 1, HEAD_DIM,
                       dtype=torch.float32, device=dev) * 0.3).to(dtype)
    bt = torch.zeros(r, max(nblk), dtype=torch.int32, device=dev)
    for i in range(r):
        bt[i, :nblk[i]] = starts[i] + torch.arange(nblk[i], device=dev)
    # --- cumulative query-row ends [R] and key prefix lens [R] (= L + g).
    cum = torch.cumsum(torch.full((r,), g, dtype=torch.int32, device=dev), 0)

    # --- golden reference: _coarse_screen over [0, L) then window union.
    q_bar, w_bar = golden_proxies(q, w, r, g)
    C = _coarse_screen(q_bar, w_bar, (None, None, key), bt, BLOCK, dom)
    gold, aslk_gold = _inject_local_window(C, seq_lens, g)
    score = domain_scores(q_bar, w_bar, key.reshape(-1, HEAD_DIM), bt, dom)

    # --- invoke the op (row_weights all ones => plain mean, torch-equivalent).
    W = _row_width(g)
    row_weights = torch.ones(r, g, dtype=dtype, device=dev)
    out = op(
        q, key, w, row_weights,
        actual_seq_lengths_query=cum,
        actual_seq_lengths_key=seq_lens.to(torch.int32),
        block_table=bt,
        layout_query="TND", layout_key="PA_BSND", sparse_count=COARSE,
    )  # [R, 1, W8] int32
    out = out.reshape(r, -1)
    if out.shape[1] != W:
        print(f"[{tag}] FATAL: op row width {out.shape[1]} != expected W8={W}")
        torch.npu.synchronize()
        return False

    strict = True
    tie_ok = True
    fails = []
    for i in range(r):
        row = out[i]
        Lg = seq_int[i]
        thr = Lg - g  # domain [0, L); positions >= L are own tokens
        n = int((row >= 0).sum().item())
        head = row[:n]
        tail = row[n:]
        got = sorted(int(x) for x in head)
        want = sorted(int(x) for x in gold[i] if x >= 0)
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
        # tie-safe: differences must be domain-only and every swap exactly tied.
        set_got, set_want = set(got), set(want)
        extra_g = [a for a in got if a not in set_want]
        extra_w = [b for b in want if b not in set_got]
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
                         f"{extra_w[:6]} own_diff={bad}) got[:8]={got[:8]} "
                         f"want[:8]={want[:8]}")
    verdict = "PASS(strict)" if strict else ("PASS(tie-safe)" if tie_ok else "FAIL")
    print(f"[{tag}] r={r} g={g} H={h} W8={W} seq={seq_int[:6]} -> {verdict}")
    for f in fails:
        print(f)
    torch.npu.synchronize()
    return strict or tie_ok


def main():
    op = enable()
    ok = True
    # C1 uniform short prefixes (L + g <= 4096): window = own g only, whole
    # prefix candidate set.
    ok &= check_case(op, "C1 uniform-short", r=4, g=4, h=8,
                     seq_lens=[100, 200, 300, 40])
    # C2 one truncated (L > 4096) request mixed with short rows in one batch:
    # the long row runs a real top-4096 over [0, L) + window dedup, short rows
    # are whole-prefix.
    ok &= check_case(op, "C2 mixed-trunc", r=5, g=2, h=16,
                     seq_lens=[4600, 200, 1500, 40, 300])
    # C3 all over 4k: every row a genuine score top-4096 over [0, L).
    ok &= check_case(op, "C3 over-4k", r=3, g=8, h=16,
                     seq_lens=[4600, 5120, 4200])
    # C4 non-128-multiple key domain + the exact L == 4096 boundary (coarse
    # covers the whole domain, window is own tokens only).
    ok &= check_case(op, "C4 bound-non128", r=3, g=4, h=8,
                     seq_lens=[4100, 4700, 4300])
    # C5 first-token requests (seq == g, L == 0, empty domain): the row must
    # still carry the own window [0, g) -- not a blank row.
    ok &= check_case(op, "C5 first-token", r=4, g=2, h=8,
                     seq_lens=[2, 300, 2, 1500])
    # C6 many requests in one batch: spills across cores and covers both AIV
    # parities (odd/even) at once, with L==0 divert, whole-domain short rows
    # and several truncated rows mixed. The row set/count must hold per request
    # regardless of which (odd/even, which core) owned it.
    ok &= check_case(op, "C6 multi-core-mix", r=14, g=4, h=8,
                     seq_lens=[4, 4600, 300, 4, 200, 4700, 4, 900, 1500, 4,
                               4100, 60, 4, 2500])
    # C7 g == 16 (host's max group): widest local window (2g-1 = 31) and the
    # largest W8 output row.
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
