# -*- coding: utf-8 -*-
"""NPU probe for npu_indexer_coarse_screen (coarse top-4096 + own g tokens).

Run ON the A3 (910B/910_93) box AFTER the op has been compiled in (the
vllm_ascend wheel was rebuilt once so the torch binding + aclnn op are live):

    cd <repo-root>
    python vllm_ascend/attention/probe_indexer_coarse_screen_v2.py

Contract under test (A1 + own-window: the op emits the coarse screen AND the
request's own g tokens; the wider local window stays in python). For each decode
request r with group width g (query rows [r*g, r*g+g)) and key prefix length
seq_lens[r] = L + g, the op must produce ONE row of width
``Align8(COARSE + g)`` (COARSE = 4096) whose valid front is exactly

  [ coarse top-min(L, COARSE) of the domain-[0, L) proxy score ]
  [ the request's own g tokens L .. L+g-1 ]
  [ -1 pad to Align8(COARSE + g) ]

so valid COUNT == min(L, COARSE) + g. L == 0 rows (seq_lens == g, first token)
have an empty coarse segment and are ``[0 .. g-1] + -1 pad``:

  score[r, p] = sum_h w_bar[r,h] * relu(q_bar[r,h] . k[r,p]),  p in [0, L)
  q_bar/w_bar = the request's g-row mean proxy (row_weights all ones here =>
  plain mean, byte-identical to python's q_bar/w_bar)

The domain EXCLUDES the request's own g tokens [L, L+g) (the paper recalls the
pool at the group's FIRST position), which is why the own tokens are appended
verbatim with no dedup. pivot_indexer then runs
``_inject_local_window(C, seq_lens, g)`` on the op row to union in the wider
group window [L-g+1, L+g), so the probe validates both ends:
  (1) op row valid SET == the golden row set -- the torch reference
      ``_coarse_screen(q_bar, w_bar, ..., dom=[0,L))`` valid set plus the own
      tokens [L, L+g); count == min(L, COARSE) + g; structure = valid front
      (unique, < seq_lens), then -1 tail;
  (2) the python window pass on the op row == the same pass on the torch
      reference row -- i.e. op + python window reproduces pivot_indexer's
      candidate set.

Two verdicts per case (same scheme as the refine op probe):
  * STRICT   : op row valid SET equals the golden reference SET (order within
               a row is irrelevant -- the refine op re-sorts by score), the
               valid COUNT equals the reference's, and (2) holds.
  * TIE-SAFE : a differing set is still a PASS only if every differing pair
               lies in the coarse domain [0, L) and carries an EXACTLY equal
               proxy score (a legitimate tie alternative). Any real bug --
               wrong key read / wrong pooling / domain leak / own-token loss
               / count mismatch -- fails loudly (own tokens are >= L, so a
               mismatch there always trips the domain guard).

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
    """The op's fixed per-row output width = Align8(sparse_count + g) (A1 + own window)."""
    return (COARSE + g + 7) & ~7


def golden_row(C_ref_row, L: int, g: int):
    """Expected op row valid entries (order-insensitive): coarse valid set ++ own [L, L+g)."""
    return [int(x) for x in C_ref_row if x >= 0] + list(range(L, L + g))


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
    dom_c = seq_c - g  # [R] domain prefix length L (pool is over [0, L))
    # cumsum promotes int32 -> int64 (op's actual_seq_lengths_query must be
    # int32), so cast back explicitly.
    cum_c = torch.cumsum(torch.full((r,), g, dtype=torch.int32), 0).to(torch.int32)  # [R] cum ends

    # --- golden reference (host): coarse only over [0, L), then the window
    # pass that pivot_indexer runs on top of the op row (A1 contract).
    q_bar, w_bar = golden_proxies(q_c, w_c, r, g)
    C_ref = _coarse_screen(q_bar, w_bar, (None, None, key_c), bt_c, BLOCK, dom_c)
    gold_win, aslk_win = _inject_local_window(C_ref, seq_c, g)
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
    # (2) python window pass on the op row vs on the torch coarse row.
    op_win, _ = _inject_local_window(out.to(torch.int64), seq_c, g)
    win_ok = all(
        sorted(int(x) for x in op_win[i] if x >= 0)
        == sorted(int(x) for x in gold_win[i] if x >= 0)
        for i in range(r)
    )

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
        want = sorted(golden_row(C_ref[i], Lg - g, g))  # coarse valid ++ own [L, L+g)
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
    if not win_ok:
        strict = False
        fails.append("  window-pass mismatch: op row + _inject_local_window != "
                     "torch coarse row + _inject_local_window")
    verdict = "PASS(strict)" if strict else ("PASS(tie-safe)" if tie_ok else "FAIL")
    print(f"[{tag}] r={r} g={g} H={h} W={W} win={'ok' if win_ok else 'DIFF'} "
          f"seq={seq_int[:6]} -> {verdict}")
    for f in fails:
        print(f)
    torch.npu.synchronize()
    return strict or tie_ok


def main():
    op = enable()
    ok = True
    # C1 uniform short prefixes (L + g <= 4096): every row is the whole domain
    # [0, L) front-loaded, then the own g tokens [L, L+g), then -1 to Align8.
    ok &= check_case(op, "C1 uniform-short", r=4, g=4, h=8,
                     seq_lens=[100, 200, 300, 40])
    # C2 one truncated (L > 4096) request mixed with short rows in one batch:
    # the long row runs a real top-4096 over [0, L), short rows are whole-domain.
    ok &= check_case(op, "C2 mixed-trunc", r=5, g=2, h=16,
                     seq_lens=[4600, 200, 1500, 40, 300])
    # C3 all over 4k: every row a genuine score top-4096 over [0, L).
    ok &= check_case(op, "C3 over-4k", r=3, g=8, h=16,
                     seq_lens=[4600, 5120, 4200])
    # C4 the exact L == 4096 boundary (whole domain recalled, no coarse -1 pad)
    # plus non-128-multiple domains (L = 4696, 4296).
    ok &= check_case(op, "C4 non128-dom", r=3, g=4, h=8,
                     seq_lens=[4100, 4700, 4300])
    # C5 first-token requests (seq == g, L == 0, empty coarse domain): the op row
    # must be [0, g) + -1 pad (own tokens only, no coarse segment).
    ok &= check_case(op, "C5 first-token", r=4, g=2, h=8,
                     seq_lens=[2, 300, 2, 1500])
    # C6 many requests in one batch: spills across cores and covers both AIV
    # parities (odd/even) at once, with L==0 divert, whole-domain short rows
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
