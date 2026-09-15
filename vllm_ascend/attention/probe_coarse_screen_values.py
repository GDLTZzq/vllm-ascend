# -*- coding: utf-8 -*-
"""Read the op's own coarse rows out of the row itself, as raw (value,index) pairs.

Requires a build with COARSE_SCREEN_DUMP_VALUES=1
(csrc/attention/indexer_coarse_screen/op_kernel/arch22/indexer_coarse_screen_service_vector.h).
In that build CopyOutCoarseRow copies the merge accumulator globalTopkUb_ verbatim into
the row's coarse段, so

    row word 2k   = float32 bit pattern of the k-th ranked score
    row word 2k+1 = that same pair's index

The canary/order probes see only the INDEX half, which cannot separate the remaining
failure modes. Reading BOTH halves of every pair can. The canary's truth is *index
based* -- q = e0 and key[p,0] = 10 when p % PERIOD == 0 else 1 -- and this probe never
hard-codes the score *values*: the op's normalisation over heads/groups is unknown a
priori (mean vs weighted sum), so the two levels are recovered from the data itself and
only their ordering and their consistency with the index level are used.

Half-row caveat: the coarse segment is only min(L,sparseCount) WORDS wide, so the dump
carries just half that many (value,index) pairs -- the readable window is the op's own
top-c/2 ranked items, not the whole domain. The level must therefore occupy well under
half the domain; the canary's plain 1/2 split would put only level-10 items in the
window and make a *working* sort look like KEYS-CONSTANT. PERIOD = 8 (1/8 of positions)
keeps both levels visible.

Verdicts:
  SORT-OK       values non-increasing, and each value equals its own index's level
                -> the sort ranks by score; the defect is downstream (assembly,
                which is already verified byte-identical to production).
  SORT-NO-OP    pairs intact (value == its index's level for ~all k) but unordered
                -> the key array held distinct scores and the comparator did nothing.
  STAGING-POS   values follow the output position k, not the pair's own index
                -> the staging stage wrote the wrong thing into the value half.
  KEYS-CONSTANT all values collapse to one -> the key row fed to Sort32 was constant.
  MULTI-LEVEL / NAN / VALUES-UNEXPLAINED -> see the printed histogram + samples.

    source vllm_ascend/_cann_ops_custom/vendors/custom_transformer/bin/set_env.bash
    python vllm_ascend/attention/probe_coarse_screen_values.py
"""

import math

import torch

import vllm_ascend  # noqa: F401
from vllm_ascend.utils import enable_custom_op

COARSE = 4096
BLOCK = 128
HD = 128
CHUNK = 512  # kernel s2 base block == merge granularity
PERIOD = 8   # level-10 positions; must be >> 2 so both levels fit the readable half-row


def build(L, g=4, h=8):
    dev = torch.npu.current_device()
    seq = L + g
    q = torch.zeros(g, h, HD, dtype=torch.bfloat16)
    q[:, :, 0] = 1.0
    w = torch.ones(g, h, dtype=torch.bfloat16)
    nb = max(1, -(-seq // BLOCK))
    key = torch.zeros(nb, BLOCK, 1, HD, dtype=torch.bfloat16)
    pos = torch.arange(nb * BLOCK)
    lvl = torch.where(pos % PERIOD == 0, torch.full_like(pos, 10.0, dtype=torch.float32),
                      torch.ones_like(pos, dtype=torch.float32))
    key.view(-1, HD)[:, 0] = lvl.to(torch.bfloat16)
    bt = torch.arange(nb, dtype=torch.int32).reshape(1, -1)
    return q.to(dev), key.to(dev), w.to(dev), bt.to(dev), g, seq


def analyze(L, row):
    """Pure: row = the op's int32 output row (word 2k = score bits, word 2k+1 = index).

    Kept free of torch.npu / the op so it can be exercised offline (see selftest()).
    """
    c = min(L, COARSE)
    n = c // 2
    words = row[:2 * n]
    assert words.dtype in (torch.int32, torch.int64), f"unexpected dtype {words.dtype}"
    w32 = words.to(torch.int32).contiguous()          # int32 words == float32 bits
    val = w32[0::2].contiguous().view(torch.float32).tolist()   # even words = scores
    idx = [int(x) for x in words[1::2]]                         # odd words  = indices

    nnan = sum(1 for x in val if math.isnan(x))
    clean = [(k, v, i) for k, (v, i) in enumerate(zip(val, idx))
             if not math.isnan(v) and not math.isinf(v)]
    uniq = sorted({round(v, 4) for _, v, _ in clean}, reverse=True)
    l_hi, l_lo = (uniq[0], uniq[-1]) if len(uniq) >= 2 else (None, None)
    gap = (l_hi - l_lo) if l_hi is not None else 0.0
    tol = 0.05 * gap if gap > 0 else 0.0              # scale-free, tiny vs the gap

    n_hi = sum(1 for _, v, _ in clean if l_hi is not None and abs(v - l_hi) <= tol)
    n_lo = sum(1 for _, v, _ in clean if l_lo is not None and abs(v - l_lo) <= tol)
    nneg = sum(1 for i in idx if i < 0)

    def frac(mode):
        """mode 'idx': hi where the pair's OWN INDEX is a level-10 position; 'pos': where k is."""
        tot = ok = 0
        for k, v, i in clean:
            key = i if mode == "idx" else k
            if key < 0:
                continue
            tot += 1
            if abs(v - (l_hi if key % PERIOD == 0 else l_lo)) <= tol:
                ok += 1
        return ok / max(1, tot)

    pair_ok = 0.0 if l_hi is None else frac("idx")
    pos_ok = 0.0 if l_hi is None else frac("pos")
    viol = sum(1 for k in range(len(val) - 1)
               if not math.isnan(val[k]) and not math.isnan(val[k + 1])
               and val[k + 1] > val[k] + tol)
    ident = sum(1 for k in range(n) if idx[k] == k) / max(1, n)
    rev = sum(1 for k in range(n) if idx[k] == c - 1 - k) / max(1, n)
    runs = 1 + sum(1 for k in range(n - 1) if idx[k + 1] // CHUNK != idx[k] // CHUNK)

    if nnan:
        tag = f"NAN  {nnan}/{len(val)} values are NaN (comparisons all false -> no reorder)"
    elif len(uniq) == 0:
        tag = "EMPTY"
    elif len(uniq) == 1:
        tag = f"KEYS-CONSTANT  every value == {uniq[0]} (the Sort32 key row was constant)"
    elif len(uniq) > 2:
        tag = f"MULTI-LEVEL  {len(uniq)} distinct values (not the expected 2 levels)"
    elif viol == 0 and pair_ok > 0.99:
        tag = "SORT-OK      values non-increasing + pairs consistent -> defect downstream"
    elif pair_ok > 0.99:
        tag = "SORT-NO-OP   pairs intact but unordered -> comparator did nothing"
    elif pos_ok > 0.99:
        tag = "STAGING-POS  values follow output position, not the pair's own index"
    else:
        tag = "VALUES-UNEXPLAINED"

    print(f"L={L:5d} c={c:5d} pairs={n:5d} lvl=({l_hi},{l_lo}) "
          f"n_hi={n_hi:5d} n_lo={n_lo:5d} uniq={len(uniq):4d} nan={nnan:4d} "
          f"viol={viol:5d} pair_ok={pair_ok:.3f} pos_ok={pos_ok:.3f} "
          f"ident={ident:.3f} rev={rev:.3f} n_neg={nneg:4d} chunkruns={runs:4d}")
    print(f"   {tag}")
    print(f"   val[:16]={[round(x, 4) for x in val[:16]]}")
    print(f"   idx[:16]={idx[:16]}")
    print(f"   chunk(idx[:16])={[i // CHUNK for i in idx[:16]]}")
    if len(uniq) > 2:
        hist = {}
        for _, v, _ in clean:
            hist[round(v, 4)] = hist.get(round(v, 4), 0) + 1
        top = sorted(hist.items(), key=lambda kv: -kv[1])[:10]
        print(f"   value histogram (top 10 of {len(uniq)}): {top}")
    return tag


def to_row(L, pairs):
    """(value, index) list -> an int32 row with the words the kernel would write."""
    v = torch.tensor([p[0] for p in pairs], dtype=torch.float32)
    i = torch.tensor([p[1] for p in pairs], dtype=torch.int32)
    row = torch.empty(2 * len(pairs), dtype=torch.int32)
    row[0::2] = v.view(torch.int32)
    row[1::2] = i
    return row


def selftest():
    """Offline check of the verdict logic -- no NPU, no op, no rebuild.

    Each case is an (name, L, pairs) that a real kernel could emit; the expected
    verdict is asserted so a wrong probe cannot waste a rebuild.
    """
    L = 256  # 128 readable pairs = the op's top-128 of 256: 32 level-10 + 96 level-1
    hi = [i for i in range(L) if i % PERIOD == 0]
    lo = [i for i in range(L) if i % PERIOD != 0]
    lvl = lambda i: 10.0 if i % PERIOD == 0 else 1.0  # noqa: E731
    cases = [
        ("correct sort (top half = all level-10 first)", L,
         [(80.0, i) for i in hi] + [(8.0, i) for i in lo], "SORT-OK"),
        ("correct sort, op normalises to 10/1 (levels differ)",
         L, [(lvl(i), i) for i in hi + lo], "SORT-OK"),
        ("dead sort, pairs intact, raw insertion order", L,
         [(lvl(k), k) for k in range(L)], "SORT-NO-OP"),
        ("constant key row", L, [(7.5, k) for k in range(L)], "KEYS-CONSTANT"),
        ("values follow output position (index order reversed)",
         L, [(lvl(k), L - 1 - k) for k in range(L)], "STAGING-POS"),
        ("NaN key half", L, [(float("nan") if k < L // 2 else 8.0, k) for k in range(L)], "NAN"),
    ]
    bad = 0
    for name, LL, pairs, want in cases:
        print(f"-- selftest: {name}  (expect {want})")
        got = analyze(LL, to_row(LL, pairs))
        ok = got.split()[0] == want
        bad += (not ok)
        print(f"   -> {'PASS' if ok else 'FAIL'} (got {got.split()[0]}, want {want})")
    print(f"=== selftest {'PASS' if not bad else f'FAIL ({bad})'} ===")
    return bad


def run(op, L):
    q, key, w, bt, g, seq = build(L)
    dev = q.device
    row = op(q, key, w, torch.ones(1, g, dtype=torch.bfloat16, device=dev),
             actual_seq_lengths_query=torch.tensor([g], dtype=torch.int32, device=dev),
             actual_seq_lengths_key=torch.tensor([seq], dtype=torch.int32, device=dev),
             block_table=bt, layout_query="TND", layout_key="PA_BSND",
             sparse_count=COARSE).reshape(1, -1).cpu()[0]
    tag = analyze(L, row)
    torch.npu.synchronize()
    return tag


def main():
    enable_custom_op()
    op = torch.ops._C_ascend.npu_indexer_coarse_screen
    torch.npu.synchronize()
    print("=== op's own (value,index) pairs, in the op's own output order ===")
    print(f"-- expect: the level-10 (index % {PERIOD} == 0) pairs first, then the level-1 ones --")
    for L in [128, 512, 1024, 2048, 4096, 4200, 4600]:
        run(op, L)
    print("=== done ===")


if __name__ == "__main__":
    main()
