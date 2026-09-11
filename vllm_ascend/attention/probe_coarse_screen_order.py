# -*- coding: utf-8 -*-
"""Order-correctness probe for npu_indexer_coarse_screen (over2k path).

Why: every L <= 4096 case in probe_indexer_coarse_screen_v2 passes as a SET
because the op keeps the whole domain [0,L) -- the set is trivially right no
matter how scrambled the ORDER is. So the sort/merge pipeline has never been
validated. This probe scores a single request against the fp32 reference and
asks, per L:

  * is the coarse segment (row[0:c], c = min(L,4096)) actually sorted
    DESCENDING by the true score?  (desc_viol = # adjacent inversions)
  * is each output position i at its true rank?  (max_rank_dev, L<=4096)
  * for L>4096: overlap with the true top-c, and the score-multiset gap.

The sweep walks chunk boundaries so the failure localizes:
  L <= 512   -> exactly ONE SortAll chunk, ZERO merges: isolates SortAll+score.
  L > 512    -> real 2-list merges kick in: isolates the accumulator merge.

    source vllm_ascend/_cann_ops_custom/vendors/custom_transformer/bin/set_env.bash
    python vllm_ascend/attention/probe_coarse_screen_order.py
"""

import torch

import vllm_ascend  # noqa: F401
from vllm_ascend.utils import enable_custom_op

COARSE = 4096
BLOCK = 128
HD = 128
CHUNK = 512


def ref_scores(q, w, key, L):
    """fp32 reference score[p], p in [0,L): sum_h w_bar[h]*relu(q_bar[h].k[p]).

    Byte-identical to pivot_indexer.select_topk's proxy mean + _coarse_screen's
    formula (same bf16 q/w/key, cast to fp32 for the dot/accumulate).
    """
    qb = q.to(torch.float32).mean(0)          # [H, HD]
    wb = w.to(torch.float32).mean(0)          # [H]
    kk = key.to(torch.float32).reshape(-1, HD)[:L]   # [L, HD]
    dots = torch.relu(qb @ kk.t())            # [H, L]
    return (wb[:, None] * dots).sum(0)        # [L]


def run(op, L, g=4, h=8):
    dev = torch.npu.current_device()
    seq = L + g
    torch.manual_seed(0xD1A6 + L)  # same stream as probe_coarse_screen_diag
    q = (torch.randn(g, h, HD, dtype=torch.float32) * 0.08).to(torch.bfloat16)
    w = (torch.rand(g, h, dtype=torch.float32) * 0.5 + 0.5).to(torch.bfloat16)
    nb = max(1, -(-seq // BLOCK))
    key = (torch.randn(nb, BLOCK, 1, HD, dtype=torch.float32) * 0.3).to(torch.bfloat16)
    bt = torch.arange(nb, dtype=torch.int32).reshape(1, -1)

    row = op(q.to(dev), key.to(dev), w.to(dev),
             torch.ones(1, g, dtype=torch.bfloat16, device=dev),
             actual_seq_lengths_query=torch.tensor([g], dtype=torch.int32, device=dev),
             actual_seq_lengths_key=torch.tensor([seq], dtype=torch.int32, device=dev),
             block_table=bt.to(dev), layout_query="TND", layout_key="PA_BSND",
             sparse_count=COARSE).reshape(1, -1).cpu()[0]

    n = int((row >= 0).sum())
    c = min(L, COARSE)
    coarse = [int(x) for x in row[:c]]
    score = ref_scores(q, w, key, L)                     # [L] fp32
    rng = float(score.max() - score.min()) or 1.0
    tol = 1e-3 * rng

    # permutation check + descending-order violations + per-rank deviation
    perm = sorted(coarse) == list(range(c)) if L <= COARSE else None
    s = torch.tensor([float(score[i]) for i in coarse])
    # descending order requires s[i] > s[i+1]; an inversion is s[i+1] > s[i]
    inv = (s[1:] > s[:-1] + tol)
    desc_viol = int(inv.sum())
    rank_dev = 0
    if L <= COARSE:
        order = torch.argsort(score, descending=True)
        rk = torch.empty(L, dtype=torch.long)
        rk[order] = torch.arange(L)
        rank_dev = int((rk[torch.tensor(coarse)] - torch.arange(c)).abs().max())

    # overlap / score-multiset gap for the truncated case
    ovlp = gap = None
    if c == COARSE:
        truth = set(int(x) for x in torch.argsort(score, descending=True)[:COARSE])
        ovlp = len(set(coarse) & truth)
        top_s = torch.sort(score, descending=True).values[:COARSE]
        gap = float((torch.sort(s).values - torch.sort(top_s).values).abs().max())

    # where do the inversions sit (bucket = output-rank position // 512)?
    buckets = {}
    for i in range(c - 1):
        if inv[i]:
            buckets[i // CHUNK] = buckets.get(i // CHUNK, 0) + 1
    first_bad = int(torch.nonzero(inv)[0]) if desc_viol else -1

    tag = "OK" if (desc_viol == 0 and (ovlp is None or ovlp == COARSE)) else "BAD"
    extra = f" ovlp={ovlp} gap={gap:.3e}" if ovlp is not None else \
            f" perm={perm} rankdev={rank_dev}"
    print(f"L={L:5d} seq={seq:5d} n={n:5d} c={c:5d} desc_viol={desc_viol:5d} "
          f"first={first_bad:5d}->blk{first_bad // CHUNK if first_bad >= 0 else -1}{extra} {tag}")
    if desc_viol:
        print(f"   viol_by_out_blk={dict(sorted(buckets.items()))}")
        j = first_bad
        print(f"   @pos{j}(blk{j // CHUNK}): idx {coarse[j]}(s={float(score[coarse[j]]):.5f}) "
              f"> idx {coarse[j + 1]}(s={float(score[coarse[j + 1]]):.5f}) -> inverted")
    torch.npu.synchronize()
    return desc_viol


def main():
    enable_custom_op()
    op = torch.ops._C_ascend.npu_indexer_coarse_screen
    torch.npu.synchronize()
    print("=== order/ranking diagnostic (bf16, r=1) ===")
    print("-- 1 chunk (no merge): isolates SortAll + score --")
    for L in [128, 256, 384, 512]:
        run(op, L)
    print("-- multi chunk (2-list merge live): isolates the accumulator --")
    for L in [513, 640, 1024, 1536, 2048, 2560, 3072, 4096]:
        run(op, L)
    print("-- truncated (L > 4096): merge + real top-4096 selection --")
    for L in [4097, 4600]:
        run(op, L)
    print("=== done ===")


if __name__ == "__main__":
    main()
