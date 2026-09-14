# -*- coding: utf-8 -*-
"""Canary probe: are the over2k coarse scores REAL, or is the sort a no-op?

Why the other probes can't answer this: for L <= 4096 the coarse segment is the
WHOLE domain, so every permutation "passes" a set check; and for L > 4096 any
4096-of-L subset still overlaps the true top-4096 heavily (at L=4600 identity
selection gives ovlp ~= 3647, exactly what was observed). So both v2 and the
order probe are blind to the failure mode "the op sorts by a CONSTANT".

This probe makes the score humanly checkable. The true proxy score is given
exactly TWO bf16-representable levels -- 10.0 at EVEN positions, 1.0 at ODD --
by setting q = e0 and key[p,0] = 10 / 1, so score(p) = 8 * (10 or 1) for the
8 heads. Then a working sort MUST emit

    [0, 2, 4, 6, ...]   (all evens, ascending within ties)   then [1, 3, 5, ...]

whereas a dead / constant-score sort emits the raw insertion order

    [0, 1, 2, 3, ...].

Why two levels rather than strict monotonicity: bf16 cannot resolve adjacent
integers above 256, so a "score = beta*p" ramp would COLLIDE in bf16 and look
exactly like a constant. 10.0 and 1.0 are bf16-exact, so a real sort has no
ties between the two levels at all.

Metric = even-fraction of the first half of the coarse segment:
    1.00  -> scores are real (the sort reorders by score)
    ~0.50 -> scores are constant (stable no-op sort -> raw order)

    source vllm_ascend/_cann_ops_custom/vendors/custom_transformer/bin/set_env.bash
    python vllm_ascend/attention/probe_coarse_screen_canary.py
"""

import torch

import vllm_ascend  # noqa: F401
from vllm_ascend.utils import enable_custom_op

COARSE = 4096
BLOCK = 128
HD = 128


def run(op, L, g=4, h=8):
    dev = torch.npu.current_device()
    seq = L + g
    # q = e0 for every (row, head) -> q_bar = e0; w = 1 -> w_bar = 1.
    q = torch.zeros(g, h, HD, dtype=torch.bfloat16)
    q[:, :, 0] = 1.0
    w = torch.ones(g, h, dtype=torch.bfloat16)
    # key[p, :, 0, 0] = 10 (p even) / 1 (p odd); all other dims 0.
    nb = max(1, -(-seq // BLOCK))
    key = torch.zeros(nb, BLOCK, 1, HD, dtype=torch.bfloat16)
    pos = torch.arange(nb * BLOCK)
    lvl = torch.where(pos % 2 == 0, torch.full_like(pos, 10.0, dtype=torch.float32),
                      torch.ones_like(pos, dtype=torch.float32))
    key.view(-1, HD)[:, 0] = lvl.to(torch.bfloat16)
    bt = torch.arange(nb, dtype=torch.int32).reshape(1, -1)

    row = op(q.to(dev), key.to(dev), w.to(dev),
             torch.ones(1, g, dtype=torch.bfloat16, device=dev),
             actual_seq_lengths_query=torch.tensor([g], dtype=torch.int32, device=dev),
             actual_seq_lengths_key=torch.tensor([seq], dtype=torch.int32, device=dev),
             block_table=bt.to(dev), layout_query="TND", layout_key="PA_BSND",
             sparse_count=COARSE).reshape(1, -1).cpu()[0]

    c = min(L, COARSE)
    coarse = [int(x) for x in row[:c]]
    valid = [x for x in coarse if x >= 0]
    ev = [1 if x % 2 == 0 else 0 for x in valid]
    half = len(ev) // 2
    ef = sum(ev[:half]) / max(1, half)
    tag = "REAL-SORT" if ef > 0.99 else ("DEAD-SORT" if ef < 0.75 else "MIXED")
    print(f"L={L:5d} c={c:5d} n={len(valid):5d} evenfrac(1st half)={ef:.3f} "
          f"nminus={int((row < 0).sum())} {tag}")
    print(f"   out[:32]={coarse[:32]}")
    torch.npu.synchronize()
    return ef


def main():
    enable_custom_op()
    op = torch.ops._C_ascend.npu_indexer_coarse_screen
    torch.npu.synchronize()
    print("=== score canary: even positions score 10, odd score 1 ===")
    print("-- whole domain (L<=4096): a real sort emits all evens first --")
    for L in [128, 512, 1024, 2048, 4096]:
        run(op, L)
    print("-- truncated (L>4096): true top-4096 = 2300 evens then 1796 odds --")
    for L in [4200, 4600]:
        run(op, L)
    print("=== done ===")


if __name__ == "__main__":
    main()
