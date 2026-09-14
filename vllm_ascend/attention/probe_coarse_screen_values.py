# -*- coding: utf-8 -*-
"""Read the op's OWN coarse scores (value half) out of the output row.

Requires a build with COARSE_SCREEN_DUMP_VALUES=1
(csrc/.../arch22/indexer_coarse_screen_service_vector.h). In that build
CopyOutCoarseRow writes the sorted globalTopkUb_ *value* half into the row
instead of the indices, so the row's first min(L,4096) int32 words are the
float32 bit patterns of the op's own scores, in the op's own output order.

Why this is the decisive measurement: the canary/order probes only see the
INDEX order, which cannot distinguish

  (a) the sort key is wrong  (Sort32/MrgSort never ranked by the score), from
  (b) the output assembly reads the wrong half (values sorted, indices not).

Dumping the values separates them:
  values monotonically non-increasing  -> (b) the sort worked, assembly is wrong
  values interleaved / not decreasing  -> (a) the key fed to Sort32 is wrong

Input is the canary's two-level setup: q=e0, w=1, key[p,:,0,0] = 10 (even p) /
1 (odd p), so the true score is 80 at even and 8 at odd positions. A working
sort MUST emit 80...80 then 8...8; a dead one emits 80,8,80,8,...

    source vllm_ascend/_cann_ops_custom/vendors/custom_transformer/bin/set_env.bash
    python vllm_ascend/attention/probe_coarse_screen_values.py
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
    q = torch.zeros(g, h, HD, dtype=torch.bfloat16)
    q[:, :, 0] = 1.0
    w = torch.ones(g, h, dtype=torch.bfloat16)
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
    vals = row[:c].clone().view(torch.float32)          # int32 words -> float bits
    v = [float(x) for x in vals]

    # expected under a working sort: ceil(c/2) highs (80) then floor(c/2) lows (8)
    hi = 80.0
    tol = 1e-2
    n_hi = sum(1 for x in v if abs(x - hi) < tol)
    n_lo = sum(1 for x in v if abs(x - 1.0 * 8) < tol)
    n_other = c - n_hi - n_lo
    # desc_viol counts places where the next value is definitely larger
    viol = sum(1 for i in range(c - 1) if v[i + 1] > v[i] + tol)
    tag = "SORT-OK(values monotonically decreasing)" if viol == 0 else "SORT-DEAD/KEY-WRONG"
    print(f"L={L:5d} c={c:5d} n_hi(80)={n_hi:5d} n_lo(8)={n_lo:5d} other={n_other:5d} "
          f"desc_viol={viol:5d} {tag}")
    print(f"   vals[:32]={[round(x, 3) for x in v[:32]]}")
    if n_other:
        print(f"   unexpected values (first 8): "
              f"{[round(x, 3) for x in v if abs(x - hi) > tol and abs(x - 8.0) > tol][:8]}")
    torch.npu.synchronize()
    return viol


def main():
    enable_custom_op()
    op = torch.ops._C_ascend.npu_indexer_coarse_screen
    torch.npu.synchronize()
    print("=== op's own coarse values, in the op's own output order ===")
    print("-- expect 80 ... 80 then 8 ... 8 if the sort ranks by score --")
    ok = True
    for L in [128, 512, 1024, 2048, 4096, 4600]:
        ok &= (run(op, L) == 0)
    print("=== ", "SORT OK" if ok else "SORT NOT RANKING BY SCORE", " ===")


if __name__ == "__main__":
    main()
