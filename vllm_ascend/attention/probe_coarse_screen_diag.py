# -*- coding: utf-8 -*-
"""Single-request over2k diagnostic for npu_indexer_coarse_screen (no rebuild).

One request at a time (r=1), sweeping the domain length L, dumping the raw op
row split at the acc_U/acc_L boundary (2048) so we can see WHICH half of the
top-4096 accumulation is wrong. Self-contained (only needs pivot_indexer).

    source vllm_ascend/_cann_ops_custom/vendors/custom_transformer/bin/set_env.bash
    python vllm_ascend/attention/probe_coarse_screen_diag.py
"""

import torch

import vllm_ascend  # noqa: F401
from vllm_ascend.attention.pivot_indexer import _coarse_screen
from vllm_ascend.utils import enable_custom_op

COARSE = 4096
BLOCK = 128
HD = 128


def run(op, L, g=4, h=8):
    dev = torch.npu.current_device()
    seq = L + g
    torch.manual_seed(0xD1A6 + L)
    q = (torch.randn(g, h, HD, dtype=torch.float32) * 0.08).to(torch.bfloat16)
    w = (torch.rand(g, h, dtype=torch.float32) * 0.5 + 0.5).to(torch.bfloat16)
    nb = max(1, -(-seq // BLOCK))
    key = (torch.randn(nb, BLOCK, 1, HD, dtype=torch.float32) * 0.3).to(torch.bfloat16)
    bt = torch.arange(nb, dtype=torch.int32).reshape(1, -1)
    # golden proxies: byte-identical to pivot_indexer.select_topk's plain mean
    qbar = q.view(1, g, h, HD).mean(dim=1)
    wbar = w.view(1, g, h).mean(dim=1)
    C = _coarse_screen(qbar, wbar, (None, None, key), bt, BLOCK, torch.tensor([L]))
    want = sorted([int(x) for x in C[0] if x >= 0] + list(range(L, L + g)))

    row = op(q.to(dev), key.to(dev), w.to(dev),
             torch.ones(1, g, dtype=torch.bfloat16, device=dev),
             actual_seq_lengths_query=torch.tensor([g], dtype=torch.int32, device=dev),
             actual_seq_lengths_key=torch.tensor([seq], dtype=torch.int32, device=dev),
             block_table=bt.to(dev), layout_query="TND", layout_key="PA_BSND",
             sparse_count=COARSE).reshape(1, -1).cpu()[0]
    got = sorted(int(x) for x in row if x >= 0)
    eg = [a for a in got if a not in set(want)]
    ew = [b for b in want if b not in set(got)]
    fn = next((i for i in range(row.numel()) if row[i] < 0), row.numel())
    print(f"L={L:5d} seq={seq:5d} W={row.numel()} n={len(got)} ref={len(want)} "
          f"minus={int((row < 0).sum())} first_neg={fn} dup={len(got)-len(set(got))} "
          f"{'OK' if got == want else 'DIFF'}")
    if got != want:
        print(f"   extra={eg[:8]} (lo<2048 {len([a for a in eg if a<2048])}, "
              f"hi>=2048 {len([a for a in eg if a>=2048])})")
        print(f"   miss ={ew[:8]} (lo<2048 {len([b for b in ew if b<2048])}, "
              f"hi>=2048 {len([b for b in ew if b>=2048])})")
        print(f"   row[:12]={[int(x) for x in row[:12]]} "
              f"row[2040:2052]={[int(x) for x in row[2040:2052]]}")
        print(f"   want[:12]={want[:12]}")
    return got == want


def main():
    enable_custom_op()
    op = torch.ops._C_ascend.npu_indexer_coarse_screen
    torch.npu.synchronize()
    print("=== single-request by-domain-length diagnostic (bf16) ===")
    ok = True
    for L in [200, 520, 1024, 1536, 2048, 2049, 2560, 4096, 4600]:
        ok &= run(op, L)
        torch.npu.synchronize()
    print("=== ", "ALL OK" if ok else "SOME DIFF", " ===")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
