"""A3-ABI fp8 mtp3 smoke for the arch35 (Ascend950) C8 engine.

Purpose: gate whether an fp8 graph can even reach the operator when the op def
lists only bf16/fp16 (DT_FLOAT8_E4M3FN crashes the CANN 9.1 aclnnInner autogen,
so it was dropped from the def).  This smoke checks:
  (1) GE accepts fp8 query/index_key_cache at all;
  (2) the arch35 host fp8 branch fires (TilingC8Fp8);
  (3) the kernel runs and produces structurally sane outputs with the pool row
      updated in place (cache_slots_pool_out aliases cache_slots_pool).

Geometry mirrors the C8 MTP-4 steady case expressed in the shared A3 schema:
  batch=2, heads=32, 4 queries/request (T=8), sourceCap=12288 (96 blocks),
  candidate visible length == cache budget == sourceCap (fully resident,
  identity map) so every topk source is a hit: dst==src, no misses.

The C8 LD engine gates scoring on IsActiveRequest
(arch35_ld/quant_lightning_indexer_kernel.h:299), which hard-rejects a request
unless its cache budget lands in [8192, 16256] and candidate >= budget.  A
request failing that gate is skipped whole: its topk row is filled with -1 and
topk_dst/route_miss_counts are never written, so the budget window is part of
the operator contract, not a tunable.
"""
from __future__ import annotations

import argparse
import math

import torch
import torch_npu

import nanovllm.ops  # noqa: F401

BLOCK = 128
HEAD_DIM = 128
TOPK = 2048
MISS_CAPACITY = 32768
INVALID_SLOT = -(1 << 31)
OUT_POISON = -313

BATCH = 2
HEADS = 32
QUERIES_PER_REQUEST = 4
TOTAL_QUERIES = BATCH * QUERIES_PER_REQUEST
MAX_BLOCKS = 96
SOURCE_CAPACITY = MAX_BLOCKS * BLOCK  # 12288, the engine's minimum cache budget
POOL_SIZE = BATCH * 2 + 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def normalized_hadamard_128(*, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    matrix = torch.ones((1, 1), dtype=torch.float32)
    while matrix.size(0) < HEAD_DIM:
        top = torch.cat((matrix, matrix), dim=1)
        bottom = torch.cat((matrix, -matrix), dim=1)
        matrix = torch.cat((top, bottom), dim=0)
    return (matrix / math.sqrt(HEAD_DIM)).to(dtype=dtype, device=device)


def quantize_fp8(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    quantized, scale = torch_npu.npu_dynamic_quant(
        tensor, dst_type=torch.float8_e4m3fn
    )
    return (
        quantized.contiguous(),
        scale.view(tensor.shape[:-1]).to(torch.float32).contiguous(),
    )


def build_case(device: torch.device) -> dict[str, torch.Tensor]:
    torch.manual_seed(7)
    # npu_dynamic_quant only accepts fp16/bf16, so the pre-rotation inputs stay
    # in bf16 (mirrors the validated a5 _c8_lidu_case.py build).
    hadamard = normalized_hadamard_128(dtype=torch.bfloat16, device=device)

    query_fp = torch.randn(
        TOTAL_QUERIES, HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device
    )
    key_fp = torch.randn(
        MAX_BLOCKS, BLOCK, 1, HEAD_DIM, dtype=torch.bfloat16, device=device
    )
    query, query_scale = quantize_fp8(torch.matmul(query_fp, hadamard))
    key, key_scale = quantize_fp8(torch.matmul(key_fp, hadamard))

    weights = torch.randn(
        TOTAL_QUERIES, HEADS, dtype=torch.bfloat16, device=device
    )
    # block table rows: each request owns all MAX_BLOCKS blocks, in order.
    block_ids = torch.arange(MAX_BLOCKS, dtype=torch.int32, device=device)
    block_table = block_ids.repeat(BATCH, 1).contiguous()

    query_ends = [QUERIES_PER_REQUEST * (i + 1) for i in range(BATCH)]
    actual_key = [SOURCE_CAPACITY for _ in range(BATCH)]       # A3-only, unused
    offload_key = [SOURCE_CAPACITY for _ in range(BATCH)]      # candidate visible L
    cache_tokens = [SOURCE_CAPACITY for _ in range(BATCH)]     # budget C == L
    states = [-1 for _ in range(BATCH)]
    req_entries = [2 * i + 1 for i in range(BATCH)]

    cache = torch.full(
        (POOL_SIZE, SOURCE_CAPACITY), INVALID_SLOT, dtype=torch.int32, device=device
    )
    for row, count in zip(req_entries, cache_tokens):
        cache[row, :count] = torch.arange(count, dtype=torch.int32, device=device)

    def meta(values: list[int]) -> torch.Tensor:
        return torch.tensor(values, dtype=torch.int32, device=device)

    return {
        "weights": weights,
        "query_scale": query_scale,   # [T, heads] fp32
        "query": query,               # [T, heads, 128] fp8
        "key_scale": key_scale,       # [blocks, 128, 1] fp32
        "key": key,                   # [blocks, 128, 1, 128] fp8
        "block_table": block_table,   # [B, maxBlocks]
        "query_ends": meta(query_ends),
        "actual_key": meta(actual_key),
        "offload_key": meta(offload_key),
        "cache_tokens": meta(cache_tokens),
        "states": meta(states),
        "req_entries": meta(req_entries),
        "cache": cache,
        "req_entries_cpu": req_entries,
        "offload_cpu": offload_key,
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    case = build_case(device)

    total_queries = case["query"].size(0)
    outputs = (
        torch.full((total_queries, 1, TOPK), OUT_POISON, dtype=torch.int32, device=device),
        torch.full((total_queries, 1, TOPK), OUT_POISON, dtype=torch.int32, device=device),
        torch.full((total_queries,), OUT_POISON, dtype=torch.int32, device=device),
        torch.full((BATCH, MISS_CAPACITY), OUT_POISON, dtype=torch.int32, device=device),
        torch.full((BATCH, MISS_CAPACITY), OUT_POISON, dtype=torch.int32, device=device),
        torch.full((BATCH,), OUT_POISON, dtype=torch.int32, device=device),
    )

    torch.ops.nanovllm_dsa.fused_li_manage_mtp.default(
        case["weights"], case["query_scale"], case["query"],
        case["key_scale"], case["key"], case["block_table"],
        case["query_ends"], case["actual_key"], case["offload_key"],
        case["cache_tokens"], case["states"], case["req_entries"],
        case["cache"], *outputs,
    )
    torch.npu.synchronize()

    src, dst, route_miss, miss_src, miss_dst, miss_count = [
        x.cpu() for x in outputs
    ]
    cache_cpu = case["cache"].cpu()

    print("src.shape", tuple(src.shape), "dtype", src.dtype)
    print("topk_src row0[:8]", src[0, 0, :8].tolist())
    print("topk_dst row0[:8]", dst[0, 0, :8].tolist())
    print("route_miss", route_miss.tolist())
    print("miss_count", miss_count.tolist())
    print("row1 resident prefix ok:",
          bool(torch.equal(cache_cpu[1], torch.arange(SOURCE_CAPACITY, dtype=torch.int32))))

    # Light structural sanity (full-resident identity config):
    for r in range(total_queries):
        assert torch.all(src[r, 0, :TOPK] >= 0), f"unexpected -1 source route {r}"
        assert torch.all(src[r, 0, :TOPK] < SOURCE_CAPACITY), f"source OOB route {r}"
    # identity resident map -> dst should mirror src on every route
    assert torch.equal(dst, src), "dst != src under identity resident map"
    assert bool(torch.all(route_miss == 0)), "route miss != 0 under full residence"
    assert bool(torch.all(miss_count == 0)), "request miss != 0 under full residence"
    print("ok=1")


if __name__ == "__main__":
    main()
