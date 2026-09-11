"""fp8 MTP C8 regression across heads / batch / cache-pressure on Ascend950.

The smoke (test_fused_li_manage_mtp_fp8_smoke.py) only proves the fp8 graph
reaches the arch35 kernel and survives a fully-resident request.  This runner
covers what it cannot:

  * heads=32 and heads=64 (the engine's s1BaseSize = ceil(256/heads) fork),
  * batch 1 and 6,
  * non-zero misses, which is the only way to reach the classify / union /
    victim-eviction path at all.

Reference top-k comes from the official npu_quant_lightning_indexer when
available (it shares the kernel's fp8 cube accumulation, so the top-2048
boundary is exact); otherwise a float recomputation of the dequantized fp8
dot products is used (the Hadamard pre-rotation is orthogonal so it cancels).

Assertions mirror the A5 suite's C8 contract:
  * top-2048 SET equals the reference,
  * topk_src/topk_dst/topk_miss_counts obey the miss-prefix / hit-suffix split,
  * topk_dst equals the post-update cache row gathered at topk_src,
  * union miss output is the sorted, complete, unique set of non-resident
    members of the request's top-k union, with valid unique victim slots,
  * every cache row still holds exactly C valid slots in range [0, C).

Usage:
  python3 ut_ops/test_fused_li_manage_mtp_fp8_regress.py --device npu:0
"""
from __future__ import annotations

import argparse
import math

import torch
import torch_npu

import nanovllm.ops  # noqa: F401

from _fp8_mtp_lidu import (
    MAX_CACHE_TOKENS,
    MAX_SOURCE_CAPACITY,
    TOPK,
    assert_pool_row,
    assert_request_pool_entries,
    assert_update_row,
    official_reference_topk,
    ordered_union,
)

BLOCK = 128
HEAD_DIM = 128
QUERIES_PER_REQUEST = 4
MISS_CAPACITY = 32768
OUT_POISON = -313
# Negative, matching the A5 suite: assert_pool_row treats every >=0 cell as a
# resident token, so a pool guard must not be readable as one.
POOL_GUARD = -777777777


def csv_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--heads", type=csv_ints, default=csv_ints("32,64"))
    parser.add_argument("--batch", type=csv_ints, default=csv_ints("1,6"))
    parser.add_argument(
        "--modes",
        default="resident,sampled",
        help="resident = every top-k token cached (miss=0); "
             "sampled = random C-subset of the candidate range (miss>0)",
    )
    parser.add_argument(
        "--source-len",
        type=int,
        default=16384,
        help="candidate/key span; must be 128-aligned and >= --cache-tokens",
    )
    parser.add_argument(
        "--cache-tokens",
        type=int,
        default=12288,
        help="per-request cache budget C; the engine only scores C in [8192, 16256]",
    )
    parser.add_argument("--pool-extra", type=int, default=3)
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


def reference_topk(
    query: torch.Tensor,
    query_scale: torch.Tensor,
    key: torch.Tensor,
    key_scale: torch.Tensor,
    weights: torch.Tensor,
    block_table: torch.Tensor,
    candidate_lens: list[int],
) -> torch.Tensor:
    """Float top-k over the dequantized fp8 operands (Hadamard cancels).

    Approximate: the kernel accumulates the raw fp8 dot in fp32, relu's it, and
    rounds that to bf16 before the weight/scale multiply, so a handful of swaps
    at the top-2048 boundary cannot be ruled out.  Only used when the official
    npu_quant_lightning_indexer is unavailable.
    """
    total, heads, _ = query.shape
    blocks, block, _, dim = key.shape
    q = query.to(torch.float32) * query_scale.to(torch.float32).unsqueeze(-1)
    k = (
        key.to(torch.float32).reshape(blocks * block, dim)
        * key_scale.to(torch.float32).reshape(blocks * block, 1)
    )
    w = weights.to(torch.float32)
    out = torch.empty((total, TOPK), dtype=torch.int64, device=query.device)
    for row in range(block_table.size(0)):
        rows = slice(row * QUERIES_PER_REQUEST, (row + 1) * QUERIES_PER_REQUEST)
        kb = k[block_table[row].long()].reshape(-1, dim)[: candidate_lens[row]]
        # The indexer puts a ReLU on the per-head dot product *before* the weight
        # multiply: arch35_ld/vf/quant_lightning_indexer_vector1.h:439 (float qk
        # path) and :173 (WeightedAccum); the bf16 path instead relu's in the
        # cube (arch35_ld/quant_lightning_indexer_service_cube.h:414 reluEn).
        dots = torch.relu(torch.einsum("qhd,cd->qhc", q[rows], kb))
        scores = dots * w[rows].unsqueeze(-1)
        out[rows] = torch.topk(scores.sum(dim=1), TOPK, dim=-1).indices
    return out.to(torch.int32)


def build_pool_row(
    native_rows: torch.Tensor,
    candidate_len: int,
    budget: int,
    generator: torch.Generator,
    sampled: bool,
) -> torch.Tensor:
    """One cache row holding exactly `budget` valid slots in range [0, budget)."""
    row = torch.full((candidate_len,), -1, dtype=torch.int32)
    if sampled:
        resident = torch.randperm(candidate_len, generator=generator)[:budget]
    else:
        union = torch.unique(native_rows.reshape(-1))
        union = union[union >= 0]
        if union.numel() > budget:
            raise ValueError("top-k union exceeds the cache budget")
        outside_mask = torch.ones(candidate_len, dtype=torch.bool)
        outside_mask[union] = False
        outside = torch.arange(candidate_len, dtype=torch.int64)[outside_mask]
        victims = outside[
            torch.randperm(outside.numel(), generator=generator)[
                : budget - union.numel()
            ]
        ]
        resident = torch.cat((union, victims))
    slots = torch.randperm(budget, generator=generator).to(torch.int32)
    row[resident] = slots
    return row


def build_case(args: argparse.Namespace, device: torch.device, heads: int,
               batch: int, sampled: bool) -> dict:
    torch.manual_seed(args.seed)
    torch.npu.manual_seed_all(args.seed)
    source_len = args.source_len
    blocks = source_len // BLOCK
    budget = args.cache_tokens
    total = batch * QUERIES_PER_REQUEST

    hadamard = normalized_hadamard_128(dtype=torch.bfloat16, device=device)
    query_fp = torch.empty(
        (total, heads, HEAD_DIM), dtype=torch.bfloat16, device=device
    ).uniform_(-1, 1)
    key_fp = torch.empty(
        (blocks, BLOCK, 1, HEAD_DIM), dtype=torch.bfloat16, device=device
    ).uniform_(-1, 1)
    query, query_scale = quantize_fp8(torch.matmul(query_fp, hadamard))
    key, key_scale = quantize_fp8(torch.matmul(key_fp, hadamard))
    weights = torch.empty(
        (total, heads), dtype=torch.bfloat16, device=device
    ).uniform_(0.01, 1.0)

    block_table = torch.stack(
        [torch.randperm(blocks, dtype=torch.int64).to(torch.int32) for _ in range(batch)]
    ).to(device)

    query_ends = [QUERIES_PER_REQUEST * (row + 1) for row in range(batch)]
    actual_key = [source_len] * batch          # A3-only input, unused here
    candidate_lens = [source_len] * batch
    budgets = [budget] * batch
    states = [-1] * batch
    pool_size = batch + args.pool_extra
    req_entries_cpu = torch.randperm(pool_size, dtype=torch.int64)[:batch].to(torch.int32)

    native = official_reference_topk(
        query, key, weights, query_scale, key_scale,
        torch.tensor(query_ends, dtype=torch.int32, device=device),
        torch.tensor(candidate_lens, dtype=torch.int32, device=device),
        block_table,
    )
    if native is None:
        print(
            "[warn] official npu_quant_lightning_indexer unavailable; falling "
            "back to a float reference (top-2048 boundary may drift)",
            flush=True,
        )
        native = reference_topk(
            query, query_scale, key, key_scale, weights, block_table, candidate_lens
        )

    generator = torch.Generator().manual_seed(args.seed + 2)
    pool = torch.full((pool_size, source_len), POOL_GUARD, dtype=torch.int32)
    for row in range(batch):
        pool[int(req_entries_cpu[row])] = build_pool_row(
            native[row * QUERIES_PER_REQUEST : (row + 1) * QUERIES_PER_REQUEST],
            source_len, budget, generator, sampled,
        )
    pool = pool.to(device)

    def meta(values: list[int]) -> torch.Tensor:
        return torch.tensor(values, dtype=torch.int32, device=device)

    return {
        "query": query,
        "query_scale": query_scale,
        "key": key,
        "key_scale": key_scale,
        "weights": weights,
        "block_table": block_table,
        "query_ends": meta(query_ends),
        "actual_key": meta(actual_key),
        "offload_key": meta(candidate_lens),
        "cache_tokens": meta(budgets),
        "states": meta(states),
        "req_entries": req_entries_cpu.to(device),
        "cache": pool,
        "native": native,
        "req_entries_cpu": req_entries_cpu,
        "candidate_lens": candidate_lens,
        "budget": budget,
        "batch": batch,
        "heads": heads,
        "sampled": sampled,
        "pool_size": pool_size,
    }


def validate(case: dict, old: torch.Tensor, new: torch.Tensor,
             outputs: tuple[torch.Tensor, ...]) -> list[int]:
    src, dst, route_miss, miss_src, miss_dst, miss_count = [
        tensor.cpu() for tensor in outputs
    ]
    batch = case["batch"]
    budget = case["budget"]
    source_len = case["candidate_lens"][0]
    total = batch * QUERIES_PER_REQUEST
    native = case["native"].cpu()
    old_cpu = old.cpu()
    new_cpu = new.cpu()

    assert src.shape == (total, 1, TOPK), f"topk_src shape {tuple(src.shape)}"
    assert dst.shape == (total, 1, TOPK), f"topk_dst shape {tuple(dst.shape)}"
    assert route_miss.shape == (total,), "topk_miss_counts must be [T]"
    assert miss_src.shape == (batch, MISS_CAPACITY), "miss_src must be [B, 32768]"
    assert miss_dst.shape == (batch, MISS_CAPACITY), "miss_dst must be [B, 32768]"
    assert miss_count.shape == (batch,), "miss_counts must be [B]"

    assert_request_pool_entries(case["req_entries_cpu"], batch, case["pool_size"])
    active = set(case["req_entries_cpu"].tolist())
    for row in range(case["pool_size"]):
        if row not in active and not torch.equal(old_cpu[row], new_cpu[row]):
            raise AssertionError(f"inactive request-pool row {row} changed")

    observed: list[int] = []
    for row in range(batch):
        pool_row = int(case["req_entries_cpu"][row])
        begin = row * QUERIES_PER_REQUEST
        end = begin + QUERIES_PER_REQUEST
        for query_row in range(begin, end):
            sources = src[query_row, 0]
            sample = int(route_miss[query_row])
            constructed = int(
                (old_cpu[pool_row].gather(0, native[query_row].long()) < 0).sum()
            )
            assert_update_row(
                label=f"heads={case['heads']} batch={batch} q{query_row}",
                sources=sources,
                slots=dst[query_row, 0],
                reference=native[query_row],
                old_row=old_cpu[pool_row],
                new_row=new_cpu[pool_row],
                candidate_len=source_len,
                budget=budget,
                expected_miss=constructed,
                actual_miss=sample,
            )
            observed.append(constructed)

        union = ordered_union(native[begin:end])
        expected = {token for token in union if int(old_cpu[pool_row, token]) < 0}
        count = int(miss_count[row])
        if count != len(expected):
            raise AssertionError(
                f"request {row}: union miss_count={count}, expected={len(expected)}"
            )
        emitted = miss_src[row, :count].long()
        if set(emitted.tolist()) != expected or len(set(emitted.tolist())) != count:
            raise AssertionError(f"request {row}: union miss set is not complete/unique")
        if count > 1 and not bool((emitted[1:] > emitted[:-1]).all()):
            raise AssertionError(f"request {row}: union miss output is not sorted")
        slots = miss_dst[row, :count]
        if bool((slots < 0).any()) or bool((slots >= budget).any()) or \
                len(set(slots.tolist())) != count:
            raise AssertionError(f"request {row}: union victim slots are invalid")
        tail_src = miss_src[row, count:]
        tail_dst = miss_dst[row, count:]
        if tail_src.numel() and int(tail_src.min()) != OUT_POISON:
            raise AssertionError(f"request {row}: miss_src wrote past its prefix")
        if tail_dst.numel() and int(tail_dst.min()) != OUT_POISON:
            raise AssertionError(f"request {row}: miss_dst wrote past its prefix")

        assert_pool_row(old_cpu[pool_row], source_len, budget)
        assert_pool_row(new_cpu[pool_row], source_len, budget)
    return observed


def run(args: argparse.Namespace, device: torch.device, heads: int, batch: int,
        sampled: bool) -> None:
    mode = "sampled" if sampled else "resident"
    label = f"heads={heads} batch={batch} mode={mode}"
    if not (8192 <= args.cache_tokens <= MAX_CACHE_TOKENS):
        raise ValueError(f"{label}: cache_tokens must be in [8192, {MAX_CACHE_TOKENS}]")
    if not (TOPK <= args.cache_tokens < args.source_len <= MAX_SOURCE_CAPACITY):
        raise ValueError(f"{label}: require TOPK <= C < source_len <= 2^18")
    if args.source_len % BLOCK != 0:
        raise ValueError(f"{label}: source_len must be 128-aligned")

    case = build_case(args, device, heads, batch, sampled)
    total = batch * QUERIES_PER_REQUEST
    outputs = (
        torch.full((total, 1, TOPK), OUT_POISON, dtype=torch.int32, device=device),
        torch.full((total, 1, TOPK), OUT_POISON, dtype=torch.int32, device=device),
        torch.full((total,), OUT_POISON, dtype=torch.int32, device=device),
        torch.full((batch, MISS_CAPACITY), OUT_POISON, dtype=torch.int32, device=device),
        torch.full((batch, MISS_CAPACITY), OUT_POISON, dtype=torch.int32, device=device),
        torch.full((batch,), OUT_POISON, dtype=torch.int32, device=device),
    )
    old = case["cache"].clone()
    torch.ops.nanovllm_dsa.fused_li_manage_mtp.default(
        case["weights"], case["query_scale"], case["query"],
        case["key_scale"], case["key"], case["block_table"],
        case["query_ends"], case["actual_key"], case["offload_key"],
        case["cache_tokens"], case["states"], case["req_entries"],
        case["cache"], *outputs,
    )
    torch.npu.synchronize()
    observed = validate(case, old, case["cache"], outputs)
    if sampled and max(observed) == 0:
        raise AssertionError(f"{label}: sampled mode produced no misses")
    print(
        f"{label} source_len={args.source_len} C={args.cache_tokens} "
        f"misses={observed} ok=1",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    for mode in args.modes.split(","):
        mode = mode.strip()
        if mode not in ("resident", "sampled"):
            raise ValueError("--modes entries must be resident or sampled")
        for heads in args.heads:
            if heads not in (32, 64):
                raise ValueError("heads must be 32 or 64")
            for batch in args.batch:
                if batch <= 0 or batch > 16:
                    raise ValueError("batch must be in [1, 16] (LD fork bound)")
                run(args, device, heads, batch, mode == "sampled")
    print("all cases ok=1", flush=True)


if __name__ == "__main__":
    main()
