"""Standardized fused_li_manage_mtp correctness, lifecycle and latency UT.

This file is intentionally executable without pytest so it can be copied to an
Ascend validation host together with a locally built operator package.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch
import torch_npu

import nanovllm.ops  # noqa: F401
from _op_utils import require_local_opapi


TOPK = 2048
BLOCK = 128
HEAD_DIM = 128
MISS_CAPACITY = 32768
INVALID_SLOT = -(1 << 31)
PACKED_SOURCE_CAPACITY = 1 << 17
MAX_SOURCE_CAPACITY = 1 << 21
MAX_ROUTES = 14
MAX_CACHE_TOKENS = 32640
PACKED_INVALID_SLOT = (1 << 15) - 1
# Retain the ordinary wide-union workload envelope from the backup suite.
# This is a test-fixture bound, not an operator-side occurrence-map contract.
WIDE_ORDINARY_OCCURRENCE_LIMIT = 6912


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=(
            "correctness", "lifecycle", "invalid", "perf",
            "legacy-perf", "mtp-perf", "standard-mtp-perf",
            "mixed-mtp-perf", "mixed-standard-mtp-perf",
            "mixed-state-perf",
            "transition-perf", "first-decode-perf",
            "replacement-regression", "first-decode-regression",
            "steady-semantic-regression", "sort4096-regression",
            "wide-route-regression", "wide-compact-union-regression",
            "occurrence-regression", "payload-codec-regression",
            "long-regression", "key-tag-regression", "all",
        ),
        default="all",
    )
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--heads", type=int, choices=(32, 64), default=32)
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument(
        "--q-pattern",
        default="1,2,3,4,5,6,7",
        help="comma-separated Q values; one request is created per value",
    )
    parser.add_argument(
        "--state-pattern",
        default="-3,-2,-1",
        help=(
            "comma-separated request states for mixed-state-perf; must "
            "match --q-pattern and mix at least two of -3, -2 and -1"
        ),
    )
    parser.add_argument("--source-capacity", type=int, default=16384)
    parser.add_argument("--offload-len", type=int, default=8192)
    parser.add_argument("--cache-tokens", type=int, default=8192)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--baseline-json", type=Path)
    parser.add_argument("--write-perf-json", type=Path)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--source-len", type=int, default=65536)
    parser.add_argument("--query-miss-count", type=int, default=200)
    parser.add_argument("--union-miss-count", type=int, default=300)
    parser.add_argument("--query-noise", type=float, default=0.25)
    return parser.parse_args()


def cumulative(values: list[int]) -> list[int]:
    result: list[int] = []
    total = 0
    for value in values:
        total += value
        result.append(total)
    return result


def validate_dynamic_inputs(
    *,
    query_ends: list[int],
    actual_key: list[int],
    offload_key: list[int],
    cache_tokens: list[int],
    request_state: list[int],
    req_pool_entries: list[int],
    total_queries: int,
    source_capacity: int,
    pool_size: int,
) -> None:
    """Reference for checks performed by the scheduler before tensor creation."""

    if (
        source_capacity <= 0
        or source_capacity % BLOCK
        or source_capacity > MAX_SOURCE_CAPACITY
    ):
        raise ValueError("source capacity must be 128-aligned and <=2^21")
    if pool_size <= 0:
        raise ValueError("pool_size must be positive")
    batch = len(query_ends)
    fields = (actual_key, offload_key, cache_tokens, request_state, req_pool_entries)
    if not batch or any(len(field) != batch for field in fields):
        raise ValueError("all request metadata must be non-empty and have length B")
    previous = 0
    for request in range(batch):
        end = query_ends[request]
        q = end - previous
        if not 1 <= q <= MAX_ROUTES:
            raise ValueError("Q must be in [1,14]")
        if not q <= actual_key[request] <= source_capacity:
            raise ValueError("actual_seq_lengths_key is out of range")
        if not 0 <= req_pool_entries[request] < pool_size:
            raise ValueError("req_pool_entries is out of range")
        state = request_state[request]
        if state not in (-3, -2, -1):
            raise ValueError("request_state must be -3, -2 or -1")
        if state != -3:
            length = offload_key[request]
            capacity = cache_tokens[request]
            if not capacity <= length <= actual_key[request]:
                raise ValueError("requires C <= L <= actual_seq_lengths_key")
            if length < TOPK or length % BLOCK or capacity % BLOCK:
                raise ValueError("L/C alignment or minimum is invalid")
            causal_limit = ((actual_key[request] - q) // BLOCK) * BLOCK
            if length > causal_limit:
                raise ValueError("offload prefix is not visible to every query")
            if length <= q * TOPK:
                if capacity != length:
                    raise ValueError("C must equal L for a small offload prefix")
            elif not q * TOPK <= capacity <= MAX_CACHE_TOKENS:
                raise ValueError("C is outside the multi-route cache budget")
            if capacity > MAX_CACHE_TOKENS:
                raise ValueError("C exceeds the product cache budget 32640")
        previous = end
    if previous != total_queries:
        raise ValueError("last actual_seq_lengths_query value must equal T")
    if len(set(req_pool_entries)) != batch:
        raise ValueError("active requests must use distinct pool rows")


def native_topk(
    query: torch.Tensor,
    key: torch.Tensor,
    weights: torch.Tensor,
    block_table: torch.Tensor,
    visible_lengths: list[int],
) -> torch.Tensor:
    rows: list[torch.Tensor] = []
    for row, visible in enumerate(visible_lengths):
        result = torch_npu.npu_lightning_indexer(
            query=query[row : row + 1],
            key=key,
            weights=weights[row : row + 1],
            actual_seq_lengths_query=torch.tensor(
                [1], dtype=torch.int32, device=query.device
            ),
            actual_seq_lengths_key=torch.tensor(
                [visible], dtype=torch.int32, device=query.device
            ),
            block_table=block_table[row : row + 1],
            layout_query="TND",
            layout_key="PA_BSND",
            sparse_count=TOPK,
            sparse_mode=0,
        )
        output = result[0] if isinstance(result, (tuple, list)) else result
        rows.append(output.reshape(-1)[:TOPK])
    return torch.stack(rows)


def build_case(
    args: argparse.Namespace,
    *,
    q_values: list[int],
    states: list[int],
) -> dict[str, object]:
    if len(q_values) != len(states):
        raise ValueError("q_values and states must have equal length")
    if args.source_capacity % BLOCK or args.source_capacity > MAX_SOURCE_CAPACITY:
        raise ValueError("source capacity must be 128-aligned and <=2^21")
    batch = len(q_values)
    total_queries = sum(q_values)
    query_ends = cumulative(q_values)
    # The final route sees this length; adding one block makes the aligned
    # offload prefix causally visible even for Q=14.
    actual_key = [args.offload_len + BLOCK for _ in q_values]
    offload_key = [args.offload_len for _ in q_values]
    cache_tokens = [
        args.offload_len if args.offload_len <= q * TOPK else args.cache_tokens
        for q in q_values
    ]
    # Active requests must own distinct pool rows.  Keep them deliberately
    # non-contiguous without wrapping: modulo (B + 3) aliases rows once B=7
    # (for example requests 0/5 and 1/6), creating invalid concurrent writes.
    pool_size = batch * 2 + 1
    req_entries = [request * 2 + 1 for request in range(batch)]
    validate_dynamic_inputs(
        query_ends=query_ends,
        actual_key=actual_key,
        offload_key=offload_key,
        cache_tokens=cache_tokens,
        request_state=states,
        req_pool_entries=req_entries,
        total_queries=total_queries,
        source_capacity=args.source_capacity,
        pool_size=pool_size,
    )
    torch.manual_seed(args.seed)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    device = torch.device(args.device)
    blocks = args.source_capacity // BLOCK
    query = torch.randn(
        total_queries, args.heads, HEAD_DIM, dtype=dtype, device=device
    )
    weights = torch.randn(total_queries, args.heads, dtype=dtype, device=device)
    key = torch.randn(blocks, BLOCK, 1, HEAD_DIM, dtype=dtype, device=device)
    block_table = torch.arange(blocks, dtype=torch.int32, device=device).repeat(batch, 1)
    # native_topk takes one block-table row per query.
    query_to_request = [request for request, q in enumerate(q_values) for _ in range(q)]
    route_table = block_table[
        torch.tensor(query_to_request, dtype=torch.int64, device=device)
    ].contiguous()
    cache_cpu = torch.full(
        (pool_size, args.source_capacity), INVALID_SLOT, dtype=torch.int32
    )
    for request, state in enumerate(states):
        if state != -1:
            continue
        row = req_entries[request]
        count = cache_tokens[request]
        cache_cpu[row, :count] = torch.arange(count, dtype=torch.int32)
    return {
        "q_values": q_values,
        "states": states,
        "query_ends": query_ends,
        "actual_key": actual_key,
        "offload_key": offload_key,
        "cache_tokens": cache_tokens,
        "req_entries": req_entries,
        "query": query,
        "weights": weights,
        "query_scale": torch.zeros(total_queries, args.heads, dtype=torch.float32, device=device),
        "key": key,
        "key_scale": torch.zeros(blocks, BLOCK, 1, dtype=torch.float32, device=device),
        "block_table": block_table,
        "route_table": route_table,
        "cache_seed": cache_cpu.to(device),
        "metadata": tuple(
            torch.tensor(values, dtype=torch.int32, device=device)
            for values in (
                query_ends,
                actual_key,
                offload_key,
                cache_tokens,
                states,
                req_entries,
            )
        ),
    }


def make_outputs(case: dict[str, object]) -> tuple[torch.Tensor, ...]:
    query = case["query"]
    assert isinstance(query, torch.Tensor)
    total_queries = query.size(0)
    batch = len(case["q_values"])
    device = query.device
    return (
        torch.full((total_queries, 1, TOPK), -313, dtype=torch.int32, device=device),
        torch.full((total_queries, 1, TOPK), -313, dtype=torch.int32, device=device),
        torch.full((total_queries,), -313, dtype=torch.int32, device=device),
        torch.full((batch, MISS_CAPACITY), -313, dtype=torch.int32, device=device),
        torch.full((batch, MISS_CAPACITY), -313, dtype=torch.int32, device=device),
        torch.full((batch,), -313, dtype=torch.int32, device=device),
    )


def call_custom(
    case: dict[str, object],
    cache: torch.Tensor,
    outputs: tuple[torch.Tensor, ...],
) -> None:
    query_ends, actual_key, offload_key, cache_tokens, states, req_entries = case[
        "metadata"
    ]
    torch.ops.nanovllm_dsa.fused_li_manage_mtp.default(
        case["weights"],
        case["query_scale"],
        case["query"],
        case["key_scale"],
        case["key"],
        case["block_table"],
        query_ends,
        actual_key,
        offload_key,
        cache_tokens,
        states,
        req_entries,
        cache,
        *outputs,
    )


def visible_lengths(case: dict[str, object]) -> list[int]:
    result: list[int] = []
    for request, q in enumerate(case["q_values"]):
        state = case["states"][request]
        for route in range(q):
            result.append(
                case["actual_key"][request] - (q - 1 - route)
                if state == -3
                else case["offload_key"][request]
            )
    return result


def assert_correctness(case: dict[str, object]) -> None:
    cache = case["cache_seed"].clone()
    old_cache = cache.clone()
    outputs = make_outputs(case)
    reference = native_topk(
        case["query"], case["key"], case["weights"],
        case["route_table"], visible_lengths(case)
    ).cpu()
    call_custom(case, cache, outputs)
    torch.npu.synchronize()
    src, dst, route_miss, miss_src, miss_dst, miss_count = [x.cpu() for x in outputs]
    cache_cpu = cache.cpu()
    query_start = 0
    for request, q in enumerate(case["q_values"]):
        query_end = query_start + q
        state = case["states"][request]
        row = case["req_entries"][request]
        length = case["actual_key"][request] if state == -3 else case["offload_key"][request]
        for route in range(query_start, query_end):
            valid = min(length, TOPK)
            actual_topk = src[route, 0, :valid]
            expected_topk = reference[route, :valid]
            if state == -3:
                if not torch.equal(actual_topk, expected_topk):
                    mismatch_positions = torch.nonzero(
                        actual_topk != expected_topk, as_tuple=False
                    ).flatten()
                    first_mismatch = int(mismatch_positions[0])
                    window_start = max(0, first_mismatch - 8)
                    window_end = min(valid, first_mismatch + 9)
                    same_members = torch.equal(
                        torch.sort(actual_topk).values,
                        torch.sort(expected_topk).values,
                    )
                    raise AssertionError(
                        "ordered TopK mismatch: "
                        f"request={request}, q={q}, route={route}, "
                        f"route_in_request={route - query_start}, valid={valid}, "
                        f"first_position={first_mismatch}, "
                        f"same_members_after_sort={same_members}, "
                        f"actual_window[{window_start}:{window_end}]="
                        f"{actual_topk[window_start:window_end].tolist()}, "
                        f"expected_window[{window_start}:{window_end}]="
                        f"{expected_topk[window_start:window_end].tolist()}"
                    )
            else:
                assert torch.equal(
                    torch.sort(actual_topk).values,
                    torch.sort(expected_topk).values,
                ), f"TopK mismatch at route={route}"
            if valid < TOPK:
                assert torch.all(src[route, 0, valid:] == -1)
            if state == -3:
                assert route_miss[route].item() == 0, (
                    f"non-offload route miss count mismatch: request={request}, "
                    f"route={route}, value={route_miss[route].item()}, "
                    f"all={route_miss.tolist()}, states={case['states']}"
                )
                assert torch.equal(dst[route], src[route])
            else:
                for position in range(TOPK):
                    source = int(src[route, 0, position])
                    if source >= 0:
                        actual_slot = int(dst[route, 0, position])
                        expected_slot = int(cache_cpu[row, source])
                        if actual_slot != expected_slot:
                            old_slot = int(old_cache.cpu()[row, source])
                            route_misses = int(route_miss[route])
                            raise AssertionError(
                                "TopK destination mismatch: "
                                f"request={request}, q={q}, route={route}, "
                                f"route_in_request={route - query_start}, "
                                f"position={position}, source={source}, "
                                f"actual_slot={actual_slot}, "
                                f"final_cache_slot={expected_slot}, "
                                f"initial_cache_slot={old_slot}, "
                                f"route_misses={route_misses}, "
                                f"state={state}, L={length}, "
                                f"C={case['cache_tokens'][request]}"
                            )
        if state == -3:
            assert miss_count[request].item() == 0, (
                f"non-offload request miss count mismatch: request={request}, "
                f"value={miss_count[request].item()}, all={miss_count.tolist()}, "
                f"q={case['q_values']}, states={case['states']}"
            )
            assert torch.equal(
                cache_cpu[row], torch.arange(cache_cpu.size(1), dtype=torch.int32)
            )
        elif state == -2:
            count = case["cache_tokens"][request]
            assert miss_count[request].item() == count
            assert torch.equal(miss_dst[request, :count], torch.arange(count, dtype=torch.int32))
            union = torch.unique(src[query_start:query_end].reshape(-1), sorted=True)
            union = union[union >= 0]
            selected = torch.zeros(length, dtype=torch.bool)
            selected[union.to(torch.int64)] = True
            remainder = torch.arange(length, dtype=torch.int32)[~selected]
            expected = torch.cat((union, remainder))[:count]
            assert torch.equal(miss_src[request, :count], expected)
            assert torch.all(route_miss[query_start:query_end] == TOPK)
        else:
            union = torch.unique(src[query_start:query_end].reshape(-1), sorted=True)
            expected = union[old_cache.cpu()[row, union.to(torch.int64)] < 0]
            count = int(miss_count[request])
            assert count == expected.numel()
            assert torch.equal(miss_src[request, :count], expected)
        if state in (-2, -1):
            resident_slots = cache_cpu[row, :length]
            resident_slots = resident_slots[resident_slots >= 0]
            capacity = case["cache_tokens"][request]
            if resident_slots.numel() != capacity:
                valid_slots = resident_slots[
                    (resident_slots >= 0) & (resident_slots < capacity)
                ]
                slot_counts = torch.bincount(
                    valid_slots.to(torch.int64), minlength=capacity
                )
                missing = torch.nonzero(slot_counts == 0, as_tuple=False).flatten()
                duplicate = torch.nonzero(slot_counts > 1, as_tuple=False).flatten()
                raise AssertionError(
                    "cache resident count mismatch: "
                    f"request={request}, q={q}, state={state}, L={length}, "
                    f"C={capacity}, resident={resident_slots.numel()}, "
                    f"miss_count={int(miss_count[request])}, "
                    f"missing_slots={missing[:16].tolist()}, "
                    f"duplicate_slots={duplicate[:16].tolist()}"
                )
            assert torch.equal(
                torch.sort(resident_slots).values,
                torch.arange(capacity, dtype=torch.int32),
            ), f"cache slot mapping is not a bijection for request={request}"
        query_start = query_end


def run_correctness(args: argparse.Namespace) -> None:
    q_values = [int(value) for value in args.q_pattern.split(",")]
    state_patterns = [
        [-3] * len(q_values),
        [-2] * len(q_values),
        [-1] * len(q_values),
        [(-3, -2, -1)[index % 3] for index in range(len(q_values))],
    ]
    for states in state_patterns:
        case = build_case(args, q_values=q_values, states=states)
        assert_correctness(case)
        print(f"correctness PASS q={q_values} states={states} dtype={args.dtype}")


def run_replacement_regression(args: argparse.Namespace) -> None:
    """Exercise -1 victim replacement, which L == C cannot cover."""

    # Every row begins as a valid [0, C) mapping.  L > C supplies real cold
    # sources, forcing the kernel to invalidate a victim and reuse its slot.
    # The scenarios cover the Q=1 fast path, mixed generic routes, the mature
    # MTP3 route, the seven-route local path and wider GM-backed requests.
    scenarios = (
        ([1], 8320, 8192),
        ([1, 2, 3], 8320, 8192),
        ([4], 16256, 12288),
        ([7], 16256, 14336),
        ([8], 16512, 16384),
        ([12], 24704, 24576),
        ([14], 32768, 32640),
    )
    for q_values, offload_len, cache_tokens in scenarios:
        scenario_args = argparse.Namespace(**vars(args))
        scenario_args.source_capacity = offload_len + BLOCK
        scenario_args.offload_len = offload_len
        scenario_args.cache_tokens = cache_tokens
        case = build_case(
            scenario_args,
            q_values=q_values,
            states=[-1] * len(q_values),
        )
        assert_correctness(case)
        print(
            "replacement regression PASS "
            f"q={q_values} state=-1 L={offload_len} C={cache_tokens} "
            f"dtype={args.dtype}"
        )


def run_first_decode_regression(args: argparse.Namespace) -> None:
    """Cover -2 initialization when L > C, including identity-row reuse."""

    # The normal smoke case commonly uses L == C.  These cases require the
    # first-decode path to select exactly C sources and publish the complete
    # miss_src_ids/miss_dst_slots transfer lists.  They also cover cache
    # budgets above 8192, where an oversized Arange previously corrupted the
    # destination identity list.
    scenarios = (
        ([1], 8320, 8192),
        ([1, 2, 3], 8320, 8192),
        ([4], 16256, 12288),
        ([7], 16256, 14336),
        ([8], 16512, 16384),
        ([9], 18560, 18432),
        ([12], 24704, 24576),
        ([14], 32768, 32640),
    )
    for q_values, offload_len, cache_tokens in scenarios:
        scenario_args = argparse.Namespace(**vars(args))
        scenario_args.source_capacity = offload_len + BLOCK
        scenario_args.offload_len = offload_len
        scenario_args.cache_tokens = cache_tokens

        cold_case = build_case(
            scenario_args, q_values=q_values, states=[-2] * len(q_values)
        )
        assert_correctness(cold_case)
        print(
            "first-decode regression PASS "
            f"q={q_values} state=-2 L={offload_len} C={cache_tokens} "
            f"seed=invalid dtype={args.dtype}"
        )

        # Repeat from a -3 identity pool row.  This is the lifecycle edge
        # where stale identity mappings must be fully replaced before -2
        # publishes the selected resident set and transfer lists.
        standard_case = build_case(
            scenario_args, q_values=q_values, states=[-3] * len(q_values)
        )
        transition_cache = standard_case["cache_seed"].clone()
        call_custom(standard_case, transition_cache, make_outputs(standard_case))
        torch.npu.synchronize()
        transition_case = build_case(
            scenario_args, q_values=q_values, states=[-2] * len(q_values)
        )
        transition_case["cache_seed"] = transition_cache
        assert_correctness(transition_case)
        print(
            "first-decode regression PASS "
            f"q={q_values} state=-3->-2 L={offload_len} C={cache_tokens} "
            f"dtype={args.dtype}"
        )


def run_lifecycle(args: argparse.Namespace) -> None:
    _run_lifecycle_scenario(args, q_values=[1, 4, 7])


def run_wide_route_regression(args: argparse.Namespace) -> None:
    """Exercise every Q8--Q14 route count on the shared generalized path."""

    scenarios = (
        (8, 16384, 16512),
        (9, 18432, 18560),
        (10, 20480, 20608),
        (11, 22528, 22656),
        (12, 24576, 24704),
        (13, 26624, 26752),
        (14, 32640, 32768),
    )
    for q, cache_tokens, offload_len in scenarios:
        scenario_args = argparse.Namespace(**vars(args))
        scenario_args.cache_tokens = cache_tokens
        scenario_args.offload_len = offload_len
        scenario_args.source_capacity = offload_len + BLOCK
        for state in (-3, -2, -1):
            case = build_case(scenario_args, q_values=[q], states=[state])
            assert_correctness(case)
            print(
                "wide route regression PASS "
                f"q={q} state={state} L={offload_len} C={cache_tokens} "
                f"dtype={args.dtype}"
            )

    mixed_args = argparse.Namespace(**vars(args))
    mixed_args.cache_tokens = 32640
    mixed_args.offload_len = 32768
    mixed_args.source_capacity = 32896
    mixed_case = build_case(
        mixed_args, q_values=[8, 12, 14], states=[-3, -2, -1]
    )
    assert_correctness(mixed_case)
    _run_lifecycle_scenario(mixed_args, q_values=[8, 12, 14])
    print(
        "wide mixed regression PASS q=[8, 12, 14] "
        "states=[-3, -2, -1] C=32640"
    )

    # Worst-case MTP13: fourteen disjoint cold TopK rows produce an exact
    # 28,672-source union while the cache retains the 32,640 product budget.
    stress_args = argparse.Namespace(**vars(args))
    stress_args.cache_tokens = 32640
    stress_args.offload_len = 32640 + 14 * TOPK
    stress_args.source_capacity = stress_args.offload_len + BLOCK
    stress_case = build_case(stress_args, q_values=[14], states=[-1])
    query = stress_case["query"]
    weights = stress_case["weights"]
    key = stress_case["key"]
    assert isinstance(query, torch.Tensor)
    assert isinstance(weights, torch.Tensor)
    assert isinstance(key, torch.Tensor)
    query.zero_()
    weights.fill_(1)
    key.zero_()
    key_by_source = key.reshape(
        stress_args.source_capacity, 1, HEAD_DIM
    )
    for route in range(14):
        query[route, :, route].fill_(1)
        begin = 32640 + route * TOPK
        key_by_source[begin : begin + TOPK, 0, route].fill_(1)
    reference = native_topk(
        query, key, weights, stress_case["route_table"],
        visible_lengths(stress_case),
    )
    union = torch.unique(reference.cpu().reshape(-1))
    if union.numel() != 14 * TOPK or int(union.min()) < 32640:
        raise AssertionError(
            "MTP13 stress construction did not produce 28672 disjoint "
            "cold TopK sources"
        )
    assert_correctness(stress_case)
    print(
        "wide disjoint regression PASS q=14 L=61312 C=32640 "
        "route_miss=2048 union=28672"
    )


def run_wide_compact_union_regression(args: argparse.Namespace) -> None:
    """Port the backup suite's ordinary wide-union workload through Q14."""

    scenarios = ((8, 16384, 1600), (12, 24576, 2400),
                 (14, 32640, 3000))
    for q, cache_tokens, target_union in scenarios:
        scenario_args = argparse.Namespace(**vars(args))
        scenario_args.cache_tokens = cache_tokens
        scenario_args.offload_len = cache_tokens + 4096
        scenario_args.source_capacity = scenario_args.offload_len + BLOCK
        case = build_case(scenario_args, q_values=[q], states=[-1])
        reference = native_topk(
            case["query"], case["key"], case["weights"],
            case["route_table"], visible_lengths(case)
        )
        generator = torch.Generator().manual_seed(args.seed + q * 65537)
        cache, actual_union = _controlled_union_cache(
            case, reference, target_union, generator
        )
        if actual_union != target_union:
            raise AssertionError(
                f"wide compact construction q={q} requested union="
                f"{target_union}, got {actual_union}"
            )
        row = int(case["req_entries"][0])
        cache_cpu = cache.cpu()[row]
        route_misses = []
        for route in reference.cpu():
            sources = route.to(torch.int64)
            route_misses.append(int((cache_cpu[sources] < 0).sum()))
        occurrence_total = sum(route_misses)
        if occurrence_total > WIDE_ORDINARY_OCCURRENCE_LIMIT:
            raise AssertionError(
                f"wide compact construction q={q} occurrence="
                f"{occurrence_total} exceeds retained workload limit"
            )
        case["cache_seed"] = cache
        assert_correctness(case)
        print(
            "wide compact union regression PASS "
            f"q={q} L={scenario_args.offload_len} C={cache_tokens} "
            f"union={actual_union} occurrence={occurrence_total} "
            f"route_misses={route_misses} dtype={args.dtype}"
        )


def _run_lifecycle_scenario(
    args: argparse.Namespace, *, q_values: list[int]
) -> None:
    for sequence in ((-2, -1, -1), (-3, -1), (-3, -2, -1)):
        cache: torch.Tensor | None = None
        last_outputs: tuple[torch.Tensor, ...] | None = None
        for state in sequence:
            case = build_case(args, q_values=q_values, states=[state] * len(q_values))
            if cache is None:
                cache = case["cache_seed"].clone()
            outputs = make_outputs(case)
            call_custom(case, cache, outputs)
            torch.npu.synchronize()
            last_outputs = outputs
        assert last_outputs is not None
        if sequence[-2:] == (-1, -1):
            assert torch.all(last_outputs[2] == 0)
            assert torch.all(last_outputs[5] == 0)
        print(f"lifecycle PASS q={q_values} sequence={sequence}")


def _long_scenario_args(
    args: argparse.Namespace, *, cache_tokens: int
) -> argparse.Namespace:
    """Return one long-source scenario using the caller's L and physical row."""

    if args.offload_len <= PACKED_SOURCE_CAPACITY:
        raise ValueError(
            "long-regression requires --offload-len > 2^17; "
            "use correctness/replacement-regression for short sources"
        )
    if args.source_capacity != args.offload_len + BLOCK:
        raise ValueError(
            "long-regression requires --source-capacity == --offload-len + 128"
        )
    scenario_args = argparse.Namespace(**vars(args))
    scenario_args.cache_tokens = cache_tokens
    return scenario_args


def _assert_long_source_coverage(case: dict[str, object]) -> None:
    """Require this case to actually exercise the high four source-ID bits."""

    reference = native_topk(
        case["query"], case["key"], case["weights"],
        case["route_table"], visible_lengths(case)
    )
    if not bool(torch.any(reference >= PACKED_SOURCE_CAPACITY)):
        raise AssertionError(
            "long regression did not select a source above 2^17; "
            "cannot validate the 21-bit source-ID path"
        )


def _tagged_score_bits(score_bits: int, tag: int) -> int:
    """Reference the 4-bit source-high tag written into an FP32 sort key."""

    return (score_bits & ~0xF) | tag


def run_payload_codec_regression(args: argparse.Namespace) -> None:
    """Check every boundary of the 15/17-bit internal sort payload codec."""

    del args
    sources = (
        (1 << 17) - 1,
        1 << 17,
        1 << 18,
        1 << 20,
        (1 << 21) - 1,
    )
    for source in sources:
        tag = source >> 17
        assert 0 <= tag <= 15
        for slot in (0, 16383, 16384, 30719, 32639, 32766):
            payload = (slot << 17) | (source & ((1 << 17) - 1))
            decoded_source = (tag << 17) | (payload & ((1 << 17) - 1))
            decoded_slot = payload >> 17
            assert decoded_source == source
            assert decoded_slot == slot

    base_key = 0x3F800000
    for tag in range(16):
        assert (_tagged_score_bits(base_key, tag) & 0xF) == tag
    assert not (131072 > PACKED_SOURCE_CAPACITY)
    assert 131200 > PACKED_SOURCE_CAPACITY

    for pool_value in (INVALID_SLOT, -1):
        packed_slot = PACKED_INVALID_SLOT if pool_value < 0 else pool_value
        payload = packed_slot << 17
        decoded = payload >> 17
        output_slot = -1 if decoded == PACKED_INVALID_SLOT else decoded
        assert output_slot == -1

    reserved_payload = PACKED_INVALID_SLOT << 17
    assert (reserved_payload >> 17) == 32767
    print(
        "payload codec regression PASS "
        "layout=slot15/source17 valid_slot_max=32766 "
        "future_product_cache_max=32640 invalid_slot=32767 source_bits=21"
    )


def _run_key_tag_ulp_boundary() -> None:
    """Diagnose the mathematical boundary of the current 4-bit key tag.

    The Ascend input tensors are BF16/FP16, so they cannot directly inject
    arbitrary adjacent FP32 score values.  This host-side check therefore
    enumerates raw FP32 key bits around a cutoff.  It documents exactly when
    replacing the low four key bits with source_high4 can reorder two nearly
    equal candidates; the NPU half below verifies the production LI path.
    """

    base = 0x3F800000  # 1.0f; one increment is one positive FP32 ULP.
    reordered = 0
    pairs = 0
    for delta in range(17):
        # Exact ordering is source A then B on equal scores, and B then A for
        # a positive delta.  Use opposite source-high tags to maximize the
        # possible four-bit replacement while retaining a deterministic
        # source-ID tie break in the reference ordering.
        score_a = base
        score_b = base + delta
        source_a = 0
        source_b = PACKED_SOURCE_CAPACITY
        exact_a_first = (
            score_a > score_b
            or (score_a == score_b and source_a < source_b)
        )
        tagged_a = _tagged_score_bits(score_a, 15)
        tagged_b = _tagged_score_bits(score_b, 0)
        tagged_a_first = tagged_a > tagged_b
        pairs += 1
        reordered += int(exact_a_first != tagged_a_first)

    print(
        "FUSED_LI_MANAGE_MTP_KEY_TAG_ULP_DIAG "
        "tag_bits=4 ulp_deltas=0..16 "
        f"pairs={pairs} reordered_pairs={reordered} "
        "max_key_delta_ulp=15"
    )


def _build_key_tag_tie_case(args: argparse.Namespace) -> dict[str, object]:
    """Build an end-to-end long-source TopK cutoff with exact tied scores.

    A 1024-token group below source ID 2^17 and a 1152-token group above it
    receive identical maximal LI scores while every other source receives a
    lower score.  The 2048-token cutoff must therefore retain sources from
    both source_high4 values, and assert_correctness compares the ordered
    result against the official LI output.
    """

    scenario_args = argparse.Namespace(**vars(args))
    upper_candidate_count = TOPK // 2 + BLOCK
    scenario_args.source_capacity = PACKED_SOURCE_CAPACITY + upper_candidate_count
    scenario_args.offload_len = scenario_args.source_capacity - BLOCK
    scenario_args.cache_tokens = 8192
    case = build_case(scenario_args, q_values=[1], states=[-3])

    query = case["query"]
    weights = case["weights"]
    key = case["key"]
    assert isinstance(query, torch.Tensor)
    assert isinstance(weights, torch.Tensor)
    assert isinstance(key, torch.Tensor)

    query.fill_(1)
    weights.fill_(1)
    key_by_source = key.reshape(scenario_args.source_capacity, 1, HEAD_DIM)
    boundary = PACKED_SOURCE_CAPACITY
    candidate_begin = boundary - TOPK // 2
    candidate_end = boundary + upper_candidate_count

    # LI's ranking direction is an implementation detail of the official
    # kernel.  Probe both signed extremes and retain the one for which the
    # official TopK is drawn solely from the 2176 equal-score candidates.
    # Since only 1024 lie below 2^17, the final 2048-token cutoff necessarily
    # crosses the source-high boundary instead of validating only tag zero.
    selected_value: int | None = None
    for candidate_value in (1, -1):
        key.zero_()
        key_by_source[candidate_begin:candidate_end].fill_(candidate_value)
        reference = native_topk(
            case["query"], case["key"], case["weights"],
            case["route_table"], visible_lengths(case)
        )
        selected = reference[0]
        if (
            bool(torch.all((selected >= candidate_begin) & (selected < candidate_end)))
            and bool(torch.any(selected >= boundary))
        ):
            selected_value = candidate_value
            break
    if selected_value is None:
        raise AssertionError(
            "key-tag regression could not force the official TopK cutoff "
            "into the cross-2^17 equal-score candidate range"
        )
    case["key_tag_candidate_value"] = selected_value
    return case


def run_key_tag_regression(args: argparse.Namespace) -> None:
    """Validate 4-bit long-source score tagging at the ordered-TopK cutoff."""

    _run_key_tag_ulp_boundary()
    case = _build_key_tag_tie_case(args)
    _assert_long_source_coverage(case)
    assert_correctness(case)
    print(
        "key-tag regression PASS "
        f"dtype={args.dtype} source_capacity={case['key'].size(0) * BLOCK} "
        f"boundary={PACKED_SOURCE_CAPACITY} candidates={TOPK // 2 + TOPK // 2 + BLOCK} "
        f"candidate_value={case['key_tag_candidate_value']}"
    )


def run_long_regression(args: argparse.Namespace) -> None:
    """Exercise 21-bit IDs across all stateful long-source correctness edges.

    This intentionally remains separate from ``all``.  The short fixed L>C
    regressions and Q1/4/7/8/12/14 use different cache budgets, whereas
    a long physical row needs one explicit (L, C) pair per invocation.
    """

    # This fixed 2^17-boundary cutoff test is independent of the caller's L/C
    # pair and must accompany every long-source acceptance run.
    run_payload_codec_regression(args)
    run_key_tag_regression(args)
    run_occurrence_mapping_regression(args)

    scenarios = (
        ([1], 8192), ([4], 12288), ([7], 14336),
        ([8], 16384), ([12], 24576), ([14], 32640),
    )
    for q_values, cache_tokens in scenarios:
        scenario_args = _long_scenario_args(args, cache_tokens=cache_tokens)

        # Cover all three independent states with the production cache budget.
        for states in ([-3], [-2], [-1]):
            case = build_case(scenario_args, q_values=q_values, states=states)
            _assert_long_source_coverage(case)
            assert_correctness(case)
            print(
                "long correctness PASS "
                f"q={q_values} states={states} L={args.offload_len} "
                f"C={cache_tokens} dtype={args.dtype}"
            )

        # Exercise real L>C steady replacement, including high-ID incoming
        # sources, and first-decode from both an invalid and an identity row.
        replacement_case = build_case(
            scenario_args, q_values=q_values, states=[-1]
        )
        _assert_long_source_coverage(replacement_case)
        assert_correctness(replacement_case)
        print(
            "long replacement regression PASS "
            f"q={q_values} L={args.offload_len} C={cache_tokens}"
        )

        first_case = build_case(scenario_args, q_values=q_values, states=[-2])
        _assert_long_source_coverage(first_case)
        assert_correctness(first_case)

        standard_case = build_case(
            scenario_args, q_values=q_values, states=[-3]
        )
        transition_cache = standard_case["cache_seed"].clone()
        call_custom(standard_case, transition_cache, make_outputs(standard_case))
        torch.npu.synchronize()
        transition_case = build_case(
            scenario_args, q_values=q_values, states=[-2]
        )
        transition_case["cache_seed"] = transition_cache
        _assert_long_source_coverage(transition_case)
        assert_correctness(transition_case)
        print(
            "long first-decode regression PASS "
            f"q={q_values} states=-3->-2 L={args.offload_len} C={cache_tokens}"
        )

        _run_lifecycle_scenario(scenario_args, q_values=q_values)

    # One batch with different Q and states verifies that 21-bit source IDs do
    # not only work in homogeneous fast paths.  C=14336 is valid for every
    # request here and exercises the full MTP13 product budget.
    mixed_args = _long_scenario_args(args, cache_tokens=32640)
    mixed_case = build_case(
        mixed_args,
        q_values=[1, 4, 7, 8, 12, 14],
        states=[-3, -2, -1, -3, -2, -1],
    )
    _assert_long_source_coverage(mixed_case)
    assert_correctness(mixed_case)
    print(
        "long mixed correctness PASS q=[1, 4, 7, 8, 12, 14] "
        "states=[-3, -2, -1, -3, -2, -1] "
        f"L={args.offload_len} C=32640 dtype={args.dtype}"
    )

    # Boundary-only scheduler validation: this does not allocate a 2 Mi-token
    # key tensor, but verifies the externally visible 21-bit capacity limit.
    base: dict[str, object] = dict(
        query_ends=[4], actual_key=[MAX_SOURCE_CAPACITY],
        offload_key=[MAX_SOURCE_CAPACITY - BLOCK], cache_tokens=[12288],
        request_state=[-1], req_pool_entries=[0], total_queries=4,
        source_capacity=MAX_SOURCE_CAPACITY, pool_size=2,
    )
    validate_dynamic_inputs(**base)
    for bad_capacity in (0, MAX_SOURCE_CAPACITY - 1, MAX_SOURCE_CAPACITY + BLOCK):
        expect_value_error(base, "source_capacity", bad_capacity)
    expect_value_error(base, "offload_key", [MAX_SOURCE_CAPACITY])
    print("long invalid PASS cases=4")


def run_occurrence_mapping_regression(args: argparse.Namespace) -> None:
    """Cover Q1--Q14 occurrence counts and publication boundaries."""

    # Keep both the searched prefix and its required +128 physical/causal row
    # within the direct 17-bit source representation.  Long-source key-tag
    # perturbation has dedicated coverage in key-tag/long regression and must
    # not change the official TopK used to construct this occurrence fixture.
    source_len = PACKED_SOURCE_CAPACITY - BLOCK
    scenarios = (
        # Route-count coverage with small and intermediate miss sets.
        (1, 8192, 127),    # 127
        (2, 8192, 512),    # 1024
        (3, 8192, 683),    # 2049
        # Actual compact occurrence-map boundary.
        (4, 12288, 1023),  # 4092: compact
        (4, 12288, 1024),  # 4096: compact boundary
        (4, 12288, 1025),  # 4100: exact source-join fallback
        # Retain focused coverage around the backup suite's historical 6912
        # workload boundary; all three are exact source-join cases now.
        (4, 12288, 1727),  # 6908
        (4, 12288, 1728),  # 6912
        (4, 12288, 1729),  # 6916
        (4, 12288, 1840),
        (5, 12288, 1400),
        (6, 12288, 683),   # 4098
        (7, 14336, 1100),
        # Every wider Q is exercised on the same generalized path.  Values
        # alternate around the 4096 and historical 6912 occurrence points.
        (8, 16384, 511),   # 4088
        (9, 18432, 456),   # 4104
        (10, 20480, 690),  # 6900
        (11, 22528, 628),  # 6908
        (12, 24576, 576),  # 6912
        (13, 26624, 532),  # 6916
        (14, 32640, 494),  # 6916
    )
    for index, (q, cache_tokens, per_route_misses) in enumerate(scenarios):
        union_misses = per_route_misses * q
        scenario_args = argparse.Namespace(**vars(args))
        scenario_args.source_capacity = source_len + BLOCK
        scenario_args.offload_len = source_len
        scenario_args.cache_tokens = cache_tokens
        case = build_case(scenario_args, q_values=[q], states=[-1])
        generator = torch.Generator().manual_seed(
            args.seed + 271828 + index * 997
        )
        _randomize_semantic_block_table(case, generator)
        reference = native_topk(
            case["query"], case["key"], case["weights"],
            case["route_table"], visible_lengths(case),
        )
        seed, actual_unions, route_misses = _homogeneous_controlled_cache(
            reference.cpu(), q=q, batch=1, source_len=source_len,
            source_capacity=source_len + BLOCK, cache_tokens=cache_tokens,
            req_entries=case["req_entries"],
            per_query_misses=per_route_misses,
            union_misses=union_misses, generator=generator,
        )
        if (actual_unions != [union_misses] or
                route_misses != [per_route_misses] * q):
            raise AssertionError(
                "occurrence regression cache construction did not preserve "
                f"the requested Q{q} miss shape"
            )
        query = case["query"]
        assert isinstance(query, torch.Tensor)
        case["cache_seed"] = _force_topk_hit_slot_zero(
            case, reference, seed
        ).to(query.device)
        updated_cache, _, observed_unions = _validate_strict_steady_semantics(
            case, reference
        )
        if observed_unions != [union_misses]:
            raise AssertionError(
                "occurrence regression observed union mismatch: "
                f"actual={observed_unions}, expected={[union_misses]}"
            )
        repeat_case = dict(case)
        repeat_case["cache_seed"] = updated_cache.clone()
        _, repeat_outputs, repeat_unions = _validate_strict_steady_semantics(
            repeat_case, reference
        )
        if (repeat_unions != [0] or bool(torch.any(repeat_outputs[2] != 0))
                or bool(torch.any(repeat_outputs[5] != 0))):
            raise AssertionError(
                "occurrence regression did not converge on repeat"
            )
        print(
            f"occurrence regression PASS q=[{q}] L={source_len} "
            f"C={cache_tokens} per_route_misses={per_route_misses} "
            f"union={union_misses} random_block_table=1 repeat_stable=1 "
            f"dtype={args.dtype}"
        )

    # Maximum disjoint Q4/Q7/Q14 unions validate source-join publication
    # fallback beyond the compact occurrence map and at maximum supported Q.
    for q, cache_tokens in ((4, 12288), (7, 14336), (14, 32640)):
        source_len = cache_tokens + q * TOPK
        stress_args = argparse.Namespace(**vars(args))
        stress_args.cache_tokens = cache_tokens
        stress_args.offload_len = source_len
        stress_args.source_capacity = source_len + BLOCK
        stress_case = build_case(stress_args, q_values=[q], states=[-1])
        query = stress_case["query"]
        weights = stress_case["weights"]
        key = stress_case["key"]
        assert isinstance(query, torch.Tensor)
        assert isinstance(weights, torch.Tensor)
        assert isinstance(key, torch.Tensor)
        query.zero_()
        weights.fill_(1)
        key.zero_()
        key_by_source = key.reshape(
            stress_args.source_capacity, 1, HEAD_DIM
        )
        for route in range(q):
            query[route, :, route].fill_(1)
            begin = cache_tokens + route * TOPK
            key_by_source[begin : begin + TOPK, 0, route].fill_(1)
        reference = native_topk(
            query, key, weights, stress_case["route_table"],
            visible_lengths(stress_case),
        )
        union = torch.unique(reference.cpu().reshape(-1), sorted=True)
        if union.numel() != q * TOPK or int(union.min()) < cache_tokens:
            raise AssertionError(
                f"Q{q} disjoint occurrence fallback construction failed"
            )
        assert_correctness(stress_case)
        print(
            f"occurrence fallback regression PASS q={q} L={source_len} "
            f"C={cache_tokens} route_miss=2048 union={q * TOPK}"
        )


def expect_value_error(base: dict[str, object], field: str, value: object) -> None:
    kwargs = dict(base)
    kwargs[field] = value
    try:
        validate_dynamic_inputs(**kwargs)
    except ValueError:
        return
    raise AssertionError(f"dynamic validator accepted invalid field {field}={value}")


def run_invalid(args: argparse.Namespace) -> None:
    base: dict[str, object] = dict(
        query_ends=[4], actual_key=[8320], offload_key=[8192],
        cache_tokens=[8192], request_state=[-1], req_pool_entries=[0],
        total_queries=4, source_capacity=16384, pool_size=2,
    )
    invalids = (
        ("query_ends", [0]), ("query_ends", [15]),
        ("actual_key", [3]), ("actual_key", [20000]),
        ("offload_key", [2047]), ("offload_key", [8256]),
        ("cache_tokens", [2000]), ("cache_tokens", [8064]),
        ("cache_tokens", [32768]),
        ("request_state", [0]), ("request_state", [-4]),
        ("req_pool_entries", [-1]), ("req_pool_entries", [2]),
        ("total_queries", 5),
    )
    for field, value in invalids:
        expect_value_error(base, field, value)
    # Host-side representative shape/dtype rejection.
    case = build_case(args, q_values=[4], states=[-1])
    outputs = make_outputs(case)
    bad_case = dict(case)
    bad_case["query_scale"] = case["query_scale"].to(torch.float16)
    try:
        call_custom(bad_case, case["cache_seed"].clone(), outputs)
    except RuntimeError:
        pass
    else:
        raise AssertionError("Host accepted fp16 query_dequant_scale")

    old_abi_outputs = list(make_outputs(case))
    old_abi_outputs[3] = torch.empty(
        (1, 16384), dtype=torch.int32, device=case["query"].device
    )
    old_abi_outputs[4] = torch.empty_like(old_abi_outputs[3])
    try:
        call_custom(case, case["cache_seed"].clone(), tuple(old_abi_outputs))
    except RuntimeError:
        pass
    else:
        raise AssertionError("Host accepted legacy [B,16384] miss outputs")

    q14_case = build_case(args, q_values=[14], states=[-3])
    q15_case = dict(q14_case)
    for field in ("query", "weights", "query_scale"):
        tensor = q14_case[field]
        assert isinstance(tensor, torch.Tensor)
        q15_case[field] = torch.cat((tensor, tensor[:1]), dim=0)
    route_table = q14_case["route_table"]
    assert isinstance(route_table, torch.Tensor)
    q15_case["route_table"] = torch.cat((route_table, route_table[:1]), dim=0)
    q15_case["q_values"] = [15]
    q15_case["query_ends"] = [15]
    metadata = list(q14_case["metadata"])
    metadata[0] = torch.tensor(
        [15], dtype=torch.int32, device=case["query"].device
    )
    q15_case["metadata"] = tuple(metadata)
    try:
        call_custom(
            q15_case, q15_case["cache_seed"].clone(), make_outputs(q15_case)
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("Host accepted Q=15")
    print(f"invalid PASS cases={len(invalids) + 3}")


def event_us(fn, *, warmup: int, iters: int, setup=None) -> tuple[float, float]:
    for _ in range(warmup):
        if setup is not None:
            setup()
        fn()
    torch.npu.synchronize()
    samples: list[float] = []
    for _ in range(iters):
        if setup is not None:
            setup()
            torch.npu.synchronize()
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)) * 1000.0)
    ordered = sorted(samples)
    p95 = ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))]
    return statistics.median(samples), p95


def run_transition_perf(args: argparse.Namespace) -> None:
    """Compare steady -1 with identity-row to -1 conversion latency."""
    requested_q = [int(value) for value in args.q_pattern.split(",")]
    if not requested_q or any(q < 1 or q > MAX_ROUTES for q in requested_q):
        raise ValueError("transition-perf requires Q values in [1,14]")
    if len(requested_q) == 1:
        if args.batch_size <= 0:
            raise ValueError("--batch-size must be positive")
        q_values = requested_q * args.batch_size
    else:
        q_values = requested_q

    case = build_case(args, q_values=q_values, states=[-1] * len(q_values))
    steady_seed = case["cache_seed"]
    identity_seed = torch.full_like(steady_seed, INVALID_SLOT)
    identity_row = torch.arange(
        args.source_capacity, dtype=torch.int32, device=identity_seed.device
    )
    for row in case["req_entries"]:
        identity_seed[row].copy_(identity_row)

    cache = steady_seed.clone()
    outputs = make_outputs(case)

    def invoke() -> None:
        call_custom(case, cache, outputs)

    steady_median, steady_p95 = event_us(
        invoke, warmup=args.warmup, iters=args.iters,
        setup=lambda: cache.copy_(steady_seed),
    )
    transition_median, transition_p95 = event_us(
        invoke, warmup=args.warmup, iters=args.iters,
        setup=lambda: cache.copy_(identity_seed),
    )
    print(
        "FUSED_LI_MANAGE_MTP_TRANSITION_PERF "
        f"q={q_values} batch={len(q_values)} heads={args.heads} "
        f"source_capacity={args.source_capacity} "
        f"offload_len={args.offload_len} "
        f"cache_tokens={case['cache_tokens']} "
        f"steady_median_us={steady_median:.3f} "
        f"transition_median_us={transition_median:.3f} "
        f"transition_overhead_us={transition_median - steady_median:+.3f} "
        f"steady_p95_us={steady_p95:.3f} "
        f"transition_p95_us={transition_p95:.3f} "
        f"warmup={args.warmup} iters={args.iters}",
        flush=True,
    )


def _with_request_states(
    case: dict[str, object], states: list[int]
) -> dict[str, object]:
    """Reuse a case's payload while replacing only request-state metadata."""
    if len(states) != len(case["q_values"]):
        raise ValueError("state count must match the case batch")
    result = dict(case)
    result["states"] = states
    metadata = list(case["metadata"])
    state_tensor = metadata[4]
    assert isinstance(state_tensor, torch.Tensor)
    metadata[4] = torch.tensor(
        states, dtype=torch.int32, device=state_tensor.device
    )
    result["metadata"] = tuple(metadata)
    return result


def _assert_first_decode_cache(case: dict[str, object], cache: torch.Tensor) -> None:
    """Check that an untimed -2 invocation produced a complete resident row."""
    cache_cpu = cache.cpu()
    for request, row in enumerate(case["req_entries"]):
        capacity = int(case["cache_tokens"][request])
        slots = cache_cpu[int(row)]
        resident = slots[(slots >= 0) & (slots < capacity)]
        expected = torch.arange(capacity, dtype=torch.int32)
        if resident.numel() != capacity or not torch.equal(
            torch.sort(resident).values, expected
        ):
            raise AssertionError(
                "first-decode initialization did not create a complete "
                f"resident row: request={request}, C={capacity}, "
                f"resident_count={resident.numel()}"
            )


def run_first_decode_perf(args: argparse.Namespace) -> None:
    """Measure -2 initialization separately from its ready -1 steady state."""
    requested_q = [int(value) for value in args.q_pattern.split(",")]
    if len(requested_q) != 1 or not 1 <= requested_q[0] <= MAX_ROUTES:
        raise ValueError("first-decode-perf requires one --q-pattern value in [1,14]")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.offload_len < TOPK or args.offload_len % BLOCK:
        raise ValueError("--offload-len must be >=2048 and 128-aligned")

    q = requested_q[0]
    q_values = [q] * args.batch_size
    first_decode_case = build_case(args, q_values=q_values, states=[-2] * args.batch_size)
    query_ends, _, offload_key, _, _, _ = first_decode_case["metadata"]

    def official() -> torch.Tensor:
        result = torch_npu.npu_lightning_indexer(
            query=first_decode_case["query"],
            key=first_decode_case["key"],
            weights=first_decode_case["weights"],
            actual_seq_lengths_query=query_ends,
            actual_seq_lengths_key=offload_key,
            block_table=first_decode_case["block_table"],
            layout_query="TND",
            layout_key="PA_BSND",
            sparse_count=TOPK,
            sparse_mode=0,
        )
        return result[0] if isinstance(result, (tuple, list)) else result

    empty_seed = first_decode_case["cache_seed"]
    assert isinstance(empty_seed, torch.Tensor)
    first_decode_cache = empty_seed.clone()
    first_decode_outputs = make_outputs(first_decode_case)
    official_median, official_p95 = event_us(
        official, warmup=args.warmup, iters=args.iters
    )
    first_decode_median, first_decode_p95 = event_us(
        lambda: call_custom(
            first_decode_case, first_decode_cache, first_decode_outputs
        ),
        warmup=args.warmup,
        iters=args.iters,
        setup=lambda: first_decode_cache.copy_(empty_seed),
    )

    # Run -2 once outside the measured region.  Reusing the resulting cache
    # makes the following -1 call a same-query, ready-resident reference.
    ready_seed = empty_seed.clone()
    call_custom(first_decode_case, ready_seed, make_outputs(first_decode_case))
    torch.npu.synchronize()
    _assert_first_decode_cache(first_decode_case, ready_seed)
    ready_case = _with_request_states(first_decode_case, [-1] * args.batch_size)
    ready_cache = ready_seed.clone()
    ready_outputs = make_outputs(ready_case)
    ready_median, ready_p95 = event_us(
        lambda: call_custom(ready_case, ready_cache, ready_outputs),
        warmup=args.warmup,
        iters=args.iters,
        setup=lambda: ready_cache.copy_(ready_seed),
    )
    print(
        "FUSED_LI_MANAGE_MTP_FIRST_DECODE_PERF "
        f"mtp={q - 1} batch={args.batch_size} heads={args.heads} "
        f"source_capacity={args.source_capacity} "
        f"offload_len={args.offload_len} "
        f"cache_tokens={first_decode_case['cache_tokens']} "
        f"official_li_mtp{q - 1}_us={official_median:.3f} "
        f"first_decode_lim_mtp{q - 1}_us={first_decode_median:.3f} "
        f"first_decode_overhead_us={first_decode_median - official_median:+.3f} "
        f"ready_steady_lim_mtp{q - 1}_us={ready_median:.3f} "
        f"init_extra_vs_ready_steady_us={first_decode_median - ready_median:+.3f} "
        f"official_p95_us={official_p95:.3f} "
        f"first_decode_p95_us={first_decode_p95:.3f} "
        f"ready_steady_p95_us={ready_p95:.3f} "
        f"warmup={args.warmup} iters={args.iters}",
        flush=True,
    )


def _legacy_random_prefix(values, count, generator):
    if count < 0 or count > values.numel():
        raise AssertionError(f"cannot sample {count} values from {values.numel()}")
    if count == 0:
        return values[:0]
    return values[torch.randperm(values.numel(), generator=generator)[:count]]


def _legacy_balanced_cache(
    topk, *, batch, source_len, source_capacity, cache_tokens, req_entries,
    per_query_misses, union_misses, generator,
):
    """Reproduce the N <= U <= 2N workload used by ops_lim_mtp README."""
    if not per_query_misses <= union_misses <= 2 * per_query_misses:
        raise ValueError("legacy comparison requires N <= union misses <= 2N")
    cache = torch.full((batch + 3, source_capacity), INVALID_SLOT, dtype=torch.int32)
    pairs = ((0b0011, 0b1100), (0b0101, 0b1010), (0b1001, 0b0110))
    for request in range(batch):
        rows = topk[request * 4 : (request + 1) * 4].to(torch.int64)
        union = torch.unique(rows.reshape(-1), sorted=True)
        membership = torch.zeros(source_len, dtype=torch.uint8)
        for route, ids in enumerate(rows):
            membership[ids] |= 1 << route
        buckets = {
            mask: torch.nonzero(membership == mask).flatten().to(torch.int64)
            for mask in range(1, 16)
        }
        common_count = 2 * per_query_misses - union_misses
        pair_units = union_misses - per_query_misses
        if common_count > buckets[0b1111].numel():
            raise AssertionError("insufficient common membership; adjust --query-noise")
        capacities = [min(buckets[a].numel(), buckets[b].numel()) for a, b in pairs]
        pair_counts = [min(pair_units // 3, int(cap)) for cap in capacities]
        remaining = pair_units - sum(pair_counts)
        while remaining:
            progressed = False
            for index, capacity in enumerate(capacities):
                if pair_counts[index] < capacity:
                    pair_counts[index] += 1
                    remaining -= 1
                    progressed = True
                    if remaining == 0:
                        break
            if not progressed:
                raise AssertionError("insufficient pair membership; adjust --query-noise")
        selected = [_legacy_random_prefix(buckets[0b1111], common_count, generator)]
        for (left, right), count in zip(pairs, pair_counts):
            selected += [
                _legacy_random_prefix(buckets[left], count, generator),
                _legacy_random_prefix(buckets[right], count, generator),
            ]
        misses = torch.cat(selected)
        if misses.numel() != union_misses:
            raise AssertionError("legacy union miss construction failed")
        if [int(torch.isin(misses, ids).sum()) for ids in rows] != [per_query_misses] * 4:
            raise AssertionError("legacy per-query miss construction failed")
        missing = torch.zeros(source_len, dtype=torch.bool)
        missing[misses] = True
        hits = union[~missing[union]]
        in_union = torch.zeros(source_len, dtype=torch.bool)
        in_union[union] = True
        fillers = torch.arange(source_len, dtype=torch.int64)[~in_union]
        cached = torch.cat((
            hits,
            _legacy_random_prefix(fillers, cache_tokens - hits.numel(), generator),
        ))
        pool_row = int(req_entries[request])
        cache[pool_row, cached] = torch.randperm(
            cache_tokens, generator=generator, dtype=torch.int64
        ).to(torch.int32)
    return cache.contiguous()


def _balanced_missing_sources(
    rows, *, source_len, per_query_misses, union_misses, generator,
):
    """Select one source set with exact union and per-route miss counts."""
    q = rows.shape[0]
    if not per_query_misses <= union_misses <= q * per_query_misses:
        raise ValueError(
            "balanced misses require N <= union_misses <= Q*N"
        )
    membership = torch.zeros(source_len, dtype=torch.int32)
    for route, ids in enumerate(rows):
        valid = ids[(ids >= 0) & (ids < source_len)]
        membership[valid] |= 1 << route
    active_sources = torch.nonzero(membership).flatten().to(torch.int64)
    active_masks = membership[active_sources]
    sorted_masks, order = torch.sort(active_masks)
    sorted_sources = active_sources[order]
    unique_masks, mask_counts = torch.unique_consecutive(
        sorted_masks, return_counts=True
    )
    buckets: dict[int, torch.Tensor] = {}
    offset = 0
    for mask, count in zip(unique_masks.tolist(), mask_counts.tolist()):
        buckets[int(mask)] = sorted_sources[offset : offset + count]
        offset += count
    masks = list(buckets)
    mask_values = torch.tensor(masks, dtype=torch.int64)
    route_bits = 1 << torch.arange(q, dtype=torch.int64)
    memberships = (mask_values[:, None] & route_bits[None, :]).ne(0)
    membership_counts = memberships.to(torch.int64)
    capacities = torch.tensor(
        [int(buckets[mask].numel()) for mask in masks], dtype=torch.int64
    )

    # The variables are the number of selected sources from every membership
    # bucket.  Evaluate all active masks as tensors: the former nested Python
    # loops became prohibitively slow at Q11+ even after the inactive 2^Q masks
    # were removed from the search space.
    for attempt in range(256):
        available = capacities.clone()
        selected = torch.zeros_like(available)
        deficits = torch.full((q,), per_query_misses, dtype=torch.int64)
        solved = True
        for position in range(union_misses):
            remaining = union_misses - position
            next_deficits = deficits[None, :] - membership_counts
            valid = available.gt(0)
            required = deficits.eq(remaining)
            forbidden = deficits.eq(0)
            if bool(required.any()):
                valid &= memberships[:, required].all(dim=1)
            if bool(forbidden.any()):
                valid &= ~memberships[:, forbidden].any(dim=1)
            valid &= next_deficits.ge(0).all(dim=1)
            valid &= next_deficits.le(remaining - 1).all(dim=1)
            next_remaining = remaining - 1
            next_total = next_deficits.sum(dim=1)
            valid &= next_total.ge(next_remaining)
            valid &= next_total.le(q * next_remaining)
            route_capacity = (
                available[:, None] * membership_counts
            ).sum(dim=0)
            valid &= (
                route_capacity[None, :] - membership_counts
            ).ge(next_deficits).all(dim=1)
            candidate_indices = torch.nonzero(valid).flatten()
            if candidate_indices.numel() == 0:
                solved = False
                break
            candidate_memberships = memberships[candidate_indices].to(torch.float64)
            probability = deficits.to(torch.float64) / remaining
            scores = ((candidate_memberships - probability) ** 2).sum(dim=1)
            desired_membership = float(deficits.sum()) / remaining
            scores += 0.25 * (
                candidate_memberships.sum(dim=1) - desired_membership
            ) ** 2
            scores -= available[candidate_indices].clamp(max=256) * 1.0e-5
            # Early attempts retain the score-greedy solution.  Progressively
            # widen the randomized prefix so Q10+ does not retry the same
            # eight locally optimal membership masks 256 times.
            width = min(int(candidate_indices.numel()), 1 + attempt * 4)
            best = torch.topk(scores, width, largest=False).indices
            choice = int(torch.randint(width, (), generator=generator))
            bucket_index = int(candidate_indices[best[choice]])
            available[bucket_index] -= 1
            selected[bucket_index] += 1
            deficits -= membership_counts[bucket_index]
        if solved and bool(torch.all(deficits == 0)):
            parts = [
                _legacy_random_prefix(buckets[mask], count, generator)
                for mask, count in zip(masks, selected.tolist()) if count
            ]
            missing = torch.cat(parts)
            if missing.numel() != union_misses:
                raise AssertionError("balanced union construction failed")
            counts = [
                int(torch.isin(missing, row).sum()) for row in rows
            ]
            if counts != [per_query_misses] * q:
                raise AssertionError("balanced per-route construction failed")
            return missing
    raise AssertionError(
        "cannot construct balanced misses; adjust --query-noise or workload"
    )


def _homogeneous_controlled_cache(
    topk, *, q, batch, source_len, source_capacity, cache_tokens,
    req_entries, per_query_misses, union_misses, generator,
):
    """Build every request row with exact union and per-route misses."""
    cache = torch.full(
        (batch + 3, source_capacity), INVALID_SLOT, dtype=torch.int32
    )
    actual_unions: list[int] = []
    route_misses: list[int] = []
    for request in range(batch):
        rows = topk[request * q : (request + 1) * q].to(torch.int64)
        union = torch.unique(rows.reshape(-1), sorted=True)
        union = union[(union >= 0) & (union < source_len)]
        if union_misses > union.numel() or union_misses > source_len - cache_tokens:
            raise AssertionError("requested union misses exceed cache capacity")
        missing = _balanced_missing_sources(
            rows, source_len=source_len,
            per_query_misses=per_query_misses,
            union_misses=union_misses, generator=generator,
        )
        actual_unions.append(union_misses)
        route_misses.extend(
            int(torch.isin(missing, row).sum()) for row in rows
        )

        missing_mask = torch.zeros(source_len, dtype=torch.bool)
        missing_mask[missing] = True
        hits = union[~missing_mask[union]]
        in_union = torch.zeros(source_len, dtype=torch.bool)
        in_union[union] = True
        fillers = torch.arange(source_len, dtype=torch.int64)[~in_union]
        filler_count = cache_tokens - hits.numel()
        if filler_count < 0 or filler_count > fillers.numel():
            raise AssertionError(
                "homogeneous steady cache cannot satisfy cache budget"
            )
        cached = torch.cat(
            (hits, _legacy_random_prefix(fillers, filler_count, generator))
        )
        pool_row = int(req_entries[request])
        cache[pool_row, cached] = torch.randperm(
            cache_tokens, generator=generator, dtype=torch.int64
        ).to(torch.int32)
    return cache.contiguous(), actual_unions, route_misses


def _controlled_union_cache(case, topk, union_misses, generator):
    """Build a steady cache with an exact request-level TopK miss union."""
    seed = case["cache_seed"].cpu()
    cache = torch.full_like(seed, INVALID_SLOT)
    length = int(case["offload_key"][0])
    cache_tokens = int(case["cache_tokens"][0])
    row = int(case["req_entries"][0])
    union = torch.unique(topk.reshape(-1).cpu().to(torch.int64), sorted=True)
    union = union[(union >= 0) & (union < length)]
    if union_misses < 0:
        raise ValueError("--union-miss-count must be non-negative")
    # A cache containing C distinct residents from an L-token source can miss
    # at most L-C selected tokens.  In particular C=L is necessarily zero-miss.
    actual_union_misses = min(
        union_misses, int(union.numel()), length - cache_tokens
    )
    missing = _legacy_random_prefix(union, actual_union_misses, generator)
    missing_mask = torch.zeros(length, dtype=torch.bool)
    missing_mask[missing] = True
    hits = union[~missing_mask[union]]
    in_union = torch.zeros(length, dtype=torch.bool)
    in_union[union] = True
    fillers = torch.arange(length, dtype=torch.int64)[~in_union]
    filler_count = cache_tokens - hits.numel()
    if filler_count < 0 or filler_count > fillers.numel():
        raise AssertionError("controlled steady cache cannot satisfy cache budget")
    cached = torch.cat(
        (hits, _legacy_random_prefix(fillers, filler_count, generator))
    )
    cache[row, cached] = torch.randperm(
        cache_tokens, generator=generator, dtype=torch.int64
    ).to(torch.int32)
    return (
        cache.contiguous().to(case["query"].device),
        actual_union_misses,
    )


def _controlled_mixed_union_cache(case, topk, union_misses, generator):
    """Build one controlled resident row for every heterogeneous request."""
    seed = case["cache_seed"].cpu()
    cache = torch.full_like(seed, INVALID_SLOT)
    actual_unions: list[int] = []
    route_begin = 0
    for request, q in enumerate(case["q_values"]):
        length = int(case["offload_key"][request])
        cache_tokens = int(case["cache_tokens"][request])
        row = int(case["req_entries"][request])
        union = torch.unique(
            topk[route_begin : route_begin + q].reshape(-1).cpu().to(torch.int64),
            sorted=True,
        )
        route_begin += q
        union = union[(union >= 0) & (union < length)]
        actual_union = min(union_misses, int(union.numel()), length - cache_tokens)
        missing = _legacy_random_prefix(union, actual_union, generator)
        missing_mask = torch.zeros(length, dtype=torch.bool)
        missing_mask[missing] = True
        hits = union[~missing_mask[union]]
        in_union = torch.zeros(length, dtype=torch.bool)
        in_union[union] = True
        fillers = torch.arange(length, dtype=torch.int64)[~in_union]
        filler_count = cache_tokens - hits.numel()
        if filler_count < 0 or filler_count > fillers.numel():
            raise AssertionError("controlled mixed cache cannot satisfy cache budget")
        cached = torch.cat(
            (hits, _legacy_random_prefix(fillers, filler_count, generator))
        )
        cache[row, cached] = torch.randperm(
            cache_tokens, generator=generator, dtype=torch.int64
        ).to(torch.int32)
        actual_unions.append(actual_union)
    return cache.contiguous().to(case["query"].device), actual_unions


def _randomize_semantic_block_table(case, generator) -> None:
    """Use a non-identity physical block layout in strict semantic cases."""

    source_capacity = int(case["cache_seed"].size(1))
    blocks = source_capacity // BLOCK
    batch = len(case["q_values"])
    device = case["query"].device
    table_cpu = torch.stack(
        [
            torch.randperm(blocks, generator=generator, dtype=torch.int64)
            for _ in range(batch)
        ]
    ).to(torch.int32)
    table = table_cpu.to(device)
    query_to_request = [
        request
        for request, q in enumerate(case["q_values"])
        for _ in range(q)
    ]
    route_table = table[
        torch.tensor(query_to_request, dtype=torch.int64, device=device)
    ].contiguous()
    case["block_table"] = table
    case["route_table"] = route_table


def _force_topk_hit_slot_zero(case, reference, cache):
    """Make legal slot zero observable without changing the resident set."""

    cache_cpu = cache.cpu().clone()
    reference_cpu = reference.cpu().to(torch.int64)
    route_begin = 0
    for request, q in enumerate(case["q_values"]):
        route_end = route_begin + q
        row = int(case["req_entries"][request])
        sources = torch.unique(
            reference_cpu[route_begin:route_end].reshape(-1), sorted=True
        )
        sources = sources[
            (sources >= 0) & (sources < int(case["offload_key"][request]))
        ]
        hits = sources[cache_cpu[row, sources] >= 0]
        if hits.numel() == 0:
            raise AssertionError(
                f"strict steady case request={request} has no TopK hit"
            )
        zero_sources = torch.nonzero(
            cache_cpu[row] == 0, as_tuple=False
        ).flatten()
        if zero_sources.numel() != 1:
            raise AssertionError(
                f"strict steady case request={request} has "
                f"{zero_sources.numel()} owners for slot zero"
            )
        hit_source = int(hits[0])
        zero_source = int(zero_sources[0])
        if hit_source != zero_source:
            hit_slot = int(cache_cpu[row, hit_source])
            cache_cpu[row, zero_source] = hit_slot
            cache_cpu[row, hit_source] = 0
        route_begin = route_end
    return cache_cpu.contiguous().to(cache.device)


def _validate_strict_steady_semantics(case, reference):
    """Validate the complete state transition promised for steady offload.

    Unlike the general correctness smoke test, this reconstructs the expected
    hit/miss ordering and complete cache delta from the cache state that
    existed before the call.  The returned victim slots may be selected by any
    legal scan order, so they are first validated and then used to construct
    the one exact cache image that those outputs imply.
    """

    old_cache = case["cache_seed"].cpu().clone()
    cache = case["cache_seed"].clone()
    outputs = make_outputs(case)
    call_custom(case, cache, outputs)
    torch.npu.synchronize()
    src, dst, route_miss, miss_src, miss_dst, miss_count = [
        tensor.cpu() for tensor in outputs
    ]
    reference_cpu = reference.cpu().to(torch.int64)
    expected_cache = old_cache.clone()
    expected_source_rows: list[torch.Tensor] = []
    expected_union_rows: list[torch.Tensor] = []
    route_begin = 0

    for request, q in enumerate(case["q_values"]):
        route_end = route_begin + q
        row = int(case["req_entries"][request])
        length = int(case["offload_key"][request])
        capacity = int(case["cache_tokens"][request])
        request_misses: list[torch.Tensor] = []
        request_topk: list[torch.Tensor] = []

        for route in range(route_begin, route_end):
            source_ids = reference_cpu[route]
            source_ids = source_ids[
                (source_ids >= 0) & (source_ids < length)
            ]
            if source_ids.numel() != TOPK:
                raise AssertionError(
                    f"strict steady reference route={route} returned "
                    f"{source_ids.numel()} valid sources"
                )
            if torch.unique(source_ids).numel() != TOPK:
                raise AssertionError(
                    f"strict steady reference route={route} contains duplicates"
                )
            old_slots = old_cache[row, source_ids]
            misses = torch.sort(source_ids[old_slots < 0]).values
            hits = torch.sort(source_ids[old_slots >= 0]).values
            expected_sources = torch.cat((misses, hits)).to(torch.int32)
            if not torch.equal(src[route, 0], expected_sources):
                mismatch = torch.nonzero(
                    src[route, 0] != expected_sources, as_tuple=False
                ).flatten()
                first = int(mismatch[0]) if mismatch.numel() else -1
                raise AssertionError(
                    "steady TopK hit/miss order mismatch: "
                    f"request={request}, route={route}, first={first}, "
                    f"expected_misses={misses.numel()}"
                )
            if int(route_miss[route]) != misses.numel():
                raise AssertionError(
                    "steady route miss count mismatch: "
                    f"request={request}, route={route}, "
                    f"actual={int(route_miss[route])}, "
                    f"expected={misses.numel()}"
                )
            expected_source_rows.append(expected_sources)
            request_misses.append(misses)
            request_topk.append(source_ids)

        expected_union = torch.unique(
            torch.cat(request_misses), sorted=True
        ).to(torch.int32)
        expected_union_rows.append(expected_union)
        count = int(expected_union.numel())
        if int(miss_count[request]) != count:
            raise AssertionError(
                "steady request miss count mismatch: "
                f"request={request}, actual={int(miss_count[request])}, "
                f"expected={count}"
            )
        if not torch.equal(miss_src[request, :count], expected_union):
            actual_union = miss_src[request, :count].cpu()
            expected_union_cpu = expected_union.cpu()
            mismatch = torch.nonzero(
                actual_union != expected_union_cpu, as_tuple=False
            ).flatten()
            first = int(mismatch[0]) if mismatch.numel() else -1
            window_begin = max(0, first - 4)
            window_end = min(count, first + 5)
            raise AssertionError(
                "steady miss source union mismatch: "
                f"request={request}, count={count}, first={first}, "
                f"actual={actual_union[window_begin:window_end].tolist()}, "
                f"expected={expected_union_cpu[window_begin:window_end].tolist()}, "
                f"window=[{window_begin},{window_end})"
            )

        victim_slots = miss_dst[request, :count]
        if count:
            invalid = (victim_slots < 0) | (victim_slots >= capacity)
            if bool(invalid.any()):
                first = int(torch.nonzero(invalid, as_tuple=False)[0])
                raise AssertionError(
                    "steady victim slot is outside [0,C): "
                    f"request={request}, index={first}, "
                    f"value={int(victim_slots[first])}, C={capacity}"
                )
            if torch.unique(victim_slots).numel() != count:
                raise AssertionError(
                    f"steady victim slots are not unique for request={request}"
                )

            old_row = old_cache[row]
            resident_sources = torch.nonzero(
                (old_row >= 0) & (old_row < capacity), as_tuple=False
            ).flatten()
            resident_slots = old_row[resident_sources].to(torch.int64)
            if resident_sources.numel() != capacity or not torch.equal(
                torch.sort(resident_slots).values,
                torch.arange(capacity, dtype=torch.int64),
            ):
                raise AssertionError(
                    f"strict steady seed is not a slot bijection for request={request}"
                )
            slot_to_source = torch.empty(capacity, dtype=torch.int64)
            slot_to_source[resident_slots] = resident_sources
            victim_sources = slot_to_source[victim_slots.to(torch.int64)]
            topk_union = torch.unique(
                torch.cat(request_topk), sorted=True
            )
            protected = torch.isin(victim_sources, topk_union)
            if bool(protected.any()):
                first = int(torch.nonzero(protected, as_tuple=False)[0])
                raise AssertionError(
                    "steady eviction selected a TopK-protected source: "
                    f"request={request}, source={int(victim_sources[first])}, "
                    f"slot={int(victim_slots[first])}"
                )
            for incoming, victim, slot in zip(
                expected_union.tolist(),
                victim_sources.tolist(),
                victim_slots.tolist(),
                strict=True,
            ):
                expected_cache[row, victim] = INVALID_SLOT
                expected_cache[row, incoming] = slot
        route_begin = route_end

    actual_cache = cache.cpu()
    if not torch.equal(actual_cache, expected_cache):
        mismatch = torch.nonzero(
            actual_cache != expected_cache, as_tuple=False
        )
        first = mismatch[0].tolist()
        raise AssertionError(
            "steady cache delta or inactive pool row mismatch: "
            f"first_index={first}, actual={int(actual_cache[first[0], first[1]])}, "
            f"expected={int(expected_cache[first[0], first[1]])}"
        )

    for route, expected_sources in enumerate(expected_source_rows):
        request = next(
            request
            for request, end in enumerate(case["query_ends"])
            if route < end
        )
        row = int(case["req_entries"][request])
        expected_slots = expected_cache[
            row, expected_sources.to(torch.int64)
        ]
        if not torch.equal(dst[route, 0], expected_slots):
            actual_slots = dst[route, 0]
            mismatch = torch.nonzero(
                actual_slots != expected_slots, as_tuple=False
            ).flatten()
            first = int(mismatch[0]) if mismatch.numel() else -1
            begin = max(0, first - 4)
            end = min(TOPK, first + 5)
            source = int(expected_sources[first]) if first >= 0 else -1
            raise AssertionError(
                "steady final TopK slots mismatch: "
                f"route={route}, first={first}, source={source}, "
                f"route_misses={int(route_miss[route])}, "
                f"position_kind={'miss' if first < int(route_miss[route]) else 'hit'}, "
                f"cache_slot={int(expected_cache[row, source]) if source >= 0 else -1}, "
                f"actual={actual_slots[begin:end].tolist()}, "
                f"expected={expected_slots[begin:end].tolist()}, "
                f"sources={expected_sources[begin:end].tolist()}, "
                f"window=[{begin},{end})"
            )
    if not bool(torch.any(dst == 0)):
        raise AssertionError("steady output lost the legal cache slot zero")
    return cache, outputs, [int(row.numel()) for row in expected_union_rows]


def run_steady_semantic_regression(args: argparse.Namespace) -> None:
    """Port the mature ops_lim_mtp stage-4 checks to the standardized ABI."""

    scenarios = (
        ("mtp0", [1], 8320, 8192, 64),
        ("mixed-mtp0-2", [1, 2, 3], 8320, 8192, 64),
        ("mtp3-zero", [4], 16256, 12288, 0),
        ("mtp3-normal", [4], 16256, 12288, 300),
        ("mtp3-heavy", [4], 16256, 12288, 750),
        ("mtp3-union-over-2048", [4], 16256, 12288, 2304),
        ("mtp4", [5], 16256, 12288, 400),
        ("mtp5", [6], 16256, 12288, 400),
        ("mtp6", [7], 16256, 14336, 400),
        ("mtp7-wide", [8], 17408, 16384, 800),
        ("mtp11-wide", [12], 25600, 24576, 800),
        ("mtp13-wide", [14], 33536, 32640, 800),
    )
    for index, (label, q_values, offload_len, cache_tokens, target_union) in enumerate(
        scenarios
    ):
        scenario_args = argparse.Namespace(**vars(args))
        scenario_args.source_capacity = offload_len + BLOCK
        scenario_args.offload_len = offload_len
        scenario_args.cache_tokens = cache_tokens
        case = build_case(
            scenario_args, q_values=q_values, states=[-1] * len(q_values)
        )
        generator = torch.Generator().manual_seed(args.seed + 20011 + index * 97)
        _randomize_semantic_block_table(case, generator)
        reference = native_topk(
            case["query"], case["key"], case["weights"],
            case["route_table"], visible_lengths(case),
        )
        if len(q_values) == 1:
            seed, actual_union = _controlled_union_cache(
                case, reference, target_union, generator
            )
            actual_unions = [actual_union]
        else:
            seed, actual_unions = _controlled_mixed_union_cache(
                case, reference, target_union, generator
            )
        if actual_unions != [target_union] * len(q_values):
            raise AssertionError(
                f"strict steady scenario={label} constructed unions="
                f"{actual_unions}, expected={target_union}"
            )
        case["cache_seed"] = _force_topk_hit_slot_zero(
            case, reference, seed
        )
        updated_cache, _, observed_unions = _validate_strict_steady_semantics(
            case, reference
        )
        if observed_unions != actual_unions:
            raise AssertionError(
                f"strict steady scenario={label} observed unions="
                f"{observed_unions}, expected={actual_unions}"
            )

        # The same TopK must be fully resident after the first replacement.
        # Validate the complete second output and require an unchanged row.
        repeat_case = dict(case)
        repeat_case["cache_seed"] = updated_cache.clone()
        repeated_cache, repeat_outputs, repeat_unions = (
            _validate_strict_steady_semantics(repeat_case, reference)
        )
        if repeat_unions != [0] * len(q_values):
            raise AssertionError(
                f"strict steady scenario={label} repeat unions={repeat_unions}"
            )
        if not torch.equal(repeated_cache, updated_cache):
            raise AssertionError(
                f"strict steady scenario={label} changed cache on repeat"
            )
        if bool(torch.any(repeat_outputs[2] != 0)) or bool(
            torch.any(repeat_outputs[5] != 0)
        ):
            raise AssertionError(
                f"strict steady scenario={label} repeat counts are nonzero"
            )
        print(
            "steady semantic regression PASS "
            f"scenario={label} q={q_values} L={offload_len} C={cache_tokens} "
            f"union={target_union} random_mapping=1 random_block_table=1 "
            f"repeat_stable=1 dtype={args.dtype}"
        )


def run_sort4096_regression(args: argparse.Namespace) -> None:
    """Exercise the Q5--Q7 SortAll/fallback boundary by occurrence count."""

    scenarios = (
        (5, 2048),
        (5, 2049),
        (6, 4095),
        (7, 4096),
        (7, 4097),
    )
    for index, (q, occurrence_count) in enumerate(scenarios):
        cache_tokens = q * TOPK
        extra = ((occurrence_count + BLOCK - 1) // BLOCK) * BLOCK
        scenario_args = argparse.Namespace(**vars(args))
        scenario_args.cache_tokens = cache_tokens
        scenario_args.offload_len = cache_tokens + extra
        scenario_args.source_capacity = scenario_args.offload_len + BLOCK
        case = build_case(scenario_args, q_values=[q], states=[-1])

        query = case["query"]
        weights = case["weights"]
        key = case["key"]
        assert isinstance(query, torch.Tensor)
        assert isinstance(weights, torch.Tensor)
        assert isinstance(key, torch.Tensor)
        query.zero_()
        weights.fill_(1)
        key.zero_()
        key_by_source = key.reshape(
            scenario_args.source_capacity, 1, HEAD_DIM
        )
        for route in range(q):
            query[route, :, route].fill_(1)
            key_by_source[
                route * TOPK : (route + 1) * TOPK, 0, route
            ].fill_(1)

        reference = native_topk(
            query, key, weights, case["route_table"], visible_lengths(case)
        ).cpu().to(torch.int64)
        expected_union = torch.arange(cache_tokens, dtype=torch.int64)
        actual_union = torch.unique(reference.reshape(-1), sorted=True)
        if not torch.equal(actual_union, expected_union):
            raise AssertionError(
                f"Q{q} disjoint TopK construction failed at "
                f"occurrence={occurrence_count}"
            )

        route_misses = [occurrence_count // q] * q
        for route in range(occurrence_count % q):
            route_misses[route] += 1
        missing_parts = [
            reference[route, :route_misses[route]] for route in range(q)
        ]
        hit_parts = [
            reference[route, route_misses[route]:] for route in range(q)
        ]
        missing = torch.cat(missing_parts)
        hits = torch.cat(hit_parts)
        if missing.numel() != occurrence_count:
            raise AssertionError("sort4096 miss construction failed")

        fillers = torch.arange(
            cache_tokens, cache_tokens + occurrence_count,
            dtype=torch.int64,
        )
        cached = torch.cat((hits, fillers))
        if cached.numel() != cache_tokens:
            raise AssertionError("sort4096 resident construction failed")
        generator = torch.Generator().manual_seed(
            args.seed + 4096 + index * 997
        )
        cache = torch.full_like(case["cache_seed"].cpu(), INVALID_SLOT)
        row = int(case["req_entries"][0])
        cache[row, cached] = torch.randperm(
            cache_tokens, generator=generator, dtype=torch.int64
        ).to(torch.int32)
        case["cache_seed"] = _force_topk_hit_slot_zero(
            case, reference, cache
        ).to(query.device)

        updated_cache, outputs, observed_unions = (
            _validate_strict_steady_semantics(case, reference)
        )
        if observed_unions != [occurrence_count]:
            raise AssertionError(
                "sort4096 occurrence union mismatch: "
                f"actual={observed_unions}, expected={[occurrence_count]}"
            )
        repeat_case = dict(case)
        repeat_case["cache_seed"] = updated_cache.clone()
        _, repeat_outputs, repeat_unions = _validate_strict_steady_semantics(
            repeat_case, reference
        )
        if (repeat_unions != [0] or bool(torch.any(repeat_outputs[2] != 0))
                or bool(torch.any(repeat_outputs[5] != 0))):
            raise AssertionError("sort4096 regression did not converge")
        if int(outputs[2].sum()) != occurrence_count:
            raise AssertionError("sort4096 route miss total mismatch")
        print(
            "sort4096 regression PASS "
            f"q={q} occurrence={occurrence_count} C={cache_tokens} "
            f"branch={'SortAll4096' if occurrence_count <= 4096 else 'k-way'} "
            f"repeat_stable=1 dtype={args.dtype}"
        )


def run_mixed_perf(args: argparse.Namespace) -> None:
    """Time one invocation containing a repeated heterogeneous MTP pattern."""
    standard_mode = args.mode == "mixed-standard-mtp-perf"
    requested_q = [int(value) for value in args.q_pattern.split(",")]
    if len(requested_q) < 2 or any(
        q < 1 or q > MAX_ROUTES for q in requested_q
    ):
        raise ValueError(
            f"{args.mode} requires at least two --q-pattern values in [1,14]"
        )
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    q_values = requested_q * args.batch_size
    if not standard_mode and args.cache_tokens < max(q_values) * TOPK:
        raise ValueError(
            "mixed offload requires --cache-tokens >= max(Q) * 2048"
        )
    states = [-3 if standard_mode else -1] * len(q_values)
    case = build_case(args, q_values=q_values, states=states)
    query_ends, actual_key, offload_key, _, _, _ = case["metadata"]

    def official():
        result = torch_npu.npu_lightning_indexer(
            query=case["query"], key=case["key"], weights=case["weights"],
            actual_seq_lengths_query=query_ends,
            actual_seq_lengths_key=actual_key if standard_mode else offload_key,
            block_table=case["block_table"], layout_query="TND", layout_key="PA_BSND",
            sparse_count=TOPK, sparse_mode=0,
        )
        return result[0] if isinstance(result, (tuple, list)) else result

    native_result = official()
    native_topk = native_result.reshape(sum(q_values), TOPK)
    if standard_mode:
        seed = case["cache_seed"]
        actual_unions = None
    else:
        seed, actual_unions = _controlled_mixed_union_cache(
            case, native_topk, args.union_miss_count,
            torch.Generator().manual_seed(args.seed + 104729),
        )
    cache = seed.clone()
    outputs = make_outputs(case)
    official_median, official_p95 = event_us(
        official, warmup=args.warmup, iters=args.iters
    )
    fused_median, fused_p95 = event_us(
        lambda: call_custom(case, cache, outputs), warmup=args.warmup,
        iters=args.iters, setup=lambda: cache.copy_(seed),
    )
    label = "STANDARD" if standard_mode else "OFFLOAD"
    union_text = "" if actual_unions is None else (
        f" target_union_misses={args.union_miss_count} "
        f"actual_union_misses_min={min(actual_unions)} "
        f"actual_union_misses_max={max(actual_unions)}"
    )
    print(
        f"FUSED_LI_MANAGE_MTP_MIXED_{label}_PERF "
        f"q_pattern={requested_q} pattern_repeats={args.batch_size} "
        f"batch={len(q_values)} heads={args.heads} "
        f"source_capacity={args.source_capacity} "
        f"offload_len={args.offload_len} cache_tokens={args.cache_tokens}"
        f"{union_text} "
        f"official_median_us={official_median:.3f} "
        f"fused_median_us={fused_median:.3f} "
        f"overhead_us={fused_median - official_median:+.3f} "
        f"official_p95_us={official_p95:.3f} "
        f"fused_p95_us={fused_p95:.3f} "
        f"warmup={args.warmup} iters={args.iters}", flush=True,
    )


def run_mixed_state_perf(args: argparse.Namespace) -> None:
    """Time heterogeneous Q and -3/-2/-1 requests in one invocation."""
    requested_q = [int(value) for value in args.q_pattern.split(",")]
    requested_states = [int(value) for value in args.state_pattern.split(",")]
    if len(requested_q) < 2 or any(q < 1 or q > MAX_ROUTES for q in requested_q):
        raise ValueError(
            "mixed-state-perf requires at least two --q-pattern values in [1,14]"
        )
    if len(requested_states) != len(requested_q):
        raise ValueError("--state-pattern must have one state per --q-pattern value")
    state_set = set(requested_states)
    if not state_set <= {-3, -2, -1} or len(state_set) < 2:
        raise ValueError(
            "--state-pattern must mix at least two valid states from -3, -2 and -1"
        )
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    q_values = requested_q * args.batch_size
    states = requested_states * args.batch_size
    if args.cache_tokens < max(
        q * TOPK for q, state in zip(q_values, states) if state != -3
    ):
        raise ValueError(
            "mixed-state-perf requires --cache-tokens >= max offload Q * 2048"
        )
    case = build_case(args, q_values=q_values, states=states)
    query_ends, actual_key, offload_key, _, _, _ = case["metadata"]
    effective_key = torch.tensor(
        [
            case["actual_key"][request]
            if state == -3 else case["offload_key"][request]
            for request, state in enumerate(states)
        ],
        dtype=torch.int32,
        device=case["query"].device,
    )

    def official():
        result = torch_npu.npu_lightning_indexer(
            query=case["query"], key=case["key"], weights=case["weights"],
            actual_seq_lengths_query=query_ends,
            actual_seq_lengths_key=effective_key,
            block_table=case["block_table"], layout_query="TND",
            layout_key="PA_BSND", sparse_count=TOPK, sparse_mode=0,
        )
        return result[0] if isinstance(result, (tuple, list)) else result

    native_result = official().reshape(sum(q_values), TOPK)
    seed, all_unions = _controlled_mixed_union_cache(
        case, native_result, args.union_miss_count,
        torch.Generator().manual_seed(args.seed + 104729),
    )
    # Only -1 requests begin with a resident cache.  Repeated measurements
    # reset -2 to an empty row and -3 to an uninitialized row so each timed
    # invocation includes their real first-decode and identity-fill work.
    seed_cpu = seed.cpu()
    for request, state in enumerate(states):
        if state != -1:
            seed_cpu[int(case["req_entries"][request])].fill_(INVALID_SLOT)
    seed = seed_cpu.to(case["query"].device)
    steady_unions = [
        union for union, state in zip(all_unions, states) if state == -1
    ]
    steady_union_text = (
        f"actual_steady_union_misses_min={min(steady_unions)} "
        f"actual_steady_union_misses_max={max(steady_unions)} "
        if steady_unions else
        "actual_steady_union_misses_min=NA "
        "actual_steady_union_misses_max=NA "
    )
    cache = seed.clone()
    outputs = make_outputs(case)
    official_median, official_p95 = event_us(
        official, warmup=args.warmup, iters=args.iters
    )
    fused_median, fused_p95 = event_us(
        lambda: call_custom(case, cache, outputs),
        warmup=args.warmup, iters=args.iters,
        setup=lambda: cache.copy_(seed),
    )
    print(
        "FUSED_LI_MANAGE_MTP_MIXED_STATE_PERF "
        f"q_pattern={requested_q} state_pattern={requested_states} "
        f"pattern_repeats={args.batch_size} batch={len(q_values)} "
        f"heads={args.heads} source_capacity={args.source_capacity} "
        f"offload_len={args.offload_len} cache_tokens={args.cache_tokens} "
        f"target_steady_union_misses={args.union_miss_count} "
        f"{steady_union_text}"
        f"official_median_us={official_median:.3f} "
        f"fused_median_us={fused_median:.3f} "
        f"overhead_us={fused_median - official_median:+.3f} "
        f"official_p95_us={official_p95:.3f} "
        f"fused_p95_us={fused_p95:.3f} "
        f"warmup={args.warmup} iters={args.iters}", flush=True,
    )


def run_legacy_perf(args):
    """Homogeneous MTP A/B workload using the ops_lim_mtp timing method."""
    standard_mode = args.mode == "standard-mtp-perf"
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.source_len < TOPK or args.source_len % BLOCK:
        raise ValueError("--source-len must be >=2048 and 128-aligned")
    if (not standard_mode and not TOPK <= args.cache_tokens <= min(
        args.source_len, MAX_CACHE_TOKENS
    )):
        raise ValueError(
            "--cache-tokens must be in [2048,min(source_len,32640)]"
        )
    if not standard_mode and not 0 <= args.query_miss_count <= TOPK:
        raise ValueError("--query-miss-count must be in [0,2048]")
    if args.query_noise <= 0:
        raise ValueError("--query-noise must be positive")

    if args.mode == "legacy-perf":
        q = 4
    else:
        q_values = [int(value) for value in args.q_pattern.split(",")]
        if len(q_values) != 1 or not 1 <= q_values[0] <= MAX_ROUTES:
            raise ValueError(
                f"{args.mode} requires one --q-pattern value in [1,14]"
            )
        q = q_values[0]
    if (not standard_mode and args.source_len > q * TOPK and
            args.cache_tokens < q * TOPK):
        raise ValueError(
            f"MTP{q - 1} requires --cache-tokens >= {q * TOPK} "
            "for this source length"
        )

    batch = args.batch_size
    physical_len = args.source_len if standard_mode else args.source_len + BLOCK
    if physical_len > MAX_SOURCE_CAPACITY:
        mode_limit = "--source-len" if standard_mode else "--source-len + 128"
        raise ValueError(
            f"{mode_limit} must be <= 2^21 physical source tokens"
        )
    blocks_per_request = physical_len // BLOCK
    generator = torch.Generator().manual_seed(args.seed)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    device = torch.device(args.device)
    base_query = torch.randn(batch, 1, args.heads, HEAD_DIM, generator=generator)
    noise = torch.randn(batch, q, args.heads, HEAD_DIM, generator=generator)
    query_cpu = (base_query + args.query_noise * noise).reshape(
        batch * q, args.heads, HEAD_DIM
    ).to(dtype)
    base_weights = torch.rand(batch, 1, args.heads, generator=generator)
    weights_cpu = base_weights.expand(-1, q, -1).reshape(
        batch * q, args.heads
    ).contiguous().to(dtype)
    # Above 128K, a private [B, L] key pool can require hundreds of GiB for
    # the 1M/2M performance sweep even though the operator only requires each
    # block-table row to reference a valid physical block.  Reuse one physical
    # key pool across requests in this long-source benchmark.  Query rows and
    # cache_slots_pool rows remain independent, and the old private layout is
    # retained bit-for-bit for the existing <=128K performance baselines.
    shared_long_key = physical_len > PACKED_SOURCE_CAPACITY
    key_blocks = blocks_per_request if shared_long_key else batch * blocks_per_request
    key_generator = torch.Generator().manual_seed(args.seed + 991)
    if shared_long_key:
        # Keep the 1M/2M key tensor off host RAM; construction is outside the
        # timing region and uses the deterministic seed only once per run.
        torch.manual_seed(args.seed + 991)
        key = torch.randn(
            key_blocks, BLOCK, 1, HEAD_DIM, dtype=dtype, device=device
        )
    else:
        key_cpu = torch.randn(
            key_blocks, BLOCK, 1, HEAD_DIM,
            dtype=dtype, generator=key_generator,
        )
        key = key_cpu.to(device)
    # Preserve the old branch's global random physical-block permutation for
    # the searched prefix.  Offload modes append one private block per request
    # only to meet the standardized ABI's causal-visible-length requirement.
    searched_blocks = args.source_len // BLOCK
    if shared_long_key:
        prefix_table = torch.randperm(
            searched_blocks, generator=generator, dtype=torch.int64
        ).reshape(1, searched_blocks).expand(batch, -1).contiguous()
    else:
        prefix_table = torch.randperm(
            batch * searched_blocks, generator=generator, dtype=torch.int64
        ).reshape(batch, searched_blocks)
    if standard_mode:
        block_table_cpu = prefix_table.to(torch.int32)
    else:
        if shared_long_key:
            extra_table = torch.full(
                (batch, 1), searched_blocks, dtype=torch.int64
            )
        else:
            extra_table = torch.arange(
                batch * searched_blocks,
                batch * searched_blocks + batch,
                dtype=torch.int64,
            ).reshape(batch, 1)
        block_table_cpu = torch.cat((prefix_table, extra_table), dim=1).to(torch.int32)
    query, weights = query_cpu.to(device), weights_cpu.to(device)
    block_table = block_table_cpu.to(device)
    query_ends = torch.arange(
        q, batch * q + 1, q, dtype=torch.int32, device=device
    )
    offload_lens = torch.full((batch,), args.source_len, dtype=torch.int32, device=device)
    actual_lens = torch.full((batch,), physical_len, dtype=torch.int32, device=device)
    cache_tokens = torch.full((batch,), args.cache_tokens, dtype=torch.int32, device=device)
    states = torch.full(
        (batch,), -3 if standard_mode else -1,
        dtype=torch.int32, device=device,
    )
    req_cpu = torch.randperm(batch + 3, generator=generator, dtype=torch.int64)[:batch].to(torch.int32)
    req_entries = req_cpu.to(device)

    def official():
        result = torch_npu.npu_lightning_indexer(
            query=query, key=key, weights=weights,
            actual_seq_lengths_query=query_ends,
            actual_seq_lengths_key=actual_lens if standard_mode else offload_lens,
            block_table=block_table, layout_query="TND", layout_key="PA_BSND",
            sparse_count=TOPK, sparse_mode=0,
        )
        return result[0] if isinstance(result, (tuple, list)) else result

    topk = official().reshape(batch * q, TOPK).cpu().to(torch.int64)
    if standard_mode:
        cache_cpu = torch.full(
            (batch + 3, physical_len), INVALID_SLOT, dtype=torch.int32
        )
        actual_unions = [0] * batch
        route_misses = [0] * (batch * q)
    elif q == 4:
        cache_cpu = _legacy_balanced_cache(
            topk, batch=batch, source_len=args.source_len,
            source_capacity=physical_len, cache_tokens=args.cache_tokens,
            req_entries=req_cpu, per_query_misses=args.query_miss_count,
            union_misses=args.union_miss_count, generator=generator,
        )
        actual_unions = [args.union_miss_count] * batch
        route_misses = [args.query_miss_count] * (batch * q)
    else:
        cache_cpu, actual_unions, route_misses = _homogeneous_controlled_cache(
            topk, q=q, batch=batch, source_len=args.source_len,
            source_capacity=physical_len, cache_tokens=args.cache_tokens,
            req_entries=req_cpu, per_query_misses=args.query_miss_count,
            union_misses=args.union_miss_count, generator=generator,
        )
    case = {
        "q_values": [q] * batch,
        "states": [-3 if standard_mode else -1] * batch,
        "query": query, "weights": weights,
        "query_scale": torch.zeros(
            batch * q, args.heads, dtype=torch.float32, device=device
        ),
        "key": key,
        "key_scale": torch.zeros(
            key_blocks, BLOCK, 1, dtype=torch.float32, device=device
        ),
        "block_table": block_table,
        "metadata": (query_ends, actual_lens, offload_lens, cache_tokens, states, req_entries),
    }
    seed, cache = cache_cpu.to(device), cache_cpu.to(device)
    outputs = make_outputs(case)

    official_median, official_p95 = event_us(
        official, warmup=args.warmup, iters=args.iters
    )
    fused_median, fused_p95 = event_us(
        lambda: call_custom(case, cache, outputs),
        warmup=args.warmup, iters=args.iters, setup=lambda: cache.copy_(seed),
    )
    if standard_mode:
        print(
            "FUSED_LI_MANAGE_MTP_STANDARD_PERF "
            f"mtp={q - 1} q={q} batch={batch} heads={args.heads} "
            f"source_len={args.source_len} "
            f"key_layout={'shared' if shared_long_key else 'private'} "
            f"official_li_mtp{q - 1}_us={official_median:.3f} "
            f"standard_lim_mtp{q - 1}_us={fused_median:.3f} "
            f"standard_overhead_us={fused_median - official_median:+.3f} "
            f"official_p95_us={official_p95:.3f} "
            f"standard_p95_us={fused_p95:.3f} "
            f"warmup={args.warmup} iters={args.iters}", flush=True,
        )
        return
    if args.mode == "legacy-perf":
        print(
            "FUSED_LI_MANAGE_MTP_LEGACY_PERF "
            f"batch={batch} heads={args.heads} source_len={args.source_len} "
            f"key_layout={'shared' if shared_long_key else 'private'} "
            f"cache_tokens={args.cache_tokens} "
            f"per_query_misses={args.query_miss_count} "
            f"union_misses={args.union_miss_count} "
            f"query_noise={args.query_noise:.4f} "
            f"official_li_mtp3_us={official_median:.3f} "
            f"stage4_lim_mtp_us={fused_median:.3f} "
            f"stage4_overhead_us={fused_median - official_median:+.3f} "
            f"official_p95_us={official_p95:.3f} "
            f"stage4_p95_us={fused_p95:.3f} "
            f"warmup={args.warmup} iters={args.iters}", flush=True,
        )
        return

    union_min, union_max = min(actual_unions), max(actual_unions)
    route_min, route_max = min(route_misses), max(route_misses)
    route_mean = statistics.mean(route_misses)
    print(
        "FUSED_LI_MANAGE_MTP_HOMOGENEOUS_PERF "
        f"mtp={q - 1} q={q} batch={batch} heads={args.heads} "
        f"source_len={args.source_len} cache_tokens={args.cache_tokens} "
        f"key_layout={'shared' if shared_long_key else 'private'} "
        f"target_union_misses={args.union_miss_count} "
        f"actual_union_misses_min={union_min} "
        f"actual_union_misses_max={union_max} "
        f"route_misses_min={route_min} route_misses_max={route_max} "
        f"route_misses_mean={route_mean:.3f} "
        f"query_noise={args.query_noise:.4f} "
        f"official_li_mtp{q - 1}_us={official_median:.3f} "
        f"stage4_lim_mtp{q - 1}_us={fused_median:.3f} "
        f"stage4_overhead_us={fused_median - official_median:+.3f} "
        f"official_p95_us={official_p95:.3f} "
        f"stage4_p95_us={fused_p95:.3f} "
        f"warmup={args.warmup} iters={args.iters}", flush=True,
    )


def run_perf(args: argparse.Namespace) -> None:
    results: dict[str, dict[str, float]] = {}
    native_results: dict[int, dict[str, float]] = {}
    native_topk: dict[int, torch.Tensor] = {}
    requested_q = {int(value) for value in args.q_pattern.split(",")}
    perf_q = [q for q in range(1, MAX_ROUTES + 1) if q in requested_q]
    if not perf_q:
        raise ValueError("perf mode requires --q-pattern to contain Q=1..14")
    for q in perf_q:
        case = build_case(args, q_values=[q], states=[-1])
        query_ends, _, offload_key, _, _, _ = case["metadata"]

        def invoke_native() -> None:
            torch_npu.npu_lightning_indexer(
                query=case["query"],
                key=case["key"],
                weights=case["weights"],
                actual_seq_lengths_query=query_ends,
                actual_seq_lengths_key=offload_key,
                block_table=case["block_table"],
                layout_query="TND",
                layout_key="PA_BSND",
                sparse_count=TOPK,
                sparse_mode=0,
            )

        print(f"perf begin native_mtp{q - 1}", flush=True)
        median, p95 = event_us(
            invoke_native, warmup=args.warmup, iters=args.iters
        )
        native_results[q] = {"median_us": median, "p95_us": p95}
        results[f"native_mtp{q - 1}"] = native_results[q]
        result = torch_npu.npu_lightning_indexer(
            query=case["query"], key=case["key"], weights=case["weights"],
            actual_seq_lengths_query=query_ends,
            actual_seq_lengths_key=offload_key,
            block_table=case["block_table"], layout_query="TND",
            layout_key="PA_BSND", sparse_count=TOPK, sparse_mode=0,
        )
        native_topk[q] = (
            result[0] if isinstance(result, (tuple, list)) else result
        ).reshape(q, TOPK)

    for state_label, state in (("offload", -1), ("standard", -3)):
        for q in perf_q:
            label = f"{state_label}_mtp{q - 1}"
            case = build_case(args, q_values=[q], states=[state])
            seed = case["cache_seed"]
            actual_union_misses = 0
            if state == -1:
                generator = torch.Generator().manual_seed(args.seed + q * 104729)
                seed, actual_union_misses = _controlled_union_cache(
                    case, native_topk[q], args.union_miss_count, generator
                )
            cache = seed.clone()
            outputs = make_outputs(case)

            def invoke() -> None:
                call_custom(case, cache, outputs)

            suffix = (
                f" union_misses={actual_union_misses}"
                if state == -1 else ""
            )
            print(f"perf begin {label}{suffix}", flush=True)
            median, p95 = event_us(
                invoke,
                warmup=args.warmup,
                iters=args.iters,
                setup=lambda: cache.copy_(seed),
            )
            results[label] = {"median_us": median, "p95_us": p95}
    baselines = {}
    if args.baseline_json:
        baselines = json.loads(args.baseline_json.read_text(encoding="utf-8"))
    for label, values in results.items():
        baseline = baselines.get(label, {}).get("median_us")
        ratio = values["median_us"] / baseline if baseline else None
        print(
            f"perf {label}: median={values['median_us']:.3f}us "
            f"p95={values['p95_us']:.3f}us"
            + (f" baseline={baseline:.3f}us ratio={ratio:.4f}" if ratio else "")
        )
        if label.startswith("offload_"):
            q = int(label.rsplit("mtp", 1)[1]) + 1
            native = native_results[q]["median_us"]
            print(
                f"perf {label}_vs_native: native={native:.3f}us "
                f"overhead={values['median_us'] - native:+.3f}us"
            )
    if args.write_perf_json:
        args.write_perf_json.write_text(
            json.dumps(results, indent=2, sort_keys=True), encoding="utf-8"
        )


def main() -> None:
    args = parse_args()
    require_local_opapi()
    if args.mode in ("legacy-perf", "mtp-perf", "standard-mtp-perf"):
        run_legacy_perf(args)
        return
    if args.mode in ("mixed-mtp-perf", "mixed-standard-mtp-perf"):
        run_mixed_perf(args)
        return
    if args.mode == "mixed-state-perf":
        run_mixed_state_perf(args)
        return
    if args.mode == "transition-perf":
        run_transition_perf(args)
        return
    if args.mode == "first-decode-perf":
        run_first_decode_perf(args)
        return
    if args.mode == "long-regression":
        run_long_regression(args)
        return
    if args.mode == "key-tag-regression":
        run_key_tag_regression(args)
        return
    if args.mode == "payload-codec-regression":
        run_payload_codec_regression(args)
        return
    if args.mode == "sort4096-regression":
        run_sort4096_regression(args)
        return
    if args.mode == "occurrence-regression":
        run_occurrence_mapping_regression(args)
        return
    if args.mode == "wide-compact-union-regression":
        run_wide_compact_union_regression(args)
        return
    if args.mode == "wide-route-regression":
        run_wide_route_regression(args)
        return
    if args.mode == "all":
        run_payload_codec_regression(args)
        run_wide_route_regression(args)
        run_wide_compact_union_regression(args)
    if args.mode in ("steady-semantic-regression", "all"):
        run_steady_semantic_regression(args)
    if args.mode == "all":
        run_sort4096_regression(args)
    if args.mode in ("replacement-regression", "all"):
        run_replacement_regression(args)
    if args.mode in ("first-decode-regression", "all"):
        run_first_decode_regression(args)
    if args.mode in ("correctness", "all"):
        run_correctness(args)
    if args.mode in ("lifecycle", "all"):
        run_lifecycle(args)
    if args.mode in ("invalid", "all"):
        run_invalid(args)
    if args.mode in ("perf", "all"):
        run_perf(args)


if __name__ == "__main__":
    main()
