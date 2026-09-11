"""Shared index-management construction and assertions for the C8 fp8 MTP path.

Vendored from the A5 repo's tests/_lidu_utils.py so the unified tree can assert
the C8 cache-update contract without depending on the A5 checkout being present.
"""

from __future__ import annotations

import torch
import torch_npu


TOPK = 2048
MAX_SOURCE_CAPACITY = 1 << 18
MAX_CACHE_TOKENS = 16256


def official_reference_topk(
    query: torch.Tensor,
    key: torch.Tensor,
    weights: torch.Tensor,
    query_scale: torch.Tensor,
    key_scale: torch.Tensor,
    actual_q: torch.Tensor,
    candidate_lens: torch.Tensor,
    block_table: torch.Tensor,
) -> torch.Tensor | None:
    """Top-k from the official A5 C8 LightningIndexer, or None if not present.

    Preferred over a hand-rolled float reference because it shares the kernel's
    fp8 cube accumulation, so the top-2048 boundary is bit-identical instead of
    being a coin flip on rounding.
    """
    op = getattr(torch_npu, "npu_quant_lightning_indexer", None)
    if op is None:
        namespace = getattr(torch.ops, "_C_ascend", None)
        op = (
            getattr(namespace, "npu_lightning_indexer_quant", None)
            if namespace is not None
            else None
        )
    if op is None:
        return None
    output = op(
        query=query,
        key=key,
        weights=weights,
        query_dequant_scale=query_scale,
        key_dequant_scale=key_scale,
        actual_seq_lengths_query=actual_q,
        actual_seq_lengths_key=candidate_lens,
        block_table=block_table,
        query_quant_mode=0,
        key_quant_mode=0,
        layout_query="TND",
        layout_key="PA_BSND",
        sparse_count=TOPK,
        sparse_mode=3,
    )
    topk = output[0] if isinstance(output, tuple) else output
    if not isinstance(topk, torch.Tensor) or topk.dtype != torch.int32:
        return None
    if topk.numel() != query.size(0) * TOPK:
        return None
    return topk.reshape(query.size(0), TOPK).contiguous()


def ordered_union(rows: torch.Tensor) -> set[int]:
    seen: set[int] = set()
    for token in rows.reshape(-1).tolist():
        token = int(token)
        if token >= 0:
            seen.add(token)
    return seen


def assert_pool_row(row: torch.Tensor, candidate_len: int, budget: int) -> None:
    valid_tokens = (row[:candidate_len] >= 0).nonzero().flatten()
    if bool((row[candidate_len:] >= 0).any()):
        raise AssertionError("cache row contains a token outside candidate_len")
    if budget == 0:
        if valid_tokens.numel() != 0:
            raise AssertionError("C=0 request mutated its cache row")
        return
    if valid_tokens.numel() != budget:
        raise AssertionError(
            f"cache cardinality={valid_tokens.numel()}, expected {budget}"
        )
    slots = row[valid_tokens]
    expected = torch.arange(budget, dtype=torch.int32)
    if not torch.equal(torch.sort(slots).values, expected):
        raise AssertionError("cache slots are not the exact unique range [0,C)")


def assert_request_pool_entries(
    entries: torch.Tensor, batch: int, pool_size: int
) -> None:
    if entries.dtype != torch.int32 or entries.shape != (batch,):
        raise AssertionError("req_pool_entries must be int32[B]")
    if int(entries.min()) < 0 or int(entries.max()) >= pool_size:
        raise AssertionError("req_pool_entries contains an invalid request-pool row")
    if torch.unique(entries).numel() != batch:
        raise AssertionError("active req_pool_entries are not unique")


def assert_update_row(
    *,
    label: str,
    sources: torch.Tensor,
    slots: torch.Tensor,
    reference: torch.Tensor,
    old_row: torch.Tensor,
    new_row: torch.Tensor,
    candidate_len: int,
    budget: int,
    expected_miss: int,
    actual_miss: int,
) -> None:
    if budget == 0:
        if (
            actual_miss != 0
            or bool((sources != -1).any())
            or bool((slots != -1).any())
            or not torch.equal(old_row, new_row)
        ):
            raise AssertionError(f"{label}: C=0 row is not a strict no-op")
        return

    if sources.numel() != TOPK or slots.numel() != TOPK:
        raise AssertionError(f"{label}: expected exactly {TOPK} source IDs and slots")
    if int(sources.min()) < 0 or int(sources.max()) >= candidate_len:
        raise AssertionError(f"{label}: topk_index is outside [0,{candidate_len})")
    if torch.unique(sources).numel() != TOPK:
        raise AssertionError(f"{label}: topk_index is not unique")
    if (
        int(reference.min()) < 0
        or int(reference.max()) >= candidate_len
        or torch.unique(reference).numel() != TOPK
    ):
        raise AssertionError(f"{label}: LightningIndexer reference is invalid")
    if not torch.equal(torch.sort(sources).values, torch.sort(reference).values):
        raise AssertionError(f"{label}: top-2048 set differs from LightningIndexer")

    if not 0 <= actual_miss <= TOPK:
        raise AssertionError(
            f"{label}: miss_count={actual_miss} is outside [0,{TOPK}]"
        )
    if actual_miss > candidate_len - budget:
        raise AssertionError(
            f"{label}: miss_count={actual_miss} exceeds candidate_len-C"
        )
    old_selected_slots = old_row.gather(0, sources.long())
    new_selected_slots = new_row.gather(0, sources.long())
    recomputed_miss = int((old_selected_slots < 0).sum())
    if actual_miss != expected_miss or actual_miss != recomputed_miss:
        raise AssertionError(
            f"{label}: miss_count={actual_miss}, expected={expected_miss}, "
            f"recomputed={recomputed_miss}"
        )
    if actual_miss and bool((old_selected_slots[:actual_miss] >= 0).any()):
        raise AssertionError(f"{label}: miss prefix contains a cache hit")
    if actual_miss < TOPK:
        old_hit_slots = old_selected_slots[actual_miss:]
        if bool((old_hit_slots < 0).any()):
            raise AssertionError(f"{label}: hit suffix contains a cache miss")
        if not torch.equal(slots[actual_miss:], old_hit_slots):
            raise AssertionError(f"{label}: an old hit changed its HBM slot")

    if int(slots.min()) < 0 or int(slots.max()) >= budget:
        raise AssertionError(f"{label}: topk_slots is outside [0,{budget})")
    if torch.unique(slots).numel() != TOPK:
        raise AssertionError(f"{label}: topk_slots is not unique")
    if not torch.equal(slots, new_selected_slots):
        raise AssertionError(f"{label}: published slots differ from updated cache state")
