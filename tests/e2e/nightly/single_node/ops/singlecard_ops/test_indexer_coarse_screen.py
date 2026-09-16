"""Single-op probe for npu_indexer_coarse_screen (PIVOT-Refine coarse screen).

The op emits the WHOLE step-2 candidate row: the ranked coarse screen AND the
verbatim group-window union, fused so python needs no `_inject_local_window`.

For each request r with group width g (query rows [r*g, r*g+g)) and natural key
prefix length aslk = actual_seq_lengths_key[r] = L + g, let

    lo = max(0, aslk - 2g + 1)

(the lower bound of the group's local window [L-g+1, L+g)). The op must produce
ONE row of width Align8(sparse_count + 2g - 1) whose valid front is exactly

    [ ranked top-min(lo, sparse_count) of the domain-[0, lo) proxy score ]
    [ the window union [lo, aslk) verbatim, ascending ]
    [ -1 pad to Align8(sparse_count + 2g - 1) ]

so the valid COUNT == min(lo, sparse_count) + (aslk - lo). The two segments are
disjoint by construction (lo IS the window's own lower bound), which is why the
window needs no dedup and no membership test. Rows with aslk <= 2g-1 (first
token) have lo == 0, an empty ranked segment, and are [0 .. aslk-1] + -1 pad:

    score[r, p] = sum_h w_bar[r, h] * relu(q_bar[r, h] . k[r, p]), p in [0, lo)
    q_bar/w_bar = the request's g-row mean proxy (row_weights all ones here =>
    plain mean, byte-identical to python's q_bar/w_bar)

Contract assumptions verified against the hosts/kernels:
  - query TND [T=K*g, H, 128], g pooled rows per request;
  - aslq is CUMULATIVE with step g ([g, 2g, .., K*g]) because LAYOUT_T == TND;
  - aslk is NON-cumulative (K_LAYOUT_T == PA_BSND) and is the NATURAL length
    L+g, not L: the op itself carves the ranked domain [0, lo) out of it;
  - row_weights [K, g] bf16 == the pooling group width (tiling requires
    rank-2 with last dim in (0, 16]);
  - key PA_BSND [num_blocks, block_size, 1, 128], block_table [K, max_blocks];
  - output [K, 1, Align8(4096 + 2g - 1)] int32, valid front + -1 pad; indices
    are logical positions in [0, aslk), NOT cache slots (the kernel derives
    slots from block_table).

The reason this probe exists (rather than trusting the lightning_indexer
lineage) is the `MrgSort4` mrgDstNum > 3072 branch in
`indexer_coarse_screen_vector.h`, which splits a 4096-wide merge into
2048/1024/1024(+512) queues. No other caller in this repo drives
sparse_count=4096 (every lightning_indexer call site uses 2048/sparse_mode=3),
so that path has zero coverage. The score-descending assertion on the RANKED
prefix below is what catches a width bug there: a truncated or mis-ordered
merge still yields the right *set* (hence the set check alone is not enough)
but the wrong order. The window tail is ordered by position instead, so it is
asserted verbatim against [lo, aslk).
"""

import os

import pytest
import torch
import torch_npu

from vllm_ascend.attention.pivot_indexer import (
    _COARSE_BUDGET,
    _coarse_screen,
    _inject_local_window,
)
from vllm_ascend.utils import enable_custom_op

enable_custom_op()

# The torch _coarse_screen reference must not itself take the op path, else the
# probe would compare the op against the op.
os.environ.pop("VLLM_ASCEND_PIVOT_COARSE_USE_OP", None)

H = 64  # indexer head num (tiling caps it at QUERY_HEAD_NUM_LIMIT = 64)
DH = 128  # head dim (tiling hard-requires 128)
BLOCK = 128  # PA block size (multiple of 16, <= 1024)
DEVICE_ID = 0
DEVICE = f"npu:{DEVICE_ID}"

torch_npu.npu.set_device(DEVICE_ID)


def _row_width(g):
    """The op's fixed per-row output width = Align8(sparse_count + 2g - 1)."""
    return (_COARSE_BUDGET + 2 * g - 1 + 7) & ~7


# (name, per-request natural prefix lengths aslk, group width g). One op call
# drives all K requests, so the aslk list also exercises the per-request S2
# loop bounds. aslk >= g always holds (a request has g query rows).
CASES = [
    # lo == 0 (aslk <= 2g-1) mixed with tiny lo: ranked segment empty / 1-wide.
    ("first_token", [4, 8, 12, 300], 4),
    # lo < budget: the whole ranked domain is recalled, -1 pad after the tail.
    ("short", [64, 128, 300, 512], 4),
    ("block_aligned", [1024, 1025, 2048, 4095], 4),
    # lo > budget: a real top-4096 pick out of the domain [0, lo).
    ("truncated", [4096, 4097, 5000, 8192], 4),
    ("long", [65536, 30000, 12345, 4096], 4),
    # widest local window (2g-1 = 31) at the host's max group.
    ("g16", [4600, 400, 16], 16),
]


def _proxy_scores(q_bar, w_bar, k_cache, block_table, block_size, n_pos):
    """fp32 torch replica of the op's score over key positions [0, n_pos).

    Only used to score the op's returned positions and to derive the expected
    ranked order; correctness of the reference itself comes from _coarse_screen.
    """
    k_dim = q_bar.shape[0]
    kc = k_cache.reshape(-1, k_cache.shape[-1]).float()
    pos = torch.arange(n_pos, dtype=torch.int64)
    slots = block_table[:, pos // block_size] * block_size + pos % block_size
    k_all = kc[slots.reshape(-1)].view(k_dim, n_pos, -1)
    att = torch.relu(torch.bmm(q_bar.float(), k_all.transpose(1, 2)))
    return (att * w_bar.float().unsqueeze(-1)).sum(dim=1)


def _build_data(seq_lens, g, seed=0):
    torch.manual_seed(seed)
    k = len(seq_lens)
    blocks_per_req = [(int(length) + BLOCK - 1) // BLOCK for length in seq_lens]
    max_blocks = max(blocks_per_req)

    # Block 0 is left unreferenced padding; real blocks are numbered from 1 so
    # an out-of-range block_table entry is distinguishable from a real one.
    block_table = torch.zeros((k, max_blocks), dtype=torch.int32)
    nxt = 1
    for r, nb in enumerate(blocks_per_req):
        block_table[r, :nb] = torch.arange(nxt, nxt + nb, dtype=torch.int32)
        nxt += nb

    q = torch.randn(k * g, H, DH).to(torch.bfloat16)
    w = torch.randn(k * g, H).to(torch.bfloat16)
    k_cache = torch.randn(nxt, BLOCK, 1, DH).to(torch.bfloat16)
    seq = torch.tensor([int(length) for length in seq_lens], dtype=torch.int32)
    # row_weights all ones => the kernel's weighted mean == python's .mean(dim=1)
    row_weights = torch.ones(k, g).to(torch.bfloat16)
    return q, w, k_cache, block_table, seq, row_weights


def _mean_proxies(q, w, k, g):
    """Pool the g query rows per request in fp32 (mirrors the kernel's cast-
    accumulate-cast mean), so the reference proxies match the op's byte-wise."""
    q_bar = q.float().view(k, g, H, DH).mean(dim=1).to(torch.bfloat16)
    w_bar = w.float().view(k, g, H).mean(dim=1).to(torch.bfloat16)
    return q_bar, w_bar


@pytest.mark.parametrize(
    "seq_lens,g", [(case[1], case[2]) for case in CASES], ids=[case[0] for case in CASES]
)
def test_indexer_coarse_screen_matches_torch_reference(seq_lens, g):
    q, w, k_cache, block_table, seq, row_weights = _build_data(seq_lens, g)
    k = len(seq_lens)
    width = _row_width(g)

    # ---- reference: the production torch path, on CPU -------------------
    q_bar, w_bar = _mean_proxies(q, w, k, g)
    aslk = seq.to(torch.int64)
    lo = (aslk - 2 * g + 1).clamp(min=0)  # ranked domain [0, lo)
    ref, ref_aslk = _inject_local_window(
        _coarse_screen(q_bar, w_bar, (None, None, k_cache), block_table, BLOCK, lo),
        aslk,
        g,
    )
    assert tuple(ref.shape)[0] == k
    # The reference row is compact: its valid count is the refine op's aslk bound.
    assert ((ref >= 0).sum(dim=1).to(torch.int64) == ref_aslk.to(torch.int64)).all()

    # ---- op ------------------------------------------------------------
    dev = torch.device(DEVICE)
    out = torch.ops._C_ascend.npu_indexer_coarse_screen(
        q.to(dev).contiguous(),
        k_cache.to(dev).contiguous(),
        w.to(dev).contiguous(),  # [K*g, H] bf16, same dtype as query/key
        row_weights.to(dev).contiguous(),
        actual_seq_lengths_query=torch.arange(
            g, k * g + 1, g, dtype=torch.int32, device=dev
        ),
        actual_seq_lengths_key=seq.to(dev),
        block_table=block_table.to(dev),
        layout_query="TND",
        layout_key="PA_BSND",
        sparse_count=_COARSE_BUDGET,
    )
    assert tuple(out.shape) == (k, 1, width), out.shape
    assert out.dtype == torch.int32, out.dtype
    cols = out.cpu().view(k, width).to(torch.int64)

    # Score the whole natural prefix once; both segments index into it.
    score = _proxy_scores(q_bar, w_bar, k_cache, block_table, BLOCK, int(aslk.max()))

    for r, length in enumerate(seq_lens):
        length = int(length)
        lo_r = int(lo[r])
        n_ranked = min(lo_r, _COARSE_BUDGET)
        row = cols[r]
        # Anything negative must be the -1 pad, never uninitialised garbage.
        assert (row[row < 0] == -1).all(), f"row {r} padding is not -1"

        pos_op = row[row >= 0]
        pos_ref = ref[r][ref[r] >= 0]
        expect = n_ranked + (length - lo_r)

        assert pos_op.numel() == expect, (
            f"row {r}: op returned {pos_op.numel()} valid positions, expected "
            f"min(lo={lo_r}, {_COARSE_BUDGET}) + (aslk={length} - lo) = {expect}"
        )
        # Duplicates would silently shrink the candidate superset.
        assert pos_op.unique().numel() == pos_op.numel(), f"row {r} has duplicates"
        assert (pos_op < length).all(), f"row {r} left its own prefix"
        assert set(pos_op.tolist()) == set(pos_ref.tolist()), (
            f"row {r}: candidate set differs from the torch reference by "
            f"{sorted(set(pos_op.tolist()) ^ set(pos_ref.tolist()))[:16]}"
        )

        # ---- order -------------------------------------------------------
        # Tail: the verbatim window union [lo, aslk), ascending by position.
        assert pos_op[n_ranked:].tolist() == list(range(lo_r, length)), (
            f"row {r}: window tail is not the verbatim union [{lo_r}, {length})"
        )
        # Ranked prefix: top-min(lo, budget) of the domain [0, lo), score-
        # descending -- this is what catches the 4096-wide MrgSort branch. The
        # value comparison (not index) keeps it tie-safe. Empty when lo == 0
        # (first-token rows), where the row is the window tail alone.
        if n_ranked:
            op_ranked = pos_op[:n_ranked]
            dom_score = score[r][:lo_r]
            op_scores = dom_score[op_ranked]
            scale = max(1.0, float(op_scores.abs().max()))
            diffs = op_scores[:-1] - op_scores[1:]
            assert (diffs >= -1e-2 * scale).all(), (
                f"row {r}: ranked prefix is not score-descending "
                f"(worst inversion {float(-diffs.min()):.4g} at {int(diffs.argmin())})"
            )
            torch.testing.assert_close(
                op_scores, torch.topk(dom_score, n_ranked).values, rtol=1e-2, atol=1e-2
            )


def test_indexer_coarse_screen_rejects_other_sparse_count():
    g = 4
    q, w, k_cache, block_table, seq, row_weights = _build_data([1024, 1024], g)
    dev = torch.device(DEVICE)
    with pytest.raises(RuntimeError):
        torch.ops._C_ascend.npu_indexer_coarse_screen(
            q.to(dev).contiguous(),
            k_cache.to(dev).contiguous(),
            w.to(dev).contiguous(),
            row_weights.to(dev).contiguous(),
            actual_seq_lengths_query=torch.arange(
                g, 2 * g + 1, g, dtype=torch.int32, device=dev
            ),
            actual_seq_lengths_key=seq.to(dev),
            block_table=block_table.to(dev),
            layout_query="TND",
            layout_key="PA_BSND",
            sparse_count=2048,
        )
