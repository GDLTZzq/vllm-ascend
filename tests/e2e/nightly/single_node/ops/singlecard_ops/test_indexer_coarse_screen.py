"""Single-op probe for npu_indexer_coarse_screen (PIVOT-Refine coarse screen).

The op is npu_lightning_indexer restricted to sparse_count=4096 / sparse_mode=0:
score[r, p] = sum_h w_bar[r, h] * relu(q_bar[r, h] . k[r, p]) over the whole PA
prefix p in [0, L), top-sparse_count positions out, -1 padded.

The reason this probe exists (rather than trusting the lightning_indexer
lineage) is the `MrgSort4` mrgDstNum > 3072 branch in
`indexer_coarse_screen_vector.h`, which splits a 4096-wide merge into
2048/1024/1024(+512) queues. No other caller in this repo drives
sparse_count=4096 (every lightning_indexer call site uses 2048/sparse_mode=3),
so that path has zero coverage. The score-descending assertion below is what
catches a width bug there: a truncated or mis-ordered merge still yields the
right *set* (hence the set check alone is not enough) but the wrong order.

Contract assumptions verified against the hosts/kernels:
  - query TND [T=K, H, 128], 1 row per request;
  - aslq is CUMULATIVE ([1..K]) because LAYOUT_T == TND;
  - aslk is NON-cumulative (K_LAYOUT_T == PA_BSND) and must be L, the prefix
    BEFORE the step's g tokens -- feeding L+g would add g candidates;
  - key PA_BSND [num_blocks, block_size, 1, 128], block_table [K, max_blocks];
  - output [K, 1, 4096] int32, -1 padded; indices are logical positions in
    [0, L), NOT cache slots (the kernel derives slots from block_table).
"""

import os

import pytest
import torch
import torch_npu

from vllm_ascend.attention.pivot_indexer import _COARSE_BUDGET, _coarse_screen
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

# name -> per-request prefix lengths L (= aslk). One op call drives all K
# requests, so the L list also exercises the per-request S2 loop bounds.
CASES = [
    ("tiny", [64, 128, 300, 512]),  # L << budget: whole prefix returned, -1 pad
    ("block_aligned", [1024, 1025, 2048, 4095]),  # straddles 512-chunk bounds
    ("truncated", [4096, 4097, 5000, 8192]),  # L > budget: real top-4096 pick
    ("long", [65536, 30000, 12345, 4096]),  # deep S2 walk on one core
]


def _proxy_scores(q_bar, w_bar, k_cache, block_table, block_size, seq_lens):
    """fp32 torch replica of the op's score, [K, L_max], -inf beyond L.

    Only used to score the op's returned positions and to derive the expected
    top-k order; correctness of the reference itself comes from _coarse_screen.
    """
    k_dim = q_bar.shape[0]
    l_max = int(seq_lens.max())
    kc = k_cache.reshape(-1, k_cache.shape[-1]).float()
    pos = torch.arange(l_max, dtype=torch.int64)
    slots = block_table[:, pos // block_size] * block_size + pos % block_size
    k_all = kc[slots.reshape(-1)].view(k_dim, l_max, -1)
    att = torch.relu(torch.bmm(q_bar.float(), k_all.transpose(1, 2)))
    score = (att * w_bar.float().unsqueeze(-1)).sum(dim=1)
    beyond = pos.unsqueeze(0) >= seq_lens.to(torch.int64).unsqueeze(1)
    return score.masked_fill(beyond, float("-inf"))


def _build_data(seq_lens, seed=0):
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

    q_bar = torch.randn(k, H, DH).to(torch.bfloat16)
    w_bar = torch.randn(k, H).to(torch.bfloat16)
    k_cache = torch.randn(nxt, BLOCK, 1, DH).to(torch.bfloat16)
    seq = torch.tensor([int(length) for length in seq_lens], dtype=torch.int32)
    return q_bar, w_bar, k_cache, block_table, seq


@pytest.mark.parametrize(
    "seq_lens", [case[1] for case in CASES], ids=[case[0] for case in CASES]
)
def test_indexer_coarse_screen_matches_torch_reference(seq_lens):
    q_bar, w_bar, k_cache, block_table, seq = _build_data(seq_lens)
    k = len(seq_lens)

    # ---- reference: the production torch path, on CPU -------------------
    score = _proxy_scores(q_bar, w_bar, k_cache, block_table, BLOCK, seq)
    ref = _coarse_screen(
        q_bar, w_bar, (None, None, k_cache), block_table, BLOCK, seq
    )
    assert tuple(ref.shape) == (k, _COARSE_BUDGET)

    # ---- op ------------------------------------------------------------
    dev = torch.device(DEVICE)
    out = torch.ops._C_ascend.npu_indexer_coarse_screen(
        q_bar.to(dev).contiguous(),
        k_cache.to(dev).contiguous(),
        w_bar.to(dev).contiguous(),  # [K, H] bf16, same dtype as query/key
        actual_seq_lengths_query=torch.arange(1, k + 1, dtype=torch.int32, device=dev),
        actual_seq_lengths_key=seq.to(dev),
        block_table=block_table.to(dev),
        layout_query="TND",
        layout_key="PA_BSND",
        sparse_count=_COARSE_BUDGET,
    )
    assert tuple(out.shape) == (k, 1, _COARSE_BUDGET), out.shape
    assert out.dtype == torch.int32, out.dtype
    cols = out.cpu().view(k, _COARSE_BUDGET).to(torch.int64)

    for r, length in enumerate(seq_lens):
        length = int(length)
        row = cols[r]
        # Anything negative must be the -1 pad, never uninitialised garbage.
        assert (row[row < 0] == -1).all(), f"row {r} padding is not -1"

        pos_op = row[row >= 0]
        pos_ref = ref[r][ref[r] >= 0]
        expect = min(length, _COARSE_BUDGET)

        assert pos_op.numel() == expect, (
            f"row {r}: op returned {pos_op.numel()} valid positions, expected {expect}"
        )
        # Duplicates would silently shrink the candidate superset.
        assert pos_op.unique().numel() == pos_op.numel(), f"row {r} has duplicates"
        assert (pos_op < length).all(), f"row {r} left its own prefix"
        assert set(pos_op.tolist()) == set(pos_ref.tolist()), (
            f"row {r}: candidate set differs from the torch reference by "
            f"{sorted(set(pos_op.tolist()) ^ set(pos_ref.tolist()))[:16]}"
        )

        # ---- order: catches the 4096-wide MrgSort branch -----------------
        row_score = score[r]
        op_scores = row_score[pos_op]
        scale = max(1.0, float(op_scores.abs().max()))
        diffs = op_scores[:-1] - op_scores[1:]
        assert (diffs >= -1e-2 * scale).all(), (
            f"row {r}: op output is not score-descending "
            f"(worst inversion {float(-diffs.min()):.4g} at "
            f"{int(diffs.argmin())})"
        )
        ref_top = torch.topk(row_score, expect, dim=-1).values
        torch.testing.assert_close(op_scores, ref_top, rtol=1e-2, atol=1e-2)


def test_indexer_coarse_screen_rejects_other_sparse_count():
    q_bar, w_bar, k_cache, block_table, seq = _build_data([1024, 1024])
    dev = torch.device(DEVICE)
    with pytest.raises(RuntimeError):
        torch.ops._C_ascend.npu_indexer_coarse_screen(
            q_bar.to(dev).contiguous(),
            k_cache.to(dev).contiguous(),
            w_bar.to(dev).contiguous(),
            actual_seq_lengths_query=torch.arange(1, 3, dtype=torch.int32, device=dev),
            actual_seq_lengths_key=seq.to(dev),
            block_table=block_table.to(dev),
            layout_query="TND",
            layout_key="PA_BSND",
            sparse_count=2048,
        )
