/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 *
 * arch35 (Ascend950) adapter for the SHARED A3 op host ABI.
 *
 * This header is included ONLY on arch35 kernel builds (see the include gate in
 * the root nanovllm_fused_li_manage_mtp.cpp). It drives the self-contained C8
 * fp8 LD engine copied into arch35/ under the A3 I/O contract: 13 inputs + 7
 * outputs in the A3 order.  The orchestration below reproduces the validated
 * A5 C8 entry (a5_fused_li_manage_mtp_c8.cpp:48-189) but only the bs<=16 LD
 * branch (this tree carries no batchSize>16 GL engine), reading A3-ordered GM
 * parameters and mapping them into the C8 Init argument slots:
 *   A3 index_weights            -> weights
 *   A3 query_dequant_scale      -> queryDequantScale
 *   A3 query                    -> query
 *   A3 index_key_dequant_scale  -> keyDequantScale
 *   A3 index_key_cache          -> key
 *   A3 index_block_table        -> blockTable
 *   A3 actual_seq_lengths_query -> actualSeqLengthsQuery
 *   A3 num_cache_tokens         -> cacheTokens
 *   A3 offload_seq_lengths_key  -> candidateLens
 *   A3 req_pool_entries         -> reqPoolEntries
 *   A3 cache_slots_pool         -> cacheSlotsPool (in-place update)
 * Outputs map positionally 1:1 into the A3 output tensors.
 */

#ifndef FUSED_LI_MANAGE_MTP_ARCH35_H
#define FUSED_LI_MANAGE_MTP_ARCH35_H

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "a5_fused_li_manage_mtp_c8_tiling.h"
#include "a5_fused_li_manage_mtp_c8_workspace.h"
#include "a5_fused_li_manage_mtp_c8_qli.h"
#include "a5_fused_li_manage_mtp_c8_union.h"

// Header-level global using, matching arch22/fused_li_manage_mtp_union.h and
// arch22/lightning_indexer_vector.h: this adapter is parsed when included at
// the top of the root .cpp, before the root's own `using namespace AscendC;`,
// so unqualified SyncAll/GetBlockIdx/GM_ADDR below must resolve here.
using namespace AscendC;

// The shared A3 schema fixes the miss/union outputs at [B, 32768] (see
// op_host/fused_li_manage_mtp_infershape.cpp).  The C8 union emits one row per
// request, so it must stride its GM writes by that width; its internal
// UNION_CAPACITY (UNION_ROUTES * 2048 = 8192) is a workspace bound, not the
// output row pitch.
constexpr uint32_t A3_MISS_ROW_STRIDE = 32768U;

// Runs on every launched core of the arch35 kernel.  Parameters are the 22 GM
// addresses of the shared A3 kernel entry (same order); the c8 path ignores the
// two A3-only inputs (actual_seq_lengths_key, request_state) and aliases
// cache_slots_pool_out to the in-place cache_slots_pool input.
// Launch-level macros (KERNEL_TASK_TYPE_DEFAULT / REGISTER_TILING_DEFAULT /
// GET_TILING_DATA) and the TPipe/workspace bindings live in the __global__
// kernel entry (arch35 branch of nanovllm_fused_li_manage_mtp.cpp), which
// passes them in; helpers must not re-open the task/tiling bindings.
__aicore__ inline void arch35_run_fused_li_manage_mtp(
    AscendC::TPipe *pipe,
    const A5FusedLiManageMtpC8TilingData *tilingData,
    __gm__ uint8_t *userWorkspace,
    __gm__ uint8_t *weights, __gm__ uint8_t *queryDequantScale,
    __gm__ uint8_t *query, __gm__ uint8_t *keyDequantScale,
    __gm__ uint8_t *key, __gm__ uint8_t *blockTable,
    __gm__ uint8_t *actualSeqLengthsQuery,
    __gm__ uint8_t *actualSeqLengthsKey,
    __gm__ uint8_t *offloadSeqLengthsKey, __gm__ uint8_t *cacheTokens,
    __gm__ uint8_t *requestState, __gm__ uint8_t *reqPoolEntries,
    __gm__ uint8_t *cacheSlots, __gm__ uint8_t *topkSourceIds,
    __gm__ uint8_t *topkSlots, __gm__ uint8_t *topkMissCounts,
    __gm__ uint8_t *missSrcIds, __gm__ uint8_t *missDstSlots,
    __gm__ uint8_t *missCount, __gm__ uint8_t *cacheSlotsOut)
{
    // A3-only on the c8 (M1 steady-state) path: request_state is -1 for every
    // request (host-enforced), actual_seq_lengths_key is unused, and
    // cache_slots_pool_out aliases the cache_slots_pool input the C8 engine
    // updates in place.
    (void)actualSeqLengthsKey;
    (void)requestState;
    (void)cacheSlotsOut;

    using namespace a5_fused_li_manage_mtp_c8_workspace;
    const uint64_t scoreStride = tilingData->scoreWorkspaceStride;
    const uint64_t batchSize = tilingData->batchSize;
    // The helpers in the shared workspace header are host-side constexpr
    // functions for tiling.  Expand the same arithmetic here so an __aicore__
    // kernel never calls a host function.
    const uint64_t routePairOffset = scoreStride * batchSize;
    const uint64_t routePairBytes =
        batchSize * UNION_CAPACITY * 2U * sizeof(int32_t);
    const uint64_t routeThresholdOffset = routePairOffset + routePairBytes;
    const uint64_t routeThresholdBytes =
        batchSize * ROUTES * THRESHOLD_STRIDE * sizeof(uint16_t);
    GM_ADDR routePairRows = userWorkspace + routePairOffset;
    GM_ADDR routeThresholds = userWorkspace + routeThresholdOffset;
    // QLI LD merge workspace (ldScore/ldIndex) follows the thresholds; the host
    // allocates it via a5_fused_li_manage_mtp_c8_workspace::TotalBytes.
    GM_ADDR ldWorkspace =
        userWorkspace + routeThresholdOffset + routeThresholdBytes;
    // The public per-query output is also the Stage-1 -> Stage-2 hand-off,
    // keeping one authoritative count and avoiding a redundant GM copy.
    GM_ADDR routeMissCounts = topkMissCounts;

    // bs<=16 LD fork only (this tree carries the C8 LD engine, no GL path).
    // Host tiling rejects batchSize > 16 on the fp8 branch.
    a5_fused_li_manage_mtp_c8_impl::QuantLiMtpPhase qli(pipe, tilingData);
    qli.Init(
        query, key, weights, queryDequantScale, keyDequantScale,
        actualSeqLengthsQuery, cacheTokens, offloadSeqLengthsKey,
        reqPoolEntries, cacheSlots, blockTable, routePairRows,
        topkSourceIds, topkSlots, routeThresholds, routeMissCounts,
        userWorkspace, ldWorkspace);
    qli.Process();

    if ASCEND_IS_AIV {
        // Every (request, gS1) QLI task owner publishes route pairs/counts with
        // MTE3; every request owner consumes all four routes through MTE2 only
        // after this global barrier.
        AscendC::SyncAll();
        if ((GetBlockIdx() & 1U) == 0U) {
            pipe->Reset();
            a5_fused_li_manage_mtp_c8_impl::OrderedMissUnion unionOp;
            unionOp.Init(
                routePairRows, routeThresholds, routeMissCounts,
                userWorkspace, offloadSeqLengthsKey, reqPoolEntries,
                cacheSlots, missSrcIds, missDstSlots, missCount,
                topkSourceIds, topkSlots, cacheTokens,
                tilingData->sourceCapacity, tilingData->batchSize,
                A3_MISS_ROW_STRIDE, pipe);
            unionOp.Process(GetBlockIdx() / 2U, tilingData->usedCoreNum);
        }
    }
}

#endif  // FUSED_LI_MANAGE_MTP_ARCH35_H
