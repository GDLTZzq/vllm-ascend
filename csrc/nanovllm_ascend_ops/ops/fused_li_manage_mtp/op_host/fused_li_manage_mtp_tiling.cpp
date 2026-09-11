/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */

#include "fused_li_manage_mtp_tiling.h"
#include <algorithm>
#include <cstdint>
#include <limits>
#include "../op_kernel/fused_li_manage_mtp_template_tiling_key.h"
// The Ascend950 (arch35) fp8 path fills A5FusedLiManageMtpC8TilingData and
// sizes the C8 workspace with the shared constexpr helpers. These headers are
// host-only plain structs / constexpr functions; the fp8 branch is guarded by
// the query dtype check in TilingForNanovllmFusedLiManageMtp below.
#include "../op_kernel/arch35/a5_fused_li_manage_mtp_c8_tiling.h"
#include "../op_kernel/arch35/a5_fused_li_manage_mtp_c8_workspace.h"

using namespace ge;
using namespace AscendC;

namespace optiling {

ge::graphStatus FusedLiManageMtpTiling::GetNpuInfo(FusedLiManageMtpTilingInfo &tilingInfo) const
{
    if (context_->GetNodeName() == nullptr) {
        OPS_LOG_E("NanovllmFusedLiManageMtp", "opName got from TilingContext is nullptr.");
        return ge::GRAPH_FAILED;
    }
    tilingInfo.opName = context_->GetNodeName();
    tilingInfo.platformInfo = context_->GetPlatformInfo();
    OPS_ERR_IF(tilingInfo.platformInfo == nullptr, OPS_LOG_E(tilingInfo.opName, "GetPlatformInfo is nullptr."),
               return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(tilingInfo.platformInfo);
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    OPS_ERR_IF(aicNum == 0 || aivNum == 0, OPS_LOG_E(tilingInfo.opName, "num of core obtained is 0."),
               return ge::GRAPH_FAILED);

    tilingInfo.socVersion = ascendcPlatform.GetSocVersion();
    // No SocVersion allowlist here.  The op is compiled and registered only for
    // the SoCs in the host def (ascend910b / ascend910_93 / ascend950), and an
    // allowlist cannot name ASCEND950 unconditionally because this shared host
    // tiling also compiles against CANN 8.5.1 (no ASCEND950 enum).  The
    // aic/aiv > 0 check above plus AddConfig registration is the real guard.
    OPS_ERR_IF(context_->GetWorkspaceSizes(1) == nullptr,
               OPS_LOG_E(tilingInfo.opName, "workspace size buffer is nullptr."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(context_->GetRawTilingData() == nullptr,
               OPS_LOG_E(tilingInfo.opName, "raw tiling data is nullptr."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedLiManageMtpTiling::GetTensorInfo(FusedLiManageMtpTilingInfo &tilingInfo) const
{
    auto &op = tilingInfo.opParamInfo;
    op.query.desc = context_->GetInputDesc(QUERY_INDEX);
    op.query.shape = context_->GetInputShape(QUERY_INDEX);
    op.key.desc = context_->GetInputDesc(KEY_INDEX);
    op.key.shape = context_->GetInputShape(KEY_INDEX);
    op.weights.desc = context_->GetInputDesc(WEIGHTS_INDEX);
    op.weights.shape = context_->GetInputShape(WEIGHTS_INDEX);
    op.reqPoolEntries.desc = context_->GetInputDesc(REQ_POOL_ENTRIES_INDEX);
    op.reqPoolEntries.shape = context_->GetInputShape(REQ_POOL_ENTRIES_INDEX);
    op.cacheSlots.desc = context_->GetInputDesc(CACHE_SLOTS_INDEX);
    op.cacheSlots.shape = context_->GetInputShape(CACHE_SLOTS_INDEX);
    op.cacheTokens.desc = context_->GetInputDesc(CACHE_TOKENS_INDEX);
    op.cacheTokens.shape = context_->GetInputShape(CACHE_TOKENS_INDEX);
    op.actualSeqLengths.desc = context_->GetInputDesc(OFFLOAD_SEQ_K_INDEX);
    op.actualSeqLengths.shape = context_->GetInputShape(OFFLOAD_SEQ_K_INDEX);
    op.blockTable.desc = context_->GetInputDesc(BLOCK_TABLE_INDEX);
    op.blockTable.shape = context_->GetInputShape(BLOCK_TABLE_INDEX);
    op.topkIndexOut.desc = context_->GetOutputDesc(TOPK_INDEX);
    op.topkIndexOut.shape = context_->GetOutputShape(TOPK_INDEX);
    op.topkSlotsOut.desc = context_->GetOutputDesc(TOPK_SLOTS_INDEX);
    op.topkSlotsOut.shape = context_->GetOutputShape(TOPK_SLOTS_INDEX);
    uint32_t missCountIndex = mtp_ ? 5U : MISS_COUNT_INDEX;
    uint32_t cacheSlotsOutIndex = mtp_ ? 6U : CACHE_SLOTS_OUT_INDEX;
    if (mtp_) {
        op.topkMissCountOut.desc = context_->GetOutputDesc(2U);
        op.topkMissCountOut.shape = context_->GetOutputShape(2U);
        op.missSrcOut.desc = context_->GetOutputDesc(3U);
        op.missSrcOut.shape = context_->GetOutputShape(3U);
        op.missSlotsOut.desc = context_->GetOutputDesc(4U);
        op.missSlotsOut.shape = context_->GetOutputShape(4U);
    }
    op.missCountOut.desc = context_->GetOutputDesc(missCountIndex);
    op.missCountOut.shape = context_->GetOutputShape(missCountIndex);
    op.cacheSlotsOut.desc = context_->GetOutputDesc(cacheSlotsOutIndex);
    op.cacheSlotsOut.shape = context_->GetOutputShape(cacheSlotsOutIndex);

    OPS_ERR_IF(op.query.desc == nullptr || op.query.shape == nullptr,
               OPS_LOG_E(tilingInfo.opName, "query desc/shape is nullptr."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(op.key.desc == nullptr || op.key.shape == nullptr,
               OPS_LOG_E(tilingInfo.opName, "key desc/shape is nullptr."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(op.weights.desc == nullptr || op.weights.shape == nullptr,
               OPS_LOG_E(tilingInfo.opName, "weights desc/shape is nullptr."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(op.reqPoolEntries.desc == nullptr || op.reqPoolEntries.shape == nullptr,
               OPS_LOG_E(tilingInfo.opName, "req_pool_entries desc/shape is nullptr."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(op.cacheSlots.desc == nullptr || op.cacheSlots.shape == nullptr,
               OPS_LOG_E(tilingInfo.opName, "cache_slots desc/shape is nullptr."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(op.cacheTokens.desc == nullptr || op.cacheTokens.shape == nullptr,
               OPS_LOG_E(tilingInfo.opName, "cache_tokens desc/shape is nullptr."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(op.actualSeqLengths.desc == nullptr || op.actualSeqLengths.shape == nullptr,
               OPS_LOG_E(tilingInfo.opName, "offload_seq_lengths_key desc/shape is nullptr."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(op.blockTable.desc == nullptr || op.blockTable.shape == nullptr,
               OPS_LOG_E(tilingInfo.opName, "block_table desc/shape is nullptr."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(op.topkIndexOut.desc == nullptr || op.topkIndexOut.shape == nullptr,
               OPS_LOG_E(tilingInfo.opName, "topk_index desc/shape is nullptr."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(op.topkSlotsOut.desc == nullptr || op.topkSlotsOut.shape == nullptr,
               OPS_LOG_E(tilingInfo.opName, "topk_slots desc/shape is nullptr."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(op.missCountOut.desc == nullptr || op.missCountOut.shape == nullptr,
               OPS_LOG_E(tilingInfo.opName, "miss_count desc/shape is nullptr."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(op.cacheSlotsOut.desc == nullptr || op.cacheSlotsOut.shape == nullptr,
               OPS_LOG_E(tilingInfo.opName, "cache_slots output desc/shape is nullptr."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(mtp_ && (op.topkMissCountOut.desc == nullptr ||
                        op.topkMissCountOut.shape == nullptr ||
                        op.missSrcOut.desc == nullptr || op.missSrcOut.shape == nullptr ||
                        op.missSlotsOut.desc == nullptr || op.missSlotsOut.shape == nullptr),
               OPS_LOG_E(tilingInfo.opName, "MTP union miss outputs are nullptr."),
               return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedLiManageMtpTiling::CheckDtype(const FusedLiManageMtpTilingInfo &tilingInfo) const
{
    const auto &op = tilingInfo.opParamInfo;
    ge::DataType qType = op.query.desc->GetDataType();
    ge::DataType kType = op.key.desc->GetDataType();
    ge::DataType wType = op.weights.desc->GetDataType();
    OPS_ERR_IF(qType != kType || qType != wType,
               OPS_LOG_E(tilingInfo.opName, "query/key/weights dtype must match."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(qType != ge::DT_FLOAT16 && qType != ge::DT_BF16,
               OPS_LOG_E(tilingInfo.opName, "query/key/weights dtype must be fp16 or bf16."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(context_->GetInputDesc(QUERY_DEQUANT_SCALE_INDEX) == nullptr ||
                   context_->GetInputDesc(KEY_DEQUANT_SCALE_INDEX) == nullptr ||
                   context_->GetInputDesc(QUERY_DEQUANT_SCALE_INDEX)->GetDataType() != ge::DT_FLOAT ||
                   context_->GetInputDesc(KEY_DEQUANT_SCALE_INDEX)->GetDataType() != ge::DT_FLOAT,
               OPS_LOG_E(tilingInfo.opName, "dequant scales must be fp32."),
               return ge::GRAPH_FAILED);
    for (uint32_t input = BLOCK_TABLE_INDEX; input <= CACHE_SLOTS_INDEX; ++input) {
        OPS_ERR_IF(context_->GetInputDesc(input) == nullptr ||
                       context_->GetInputDesc(input)->GetDataType() != ge::DT_INT32,
                   OPS_LOG_E(tilingInfo.opName, "metadata inputs must be int32."),
                   return ge::GRAPH_FAILED);
    }
    OPS_ERR_IF(op.cacheSlots.desc->GetDataType() != ge::DT_INT32,
               OPS_LOG_E(tilingInfo.opName, "cache_slots dtype must be int32."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(op.reqPoolEntries.desc->GetDataType() != ge::DT_INT32,
               OPS_LOG_E(tilingInfo.opName, "req_pool_entries dtype must be int32."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(op.cacheTokens.desc->GetDataType() != ge::DT_INT32,
               OPS_LOG_E(tilingInfo.opName, "cache_tokens dtype must be int32."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(op.actualSeqLengths.desc->GetDataType() != ge::DT_INT32,
               OPS_LOG_E(tilingInfo.opName, "offload_seq_lengths_key dtype must be int32."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(op.blockTable.desc->GetDataType() != ge::DT_INT32,
               OPS_LOG_E(tilingInfo.opName, "block_table dtype must be int32."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(op.topkIndexOut.desc->GetDataType() != ge::DT_INT32 ||
                   op.topkSlotsOut.desc->GetDataType() != ge::DT_INT32 ||
                   (mtp_ && op.topkMissCountOut.desc->GetDataType() != ge::DT_INT32) ||
                   op.missCountOut.desc->GetDataType() != ge::DT_INT32 ||
                    op.cacheSlotsOut.desc->GetDataType() != ge::DT_INT32 ||
                    (mtp_ && (op.missSrcOut.desc->GetDataType() != ge::DT_INT32 ||
                              op.missSlotsOut.desc->GetDataType() != ge::DT_INT32)),
               OPS_LOG_E(tilingInfo.opName,
                         "topk_index/topk_slots/miss_count/cache_slots output must be int32."),
               return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedLiManageMtpTiling::CheckShape(FusedLiManageMtpTilingInfo &tilingInfo) const
{
    const auto &op = tilingInfo.opParamInfo;
    const auto &qShape = op.query.shape->GetStorageShape();
    const auto &kShape = op.key.shape->GetStorageShape();
    const auto &wShape = op.weights.shape->GetStorageShape();
    const auto &reqPoolShape = op.reqPoolEntries.shape->GetStorageShape();
    const auto &cacheShape = op.cacheSlots.shape->GetStorageShape();
    const auto &cacheTokensShape = op.cacheTokens.shape->GetStorageShape();
    const auto &seqShape = op.actualSeqLengths.shape->GetStorageShape();
    const auto &blockShape = op.blockTable.shape->GetStorageShape();
    const auto &indexOutShape = op.topkIndexOut.shape->GetStorageShape();
    const auto &slotsOutShape = op.topkSlotsOut.shape->GetStorageShape();
    const auto *topkMissCountOutShape =
        mtp_ ? &op.topkMissCountOut.shape->GetStorageShape() : nullptr;
    const auto &missCountOutShape = op.missCountOut.shape->GetStorageShape();
    const auto &cacheSlotsOutShape = op.cacheSlotsOut.shape->GetStorageShape();
    const auto *missSrcShape = mtp_ ? &op.missSrcOut.shape->GetStorageShape() : nullptr;
    const auto *missSlotsShape = mtp_ ? &op.missSlotsOut.shape->GetStorageShape() : nullptr;
    const gert::StorageShape *queryScaleStorage = context_->GetInputShape(QUERY_DEQUANT_SCALE_INDEX);
    const gert::StorageShape *keyScaleStorage = context_->GetInputShape(KEY_DEQUANT_SCALE_INDEX);
    OPS_ERR_IF(queryScaleStorage == nullptr || keyScaleStorage == nullptr,
               OPS_LOG_E(tilingInfo.opName, "dequant scale shapes are nullptr."),
               return ge::GRAPH_FAILED);
    const auto &queryScaleShape = queryScaleStorage->GetStorageShape();
    const auto &keyScaleShape = keyScaleStorage->GetStorageShape();

    OPS_ERR_IF(qShape.GetDimNum() != DIM_NUM_THREE,
               OPS_LOG_E(tilingInfo.opName, "query must be TND [B, N1, 128], where N1 is 32 or 64."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(kShape.GetDimNum() != DIM_NUM_FOUR,
               OPS_LOG_E(tilingInfo.opName, "key must be PA_BSND [num_blocks, block_size, 1, 128]."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(wShape.GetDimNum() != DIM_NUM_TWO,
               OPS_LOG_E(tilingInfo.opName, "weights must be [B, N1], where N1 is 32 or 64."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(queryScaleShape.GetDimNum() != DIM_NUM_TWO ||
                   queryScaleShape.GetDim(0) != qShape.GetDim(0) ||
                   queryScaleShape.GetDim(1) != qShape.GetDim(1),
               OPS_LOG_E(tilingInfo.opName, "query_dequant_scale must be [T,N]."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(keyScaleShape.GetDimNum() != DIM_NUM_THREE ||
                   keyScaleShape.GetDim(0) != kShape.GetDim(0) ||
                   keyScaleShape.GetDim(1) != kShape.GetDim(1) ||
                   keyScaleShape.GetDim(2) != kShape.GetDim(2),
               OPS_LOG_E(tilingInfo.opName, "index_key_dequant_scale must be [blocks,128,1]."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(cacheShape.GetDimNum() != DIM_NUM_TWO,
               OPS_LOG_E(tilingInfo.opName, "cache_slots must be [pool_size, source_capacity]."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(reqPoolShape.GetDimNum() != DIM_NUM_ONE,
               OPS_LOG_E(tilingInfo.opName, "req_pool_entries must be rank 1."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(cacheTokensShape.GetDimNum() != DIM_NUM_ONE,
               OPS_LOG_E(tilingInfo.opName, "cache_tokens must be rank 1."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(seqShape.GetDimNum() != DIM_NUM_ONE,
               OPS_LOG_E(tilingInfo.opName, "offload_seq_lengths_key must be rank 1."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(blockShape.GetDimNum() != DIM_NUM_TWO,
               OPS_LOG_E(tilingInfo.opName, "block_table must be rank 2."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(indexOutShape.GetDimNum() != DIM_NUM_THREE || slotsOutShape.GetDimNum() != DIM_NUM_THREE,
               OPS_LOG_E(tilingInfo.opName, "topk_index/topk_slots must be [T, 1, 2048]."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(missCountOutShape.GetDimNum() != DIM_NUM_ONE,
               OPS_LOG_E(tilingInfo.opName, "miss_count must be [B]."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(cacheSlotsOutShape.GetDimNum() != DIM_NUM_TWO,
               OPS_LOG_E(tilingInfo.opName, "cache_slots output must be rank 2."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(mtp_ && topkMissCountOutShape->GetDimNum() != DIM_NUM_ONE,
               OPS_LOG_E(tilingInfo.opName, "topk_miss_counts must be [T]."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(mtp_ && (missSrcShape->GetDimNum() != DIM_NUM_TWO ||
                        missSlotsShape->GetDimNum() != DIM_NUM_TWO),
               OPS_LOG_E(tilingInfo.opName, "MTP miss outputs must be [B, 32768]."),
               return ge::GRAPH_FAILED);

    tilingInfo.tSize = static_cast<uint32_t>(qShape.GetDim(0));
    tilingInfo.bSize = static_cast<uint32_t>(blockShape.GetDim(0));
    tilingInfo.n1Size = static_cast<uint32_t>(qShape.GetDim(1));
    tilingInfo.n2Size = static_cast<uint32_t>(kShape.GetDim(DIM_IDX_TWO));
    tilingInfo.blockSize = static_cast<uint32_t>(kShape.GetDim(DIM_IDX_ONE));
    tilingInfo.maxBlockNumPerBatch = static_cast<uint32_t>(blockShape.GetDim(DIM_IDX_ONE));
    tilingInfo.s2Size = 0;
    tilingInfo.poolSize = static_cast<uint32_t>(cacheShape.GetDim(0));
    tilingInfo.cacheSlotsSize = static_cast<uint32_t>(cacheShape.GetDim(1));

    OPS_ERR_IF(tilingInfo.bSize == 0 || tilingInfo.tSize < tilingInfo.bSize ||
                   tilingInfo.tSize > tilingInfo.bSize * 14U,
               OPS_LOG_E(tilingInfo.opName, "requires B <= T <= 14B."),
               return ge::GRAPH_FAILED);
    uint32_t metadataBatch = tilingInfo.bSize;
    for (uint32_t input = ACTUAL_SEQ_Q_INDEX; input <= REQ_POOL_ENTRIES_INDEX; ++input) {
        const gert::StorageShape *storage = context_->GetInputShape(input);
        OPS_ERR_IF(storage == nullptr || storage->GetStorageShape().GetDimNum() != DIM_NUM_ONE ||
                       storage->GetStorageShape().GetDim(0) != metadataBatch,
                   OPS_LOG_E(tilingInfo.opName, "request metadata tensors must be [B]."),
                   return ge::GRAPH_FAILED);
    }
    OPS_ERR_IF(reqPoolShape.GetShapeSize() != metadataBatch ||
                    cacheTokensShape.GetShapeSize() != metadataBatch ||
                    seqShape.GetShapeSize() != metadataBatch || blockShape.GetDim(0) != metadataBatch,
               OPS_LOG_E(tilingInfo.opName,
                         "query batch, req_pool_entries, cache_tokens, sequence lengths, and block_table batch must match."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(tilingInfo.poolSize == 0 || tilingInfo.cacheSlotsSize == 0,
               OPS_LOG_E(tilingInfo.opName, "cache_slots dimensions must be positive."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(kShape.GetDim(0) == 0, OPS_LOG_E(tilingInfo.opName, "key num_blocks must be > 0."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(tilingInfo.maxBlockNumPerBatch == 0,
               OPS_LOG_E(tilingInfo.opName, "block_table must contain at least one block per request."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(tilingInfo.maxBlockNumPerBatch > (1U << 14),
               OPS_LOG_E(tilingInfo.opName, "block_table capacity must be <= 16384 blocks for the 21-bit source format."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(tilingInfo.blockSize != 128,
               OPS_LOG_E(tilingInfo.opName, "key block_size must be 128."),
               return ge::GRAPH_FAILED);
    const uint64_t sourceCapacity =
        static_cast<uint64_t>(tilingInfo.blockSize) *
        tilingInfo.maxBlockNumPerBatch;
    OPS_ERR_IF(sourceCapacity > (1U << 21),
               OPS_LOG_E(tilingInfo.opName,
                          "cache_slots capacity must be <= 2^21 tokens."),
               return ge::GRAPH_FAILED);
    tilingInfo.s2Size = static_cast<uint32_t>(sourceCapacity);
    OPS_ERR_IF(tilingInfo.s2Size != tilingInfo.cacheSlotsSize,
               OPS_LOG_E(tilingInfo.opName,
                          "cache_slots capacity must equal table capacity."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(tilingInfo.n2Size != DECODE_N2,
               OPS_LOG_E(tilingInfo.opName, "key N2 must be 1."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(tilingInfo.n1Size != 32 && tilingInfo.n1Size != 64,
               OPS_LOG_E(tilingInfo.opName, "decode query N1 must be 32 or 64."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(qShape.GetDim(DIM_IDX_TWO) != DECODE_HEAD_DIM || kShape.GetDim(DIM_IDX_THREE) != DECODE_HEAD_DIM,
               OPS_LOG_E(tilingInfo.opName, "head_dim must be 128."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(wShape.GetDim(0) != tilingInfo.tSize || wShape.GetDim(1) != tilingInfo.n1Size,
               OPS_LOG_E(tilingInfo.opName, "weights must match query [T, N1]."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(indexOutShape.GetDim(0) != tilingInfo.tSize || indexOutShape.GetDim(1) != DECODE_N2 ||
                    indexOutShape.GetDim(2) != DECODE_OUTPUT_CAPACITY ||
                    slotsOutShape.GetDim(0) != tilingInfo.tSize || slotsOutShape.GetDim(1) != DECODE_N2 ||
                    slotsOutShape.GetDim(2) != DECODE_OUTPUT_CAPACITY,
               OPS_LOG_E(tilingInfo.opName, "topk_index/topk_slots must have shape [T, 1, 2048]."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(missCountOutShape.GetDim(0) != metadataBatch,
               OPS_LOG_E(tilingInfo.opName, "miss_count must have shape [B]."),
                return ge::GRAPH_FAILED);
    OPS_ERR_IF(mtp_ && topkMissCountOutShape->GetDim(0) != tilingInfo.tSize,
               OPS_LOG_E(tilingInfo.opName,
                         "topk_miss_counts must have shape [T]."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(mtp_ && (missSrcShape->GetDim(0) != metadataBatch || missSrcShape->GetDim(1) != 32768 ||
                        missSlotsShape->GetDim(0) != metadataBatch || missSlotsShape->GetDim(1) != 32768),
               OPS_LOG_E(tilingInfo.opName, "MTP miss outputs must have shape [B, 32768]."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(cacheSlotsOutShape.GetDim(0) != cacheShape.GetDim(0) ||
                   cacheSlotsOutShape.GetDim(1) != cacheShape.GetDim(1),
               OPS_LOG_E(tilingInfo.opName,
                         "cache_slots output must match the request-state pool shape."),
               return ge::GRAPH_FAILED);

    tilingInfo.inputQType = op.query.desc->GetDataType();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedLiManageMtpTiling::ParseAndCheck(FusedLiManageMtpTilingInfo &tilingInfo)
{
    if (GetNpuInfo(tilingInfo) != ge::GRAPH_SUCCESS || GetTensorInfo(tilingInfo) != ge::GRAPH_SUCCESS ||
        CheckDtype(tilingInfo) != ge::GRAPH_SUCCESS || CheckShape(tilingInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedLiManageMtpTiling::DoTiling(FusedLiManageMtpTilingInfo *tilingInfo)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(tilingInfo->platformInfo);
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    constexpr uint32_t SCHEDULE_BALANCED = 1;
    constexpr uint32_t SCHEDULE_AUTO = 2;
    uint32_t selectedScheduleMode =
        (tilingInfo->bSize % aicNum) == 0U ? SCHEDULE_AUTO : SCHEDULE_BALANCED;
    tilingInfo->usedCoreNum =
        selectedScheduleMode == SCHEDULE_BALANCED ? aicNum : std::min(tilingInfo->bSize, aicNum);
    uint32_t requestedAivNum = std::min(aivNum, tilingInfo->usedCoreNum * 2U);
    uint32_t blockDim = ascendcPlatform.CalcTschBlockDim(requestedAivNum, aicNum, aivNum);
    context_->SetBlockDim(blockDim);

    constexpr uint32_t MM1_RES_ELEM_SIZE = 4;
    constexpr uint32_t DOUBLE_BUFFER = 2;
    constexpr uint32_t M_BASE_SIZE = 64;
    constexpr uint32_t S2_BASE_SIZE = 512;
    uint64_t workspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    if (mtp_) {
        const uint32_t MTP_M_BASE_SIZE = 8U * tilingInfo->n1Size;
        workspaceSize += MTP_M_BASE_SIZE * S2_BASE_SIZE * MM1_RES_ELEM_SIZE * DOUBLE_BUFFER * blockDim;
    } else {
        workspaceSize += M_BASE_SIZE * S2_BASE_SIZE * MM1_RES_ELEM_SIZE * DOUBLE_BUFFER * blockDim;
    }
    uint64_t scoreStride = ((static_cast<uint64_t>(tilingInfo->s2Size) + S2_BASE_SIZE - 1) / S2_BASE_SIZE) *
                           S2_BASE_SIZE;
    if (!mtp_) {
        workspaceSize += static_cast<uint64_t>(tilingInfo->bSize) * scoreStride * sizeof(float);
    }
    if (mtp_) {
        constexpr uint32_t S1_BASE_SIZE = 8;
        constexpr uint32_t LD_HEAD_TAIL = 2;
        constexpr uint32_t VALUE_AND_INDEX = 2;
        constexpr uint32_t LD_PARAM_NUM = 16;
        workspaceSize += static_cast<uint64_t>(blockDim) * LD_HEAD_TAIL * S1_BASE_SIZE *
                         VALUE_AND_INDEX * DECODE_SPARSE_COUNT * sizeof(float);
        workspaceSize += static_cast<uint64_t>(blockDim) * LD_HEAD_TAIL * S1_BASE_SIZE *
                         LD_PARAM_NUM * sizeof(int64_t);
        constexpr uint32_t MTP_PAIR_CAPACITY = 4U * DECODE_SPARSE_COUNT;
        constexpr uint32_t MTP_THRESHOLD_STRIDE = 8U;
        constexpr uint32_t MTP_ROUTE_COUNT_STRIDE = 8U;
        const uint64_t metadataBatch = tilingInfo->bSize;
        // Keep this order synchronized with MtpWorkspace::{Pair0Offset,
        // Pair1Offset, ScoreOffset, ThresholdOffset, RouteCountOffset} in the kernel.
        workspaceSize += metadataBatch * MTP_PAIR_CAPACITY * sizeof(float) * 2U;
        workspaceSize += static_cast<uint64_t>(tilingInfo->tSize) * scoreStride * sizeof(float);
        workspaceSize += static_cast<uint64_t>(tilingInfo->tSize) * MTP_THRESHOLD_STRIDE * sizeof(float);
        workspaceSize += static_cast<uint64_t>(tilingInfo->tSize) * MTP_ROUTE_COUNT_STRIDE * sizeof(int32_t);
    }
    constexpr uint32_t PARTIAL_SLOTS_PER_CORE = 2;
    constexpr uint32_t PARTIAL_META_INTS_PER_CORE = 8;
    constexpr uint32_t TOPK_PAIR_ELEMS = DECODE_SPARSE_COUNT * 2;
    if (!mtp_) {
        workspaceSize +=
            static_cast<uint64_t>(blockDim) * PARTIAL_SLOTS_PER_CORE * TOPK_PAIR_ELEMS * sizeof(float);
        workspaceSize += static_cast<uint64_t>(blockDim) * PARTIAL_META_INTS_PER_CORE * sizeof(int32_t);
    }
    context_->GetWorkspaceSizes(1)[0] = workspaceSize;

    tilingData_.set_bSize(tilingInfo->bSize);
    tilingData_.set_tSize(tilingInfo->tSize);
    tilingData_.set_s2Size(tilingInfo->s2Size);
    tilingData_.set_blockSize(tilingInfo->blockSize);
    tilingData_.set_maxBlockNumPerBatch(tilingInfo->maxBlockNumPerBatch);
    tilingData_.set_poolSize(tilingInfo->poolSize);
    tilingData_.set_n1Size(tilingInfo->n1Size);
    tilingData_.set_cacheSlotsSize(tilingInfo->cacheSlotsSize);
    tilingData_.set_usedCoreNum(blockDim);
    tilingData_.set_scheduleMode(selectedScheduleMode);
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());

    uint32_t tilingKey = GET_TPL_TILING_KEY(static_cast<uint32_t>(tilingInfo->inputQType));
    context_->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedLiManageMtpTiling::TilingC8Fp8(FusedLiManageMtpTilingInfo &tilingInfo)
{
    if (GetNpuInfo(tilingInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    // This shared host tiling is also compiled against CANN 8.5.1, which ships
    // no ASCEND950 SocVersion enum, so the A5 is identified by exclusion of the
    // two A3 SoCs registered in the def (ascend910b / ascend910_93).  An fp8
    // graph can only be served by the Ascend950 (arch35) C8 engine, so reject
    // fp8 on A3 outright instead of falling into the A3 checks.
    const platform_ascendc::SocVersion soc = tilingInfo.socVersion;
    if (soc == platform_ascendc::SocVersion::ASCEND910B ||
        soc == platform_ascendc::SocVersion::ASCEND910_93) {
        OPS_LOG_E(tilingInfo.opName,
                  "fp8 fused_li_manage_mtp requires Ascend950; A3 SoC rejected.");
        return ge::GRAPH_FAILED;
    }

    // ----- A3-ordered input descriptors / shapes (see fused_li_manage_mtp_tiling.h) -----
    if (context_->GetInputDesc(QUERY_INDEX) == nullptr ||
        context_->GetInputDesc(KEY_INDEX) == nullptr ||
        context_->GetInputDesc(WEIGHTS_INDEX) == nullptr ||
        context_->GetInputDesc(QUERY_DEQUANT_SCALE_INDEX) == nullptr ||
        context_->GetInputDesc(KEY_DEQUANT_SCALE_INDEX) == nullptr ||
        context_->GetInputDesc(BLOCK_TABLE_INDEX) == nullptr ||
        context_->GetInputDesc(ACTUAL_SEQ_Q_INDEX) == nullptr ||
        context_->GetInputDesc(OFFLOAD_SEQ_K_INDEX) == nullptr ||
        context_->GetInputDesc(CACHE_TOKENS_INDEX) == nullptr ||
        context_->GetInputDesc(REQ_POOL_ENTRIES_INDEX) == nullptr ||
        context_->GetInputDesc(CACHE_SLOTS_INDEX) == nullptr) {
        OPS_LOG_E(tilingInfo.opName, "c8 fp8 path: an input desc is nullptr.");
        return ge::GRAPH_FAILED;
    }
    const auto inputDtype = [this](uint32_t index) {
        return context_->GetInputDesc(index)->GetDataType();
    };
    const auto inputShape = [this](uint32_t index) -> const gert::Shape * {
        const gert::StorageShape *storage = context_->GetInputShape(index);
        return storage == nullptr ? nullptr : &storage->GetStorageShape();
    };
    // query [T,N,128] uint8 carrying fp8 E4M3 bytes, key [blocks,128,1,128]
    // likewise, weights [T,N] bf16, q/k dequant scales fp32; all
    // metadata/table/pool inputs int32.
    if (inputDtype(QUERY_INDEX) != ge::DT_UINT8 ||
        inputDtype(KEY_INDEX) != ge::DT_UINT8 ||
        inputDtype(WEIGHTS_INDEX) != ge::DT_BF16 ||
        inputDtype(QUERY_DEQUANT_SCALE_INDEX) != ge::DT_FLOAT ||
        inputDtype(KEY_DEQUANT_SCALE_INDEX) != ge::DT_FLOAT) {
        OPS_LOG_E(tilingInfo.opName,
                  "c8 fp8 path requires query/key uint8(fp8), weights bf16, scales fp32.");
        return ge::GRAPH_FAILED;
    }
    for (uint32_t index :
         {BLOCK_TABLE_INDEX, ACTUAL_SEQ_Q_INDEX, ACTUAL_SEQ_K_INDEX,
          OFFLOAD_SEQ_K_INDEX, CACHE_TOKENS_INDEX, REQUEST_STATE_INDEX,
          REQ_POOL_ENTRIES_INDEX, CACHE_SLOTS_INDEX}) {
        if (context_->GetInputDesc(index) == nullptr ||
            context_->GetInputDesc(index)->GetDataType() != ge::DT_INT32) {
            OPS_LOG_E(tilingInfo.opName,
                      "c8 fp8 path requires int32 metadata/table/pool inputs.");
            return ge::GRAPH_FAILED;
        }
    }

    const auto query = inputShape(QUERY_INDEX);
    const auto key = inputShape(KEY_INDEX);
    const auto weights = inputShape(WEIGHTS_INDEX);
    const auto queryScale = inputShape(QUERY_DEQUANT_SCALE_INDEX);
    const auto keyScale = inputShape(KEY_DEQUANT_SCALE_INDEX);
    const auto actualQ = inputShape(ACTUAL_SEQ_Q_INDEX);
    const auto entries = inputShape(REQ_POOL_ENTRIES_INDEX);
    const auto pool = inputShape(CACHE_SLOTS_INDEX);
    const auto blockTable = inputShape(BLOCK_TABLE_INDEX);
    for (uint32_t index = 0U; index <= CACHE_SLOTS_INDEX; ++index) {
        if (inputShape(index) == nullptr) {
            OPS_LOG_E(tilingInfo.opName, "c8 fp8 path: an input shape is nullptr.");
            return ge::GRAPH_FAILED;
        }
    }
    // Reference c8 geometry (MTP-4 steady state).  A3 order is used everywhere.
    const int64_t packedQueries = query->GetDim(0);
    const int64_t heads = query->GetDim(1);
    const int64_t batch = actualQ->GetDim(0);
    const int64_t sourceCapacity = pool->GetDim(1);
    constexpr int64_t HEAD_DIM = 128;
    constexpr int64_t BLOCK_SIZE = 128;
    constexpr int64_t MTP4_QUERIES_PER_REQUEST = 4;
    constexpr int64_t MAX_BATCH_C8_FORK = 16;  // bs<=16 LD fork only in this tree
    constexpr int64_t MAX_SOURCE_CAPACITY = 1 << 18;
    constexpr int64_t TOPK_CAPACITY = 2048;
    constexpr int64_t MISS_CAPACITY = 32768;
    const bool rankOk =
        query->GetDimNum() == 3 && key->GetDimNum() == 4 &&
        weights->GetDimNum() == 2 && queryScale->GetDimNum() == 2 &&
        keyScale->GetDimNum() == 3 && actualQ->GetDimNum() == 1 &&
        entries->GetDimNum() == 1 && pool->GetDimNum() == 2 &&
        blockTable->GetDimNum() == 2;
    const bool dimOk =
        batch > 0 && batch <= MAX_BATCH_C8_FORK && packedQueries > 0 &&
        packedQueries == batch * MTP4_QUERIES_PER_REQUEST &&
        (heads == 32 || heads == 64) && query->GetDim(2) == HEAD_DIM &&
        key->GetDim(0) > 0 && key->GetDim(1) == BLOCK_SIZE &&
        key->GetDim(2) == 1 && key->GetDim(3) == HEAD_DIM &&
        weights->GetDim(0) == packedQueries && weights->GetDim(1) == heads &&
        queryScale->GetDim(0) == packedQueries && queryScale->GetDim(1) == heads &&
        keyScale->GetDim(0) == key->GetDim(0) && keyScale->GetDim(1) == BLOCK_SIZE &&
        keyScale->GetDim(2) == 1 && entries->GetDim(0) == batch &&
        blockTable->GetDim(0) == batch && blockTable->GetDim(1) > 0 &&
        pool->GetDim(0) > 0 && sourceCapacity > 0 &&
        sourceCapacity <= MAX_SOURCE_CAPACITY &&
        blockTable->GetDim(1) * BLOCK_SIZE == sourceCapacity;
    if (!rankOk || !dimOk) {
        OPS_LOG_E(tilingInfo.opName,
                  "c8 fp8 path shape check failed (expect MTP-4 fp8 geometry, "
                  "query [T,N,128], key [blocks,128,1,128], pool [pool, sourceCap], "
                  "block_table [B, maxBlocks] with maxBlocks*128==sourceCap<=2^18, "
                  "batch<=16).");
        return ge::GRAPH_FAILED;
    }
    // A3-only [B] metadata tensors must be rank-1 too (values are device-side and
    // enforced by the caller; M1 steady state keeps request_state == -1).
    for (uint32_t index :
         {ACTUAL_SEQ_Q_INDEX, ACTUAL_SEQ_K_INDEX, OFFLOAD_SEQ_K_INDEX,
          CACHE_TOKENS_INDEX, REQUEST_STATE_INDEX, REQ_POOL_ENTRIES_INDEX}) {
        const auto s = inputShape(index);
        if (s->GetDimNum() != 1 || s->GetDim(0) != batch) {
            OPS_LOG_E(tilingInfo.opName,
                      "c8 fp8 path requires per-request [B] metadata tensors.");
            return ge::GRAPH_FAILED;
        }
    }
    // Positional output validation (1:1 with the A3 output tensors).
    if (context_->GetOutputShape(0) == nullptr ||
        context_->GetOutputShape(1) == nullptr ||
        context_->GetOutputShape(2) == nullptr ||
        context_->GetOutputShape(3) == nullptr ||
        context_->GetOutputShape(4) == nullptr ||
        context_->GetOutputShape(5) == nullptr ||
        context_->GetOutputShape(6) == nullptr) {
        OPS_LOG_E(tilingInfo.opName, "c8 fp8 path: an output shape is nullptr.");
        return ge::GRAPH_FAILED;
    }
    const auto topk = context_->GetOutputShape(0)->GetStorageShape();
    const auto topkMiss = context_->GetOutputShape(2)->GetStorageShape();
    const auto missSrc = context_->GetOutputShape(3)->GetStorageShape();
    const auto missDst = context_->GetOutputShape(4)->GetStorageShape();
    const auto missCount = context_->GetOutputShape(5)->GetStorageShape();
    const auto poolOut = context_->GetOutputShape(6)->GetStorageShape();
    if (topk.GetDimNum() != 3 || topk.GetDim(0) != packedQueries ||
        topk.GetDim(1) != 1 || topk.GetDim(2) != TOPK_CAPACITY ||
        topkMiss.GetDimNum() != 1 || topkMiss.GetDim(0) != packedQueries ||
        missSrc.GetDimNum() != 2 || missSrc.GetDim(0) != batch ||
        missSrc.GetDim(1) != MISS_CAPACITY ||
        missDst.GetDimNum() != 2 || missDst.GetDim(0) != batch ||
        missDst.GetDim(1) != MISS_CAPACITY ||
        missCount.GetDimNum() != 1 || missCount.GetDim(0) != batch ||
        poolOut.GetDimNum() != 2 || poolOut.GetDim(0) != pool->GetDim(0) ||
        poolOut.GetDim(1) != sourceCapacity) {
        OPS_LOG_E(tilingInfo.opName, "c8 fp8 path output shape check failed.");
        return ge::GRAPH_FAILED;
    }

    platform_ascendc::PlatformAscendC platform(tilingInfo.platformInfo);
    const uint32_t aicCount = platform.GetCoreNumAic();
    const uint32_t aivCount = platform.GetCoreNumAiv();
    if (aicCount == 0 || aivCount < 2) {
        OPS_LOG_E(tilingInfo.opName, "c8 fp8 path: platform core count is invalid.");
        return ge::GRAPH_FAILED;
    }

    // C8 LD engine workspace: score region stride == 4 queries/request *
    // sourceCapacity fp16 scores per query (see reference A5 host tiling).
    const uint64_t scoreStride64 =
        static_cast<uint64_t>(MTP4_QUERIES_PER_REQUEST) * sourceCapacity *
        sizeof(uint16_t);
    if (scoreStride64 > std::numeric_limits<uint32_t>::max()) {
        OPS_LOG_E(tilingInfo.opName, "c8 fp8 path: score workspace stride overflows.");
        return ge::GRAPH_FAILED;
    }
    constexpr uint32_t C8_MIX_AIC_NUM = 24;  // LD path always fills 24 AIC

    // Write the arch35 tiling image straight into the raw tiling buffer, exactly
    // as the A3 path does (SaveToBuffer over GetRawTilingData).  Going through
    // TilingContext::GetTilingData<T>() does not resolve here: the op registers
    // FusedLiManageMtpTilingData as its default class, and GetTilingData only
    // hands back a type that resolves for this op -- it returned null for both
    // the plain A5FusedLiManageMtpC8TilingData struct and the key-55 registered
    // A5LiManageMtpC8Tiling mirror.
    auto *rawTiling = context_->GetRawTilingData();
    constexpr size_t kFp8TilingBytes = sizeof(A5FusedLiManageMtpC8TilingData);
    if (rawTiling == nullptr || context_->GetWorkspaceSizes(1) == nullptr) {
        OPS_LOG_E(tilingInfo.opName, "c8 fp8 path: tiling/workspace buffer is nullptr.");
        return ge::GRAPH_FAILED;
    }
    if (rawTiling->GetCapacity() < kFp8TilingBytes) {
        OPS_LOG_E(tilingInfo.opName,
                  "c8 fp8 path: tiling buffer too small (capacity=%lu need=%lu).",
                  static_cast<unsigned long>(rawTiling->GetCapacity()),
                  static_cast<unsigned long>(kFp8TilingBytes));
        return ge::GRAPH_FAILED;
    }
    auto *tiling =
        reinterpret_cast<A5FusedLiManageMtpC8TilingData *>(rawTiling->GetData());
    tiling->usedCoreNum = C8_MIX_AIC_NUM;
    tiling->batchSize = static_cast<uint32_t>(batch);
    tiling->packedQueryCount = static_cast<uint32_t>(packedQueries);
    tiling->poolSize = static_cast<uint32_t>(pool->GetDim(0));
    tiling->sourceCapacity = static_cast<uint32_t>(sourceCapacity);
    tiling->indexHeads = static_cast<uint32_t>(heads);
    tiling->maxBlockNumPerBatch = static_cast<uint32_t>(blockTable->GetDim(1));
    tiling->maxCandidateLen = static_cast<uint32_t>(sourceCapacity);
    tiling->keyStride = BLOCK_SIZE * HEAD_DIM;
    tiling->scaleStride = BLOCK_SIZE;
    tiling->scoreWorkspaceStride = static_cast<uint32_t>(scoreStride64);
    // GL (batchSize>16) pipeline fields are unused on this bs<=16 LD fork.
    tiling->queryTileSize = 0U;
    tiling->splitEnable = 0U;
    tiling->scoringCoreNum = 0U;
    rawTiling->SetDataSize(kFp8TilingBytes);
    context_->GetWorkspaceSizes(1)[0] =
        platform.GetLibApiWorkSpaceSize() +
        a5_fused_li_manage_mtp_c8_workspace::TotalBytes(
            scoreStride64, static_cast<uint64_t>(batch),
            static_cast<uint64_t>(heads));
    context_->SetBlockDim(platform.CalcTschBlockDim(
        C8_MIX_AIC_NUM * 2U, C8_MIX_AIC_NUM, C8_MIX_AIC_NUM * 2U));
    context_->SetScheduleMode(1);
    // Use the declared template constant directly.  Deriving the key from the
    // ge dtype enum via GET_TPL_TILING_KEY only happens to work for A3 (FP16=1,
    // BF16=27 match the declared values); ge::DT_FLOAT8_E4M3FN resolves to 36
    // under this CANN, which names no compiled instantiation and the launcher
    // fails with "not find tilingKey[36]" before the kernel is ever entered.
    context_->SetTilingKey(LI_MTP_TPL_FP8);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepareForNanovllmFusedLiManageMtp(gert::TilingParseContext * /* context */)
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingForNanovllmFusedLiManageMtp(gert::TilingContext *context)
{
    OPS_ERR_IF(context == nullptr,
               OPS_REPORT_VECTOR_INNER_ERR("NanovllmFusedLiManageMtp", "Tiling context is null."),
               return ge::GRAPH_FAILED);
    FusedLiManageMtpTilingInfo liInfo;
    FusedLiManageMtpTiling liTiling(context, true);

    // Ascend950 (arch35) fp8 early-exit.  The fp8 graph arrives as a uint8
    // carrier (see the def comment), and CheckDtype/CheckShape only know
    // bf16/fp16, so the dedicated C8 path must run before ParseAndCheck.
    // bf16/fp16 graphs (both A3 and, for now, Ascend950) never see uint8 here
    // and keep the flow below unchanged.
    const gert::CompileTimeTensorDesc *queryDesc = context->GetInputDesc(QUERY_INDEX);
    if (queryDesc != nullptr && queryDesc->GetDataType() == ge::DT_UINT8) {
        return liTiling.TilingC8Fp8(liInfo);
    }

    if (liTiling.ParseAndCheck(liInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return liTiling.DoTiling(&liInfo);
}

IMPL_OP_OPTILING(NanovllmFusedLiManageMtp)
    .Tiling(TilingForNanovllmFusedLiManageMtp)
    .TilingParse<FusedLiManageMtpCompileInfo>(TilingPrepareForNanovllmFusedLiManageMtp);

} // namespace optiling
