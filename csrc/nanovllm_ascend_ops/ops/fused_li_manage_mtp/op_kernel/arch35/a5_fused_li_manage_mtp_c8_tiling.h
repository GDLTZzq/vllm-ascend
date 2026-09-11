#ifndef A5_FUSED_LI_MANAGE_MTP_C8_TILING_H
#define A5_FUSED_LI_MANAGE_MTP_C8_TILING_H

#include <cstdint>

// Only static geometry is tiled. Query counts, request-pool entries, cache
// budgets and candidate lengths remain device tensors for graph replay.
struct A5FusedLiManageMtpC8TilingData {
    uint32_t usedCoreNum;
    uint32_t batchSize;
    uint32_t packedQueryCount;
    uint32_t poolSize;
    uint32_t sourceCapacity;
    uint32_t indexHeads;
    uint32_t maxBlockNumPerBatch;
    uint32_t maxCandidateLen;
    uint32_t keyStride;
    uint32_t scaleStride;
    uint32_t scoreWorkspaceStride;
    // ---- 组长的 arch35_payload_c8 管线（batchSize > 16）专用 ----
    // 两条 stage1 管线共用同一 tilingData：score 区/routePairs/thresholds 布局
    // 逐字段一致，host 按 batchSize>16 与否填入对应路径字段。本字段仅组长管线读。
    uint32_t queryTileSize;
    // 1 = Stage-1 打分跨核拆分（小 batch）；kernel 只在 taskCount*2 <=
    // scoringCoreNum 且全 active 时启用。0 = 原整行串行路径。
    uint32_t splitEnable;
    // 实际发射的 MIX 组数（split 时 = mixGroupCount=24，否则 = usedCoreNum）。
    // 与 usedCoreNum(=owner 数，stage2/union 语义) 分开。
    uint32_t scoringCoreNum;
};

#endif
