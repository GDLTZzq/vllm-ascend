#ifndef A5_FUSED_LI_MANAGE_MTP_C8_WORKSPACE_H
#define A5_FUSED_LI_MANAGE_MTP_C8_WORKSPACE_H

#include <cstdint>

namespace a5_fused_li_manage_mtp_c8_workspace {

constexpr uint64_t ROUTES = 4U;
constexpr uint64_t TOPK = 2048U;
constexpr uint64_t UNION_CAPACITY = ROUTES * TOPK;
constexpr uint64_t THRESHOLD_STRIDE = 16U;  // uint16 elements, one 32-byte row
constexpr uint64_t ALIGN_BYTES = 32U;

constexpr uint64_t AlignBytes(uint64_t value)
{
    return (value + ALIGN_BYTES - 1U) / ALIGN_BYTES * ALIGN_BYTES;
}

constexpr uint64_t ScoreBytes(uint64_t scoreStrideBytes, uint64_t batch)
{
    return scoreStrideBytes * batch;
}

constexpr uint64_t RoutePairOffset(uint64_t scoreStrideBytes, uint64_t batch)
{
    return ScoreBytes(scoreStrideBytes, batch);
}

constexpr uint64_t RoutePairBytes(uint64_t batch)
{
    return batch * UNION_CAPACITY * 2U * sizeof(int32_t);
}

constexpr uint64_t RouteThresholdOffset(uint64_t scoreStrideBytes,
                                        uint64_t batch)
{
    return RoutePairOffset(scoreStrideBytes, batch) + RoutePairBytes(batch);
}

constexpr uint64_t RouteThresholdBytes(uint64_t batch)
{
    return batch * ROUTES * THRESHOLD_STRIDE * sizeof(uint16_t);
}

// QLI LD 归并工作区：host 分配口径与 kernel Init 一致。
// ldScore/ldIndex 每核至多 2 个跨核行（slot=2 固定），s1BaseSizeWs=ceil(256/heads)，
// aicNum=GetBlockNum()=24（MIX_AIC_1_2）。
constexpr uint64_t M_BASE_SIZE = 256U;
constexpr uint64_t LD_AIC_NUM = 24U;
constexpr uint64_t LD_TOPK_ALIGN16 = 2048U;

constexpr uint64_t S1BaseSizeWs(uint64_t indexHeads)
{
    return (M_BASE_SIZE + indexHeads - 1U) / indexHeads;
}

constexpr uint64_t LdScoreBytes(uint64_t indexHeads)
{
    return LD_AIC_NUM * S1BaseSizeWs(indexHeads) * LD_TOPK_ALIGN16 * 2U * sizeof(uint16_t);
}

constexpr uint64_t LdIndexBytes(uint64_t indexHeads)
{
    return LD_AIC_NUM * S1BaseSizeWs(indexHeads) * LD_TOPK_ALIGN16 * 2U * sizeof(int32_t);
}

constexpr uint64_t LdOffset(uint64_t scoreStrideBytes, uint64_t batch)
{
    return RouteThresholdOffset(scoreStrideBytes, batch) + RouteThresholdBytes(batch);
}

constexpr uint64_t TotalBytes(uint64_t scoreStrideBytes, uint64_t batch,
                              uint64_t indexHeads)
{
    return AlignBytes(LdOffset(scoreStrideBytes, batch) +
                      LdScoreBytes(indexHeads) + LdIndexBytes(indexHeads));
}

} // namespace a5_fused_li_manage_mtp_c8_workspace

#endif
