/**
 * Stage-1 A5 C8 MTP3 LightningIndexer（QLI LD 引擎适配层）。
 *
 * Stage-1 工作单元是 (request, gS1 tile)，由 quant_lightning_indexer_c8 的
 * LD 引擎（块池切分 + S2 跨核并行 + LD 归并）完成打分/topk，再走 classify+publish
 * 把每 route 的 top-2048 落成 stage2（OrderedMissUnion）要读的
 * routePairs / thresholds / missCounts / topkSourceIds / topkSlots。
 *
 * MTP-4 绑定：kHeadNum=1、qSeqSize=4、bSize=requests、gSize=heads，
 * 输出行 = batch*4+route，score 区为 request-major（batch*4*sourceCapacity 行主序）。
 */

#ifndef A5_FUSED_LI_MANAGE_MTP_C8_QLI_H
#define A5_FUSED_LI_MANAGE_MTP_C8_QLI_H

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "a5_fused_li_manage_mtp_c8_tiling.h"
#include "a5_fused_li_manage_mtp_c8_workspace.h"

// QLIPreload<QLIT>::Init 的 tiling 形参类型是模板定义点的非依赖名，须在本 TU
// include arch35_ld/quant_lightning_indexer_kernel.h 之前可见。
// fused op 的 tiling 即 A5FusedLiManageMtpC8TilingData；字段映射见 InitTilingData。
using QLITilingData = A5FusedLiManageMtpC8TilingData;

#include "arch35_ld/quant_lightning_indexer_kernel.h"

namespace a5_fused_li_manage_mtp_c8_impl {
using namespace AscendC;
using namespace QLILdCommon;
using namespace QLILdKernel;

using C8MtpQliType = QLIType<
    fp8_e4m3fn_t, fp8_e4m3fn_t, float, uint16_t, int32_t, true,
    LI_LAYOUT::TND, LI_LAYOUT::PA_BSND>;

class QuantLiMtpPhase {
public:
    __aicore__ inline QuantLiMtpPhase(
        TPipe *pipe, const A5FusedLiManageMtpC8TilingData *tiling)
        : pipe_(pipe), tiling_(tiling)
    {}

    __aicore__ inline void Init(
        GM_ADDR query, GM_ADDR key, GM_ADDR weights,
        GM_ADDR queryDequantScale, GM_ADDR keyDequantScale,
        GM_ADDR actualSeqLengthsQuery, GM_ADDR cacheTokens,
        GM_ADDR candidateLens, GM_ADDR reqPoolEntries,
        GM_ADDR cacheSlotsPool, GM_ADDR blockTable,
        GM_ADDR routePairRows, GM_ADDR topkSourceIds, GM_ADDR topkSlots,
        GM_ADDR routeThresholds, GM_ADDR routeMissCounts,
        GM_ADDR userWorkspace, GM_ADDR ldWorkspace)
    {
        engine_.Init(
            query, key, weights, queryDequantScale, keyDequantScale,
            actualSeqLengthsQuery, candidateLens, blockTable,
            topkSourceIds, routePairRows, topkSlots, routeMissCounts,
            routeThresholds, cacheSlotsPool, reqPoolEntries, cacheTokens,
            userWorkspace, ldWorkspace, tiling_, pipe_);
    }

    __aicore__ inline void Process()
    {
        engine_.Process();
    }

private:
    TPipe *pipe_;
    const A5FusedLiManageMtpC8TilingData *tiling_;
    QLIPreload<C8MtpQliType> engine_;
};
} // namespace a5_fused_li_manage_mtp_c8_impl

#endif
