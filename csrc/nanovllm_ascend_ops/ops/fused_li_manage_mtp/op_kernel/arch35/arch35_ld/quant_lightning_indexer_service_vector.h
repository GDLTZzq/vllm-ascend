/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_lightning_indexer_service_vector.h
 * \brief
 */
#ifndef quant_lightning_indexer_SERVICE_VECTOR_H
#define quant_lightning_indexer_SERVICE_VECTOR_H

// 调试二分（stage1 内部）：=1 时 PublishRoute 退化为「只把 top-2048 源索引写
// topkSourceIds，跳过 slot 查表/classify/sort/阈值/missCount」。事件配对不变。
// 裸 winner 探针已验证 LD 归并正确（set_ok=True），恢复 0 跑完整 classify。
#ifndef C8_MTP_BISECT_SKIP_PUBLISH
#define C8_MTP_BISECT_SKIP_PUBLISH 0
#endif

// 一次性标量诊断：=1 时，writer（ProcessTopK isNeedLD 主 store 分支）与 merge
// （ProcessLD 归并行）各打一行 [DUMPW]/[DUMPL]，外加 blockId 1..3 的
// [CLSP0/1/2/3]。DIAG：官方测试 candidate=16384 下 stage1 classify 每条 route
// miss≈1220（应 100）虚高——根因是 classify GetSpr<AR> 竞态（见 ~1245 PIPE_ALL
// 修复），非 merge/classify 数据差异；诊断恢复 0。
#ifndef C8_MTP_DUMP_LD_SLOTS
#define C8_MTP_DUMP_LD_SLOTS 0
#endif

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "quant_lightning_indexer_common.h"
#include "../arch35_ld/vf/quant_lightning_indexer_vector1.h"
#include "../arch35_ld/vf/quant_lightning_indexer_topk.h"
#include "../arch35_ld/vf/a5_fused_li_manage_classify_vf.h"

namespace QLILdKernel {
using namespace QLILdCommon;
constexpr uint32_t TRUNK_LEN_16K = 8192;
// outBase 常驻槽位深度。必须为 2：BatchMulWeightAndReduceSum(procNum=2) 把 row1 写到
// outBase+256 elements（outStride=256，0.5KB/行），两行共 [0,512) elements = 1KB outBuf_。
// D=4 会让 slot-s 的 row1 落到 slot-s+2 的 row0 区（几何冲突）且 slot 2/3 row1 越界写。
// score 拷贝事件（VEC1_MTE3_V_EVENT / VEC1_V_MTE3_EVENT）随之按槽位（loop%OUT_PINGPONG_DEPTH）分配；
// 跨核 CV/VC 与 resMm1/weight/qScale/kScale 仍按 loop%2，不放大 CUBE 侧流水。
constexpr uint32_t OUT_PINGPONG_DEPTH = 2;
template <typename QLIT>
class QLIVector {
public:
    // =================================类型定义区=================================
    static constexpr LI_LAYOUT Q_LAYOUT_T = QLIT::layout;
    static constexpr LI_LAYOUT K_LAYOUT_T = QLIT::keyLayout;
    static constexpr bool PAGE_ATTENTION = QLIT::pageAttention;

    using QK_T = typename QLIT::queryKeyType;
    using SCORE_T = typename QLIT::scoreType;

    __aicore__ inline QLIVector(){};
    __aicore__ inline void ProcessVec1(const QLILdCommon::RunInfo &info);
    __aicore__ inline void ProcessTopK(const QLILdCommon::RunInfo &info);
    __aicore__ inline void ProcessLD();
    __aicore__ inline void InitBuffers(TPipe *pipe);
    __aicore__ inline void InitParams(const struct QLILdCommon::ConstInfo &constInfo,
                                      const struct QLILdCommon::LdSplitCoreInfo &ldInfo,
                                      const QLITilingData *__restrict tilingData);
    __aicore__ inline void InitVecWorkspaceTensor(GlobalTensor<SCORE_T> scoreGm,
                                                    GlobalTensor<SCORE_T> ldScoreGm,
                                                    GlobalTensor<int32_t> ldIndexGm);
    __aicore__ inline void InitVecInputTensor(GlobalTensor<bfloat16_t> weightsGm, GlobalTensor<float> qScaleGm,
                                              GlobalTensor<float> kScaleGm, GlobalTensor<int32_t> indiceOutGm,
                                              GlobalTensor<int32_t> blockTableGm);
    __aicore__ inline void InitMtpOutputTensor(
        GlobalTensor<int32_t> routePairsGm, GlobalTensor<int32_t> topkSlotsGm,
        GlobalTensor<int32_t> missCountGm, GlobalTensor<uint16_t> thresholdGm,
        GlobalTensor<int32_t> cacheSlotsGm, GlobalTensor<int32_t> reqPoolEntriesGm,
        GlobalTensor<int32_t> cacheTokensGm);
    __aicore__ inline void CleanInvalidOutput(int64_t invalidS1offset);
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();
    __aicore__ inline void InitLDBuffers(TPipe *pipe, const struct QLILdCommon::LdSplitCoreInfo &ldInfo);
    __aicore__ inline void SetScoreRowBase(int64_t rowBase);
    __aicore__ inline void SetLdSeqLen(int32_t actS1Size, int32_t actS2SizeOrig);
    __aicore__ inline void PublishRoute(uint32_t outputRow, uint32_t cacheRowIdx,
                                        uint32_t cacheTokenCount);

protected:
    GlobalTensor<SCORE_T> scoreGm;
    GlobalTensor<bfloat16_t> weightsGm;
    GlobalTensor<SCORE_T> ldScoreGm;
    GlobalTensor<int32_t> ldIndexGm;
    GlobalTensor<float> qScaleGm;
    GlobalTensor<float> kScaleGm;
    GlobalTensor<int32_t> indiceOutGm;
    GlobalTensor<int32_t> blockTableGm;
    // MTP classify+publish 输出/查询张量：indiceOutGm 绑定 topkSourceIds 区，
    // PublishRoute 复用其作为 source 输出；routePairs/topkSlots/missCount/threshold 独立区。
    GlobalTensor<int32_t> routePairsGm;
    GlobalTensor<int32_t> topkSlotsGm;
    GlobalTensor<int32_t> missCountGm;
    GlobalTensor<uint16_t> thresholdGm;
    GlobalTensor<int32_t> cacheSlotsGm;
    GlobalTensor<int32_t> reqPoolEntriesGm;
    GlobalTensor<int32_t> cacheTokensGm;
    // =================================常量区=================================
    // 官方 arch35 事件分派：weight/qScale（QSCALE）与 kScale（KSCALE）各用独立事件对，
    // 与 score 拷贝（VEC1_MTE3_V_EVENT+pingpong）解耦。跨 HardEvent 类的 ID 复用同官方。
    static constexpr uint32_t VEC1_V_MTE2_EVENT_KSCALE = EVENT_ID0;
    static constexpr uint32_t VEC1_MTE2_V_EVENT_KSCALE = EVENT_ID1;
    static constexpr uint32_t VEC1_V_MTE3_EVENT = EVENT_ID2;
    static constexpr uint32_t VEC1_MTE3_V_EVENT = EVENT_ID3;
    static constexpr uint32_t VEC1_V_MTE2_EVENT_QSCALE = EVENT_ID6;
    static constexpr uint32_t VEC1_MTE2_V_EVENT_QSCALE = EVENT_ID3;

    static constexpr uint32_t TOPK_V_MTE2_EVENT = EVENT_ID4;
    static constexpr uint32_t TOPK_MTE2_V_EVENT = EVENT_ID5;
    static constexpr uint32_t TOPK_V_MTE3_EVENT = EVENT_ID6;
    static constexpr uint32_t TOPK_MTE3_V_EVENT = EVENT_ID7;

    static constexpr uint32_t KSCALE_S_MTE2_EVENT = EVENT_ID7;
    static constexpr uint32_t MTE3_MTE2_EVENT = EVENT_ID0;
    static constexpr uint32_t V_MTE2_EVENT = EVENT_ID7;
    static constexpr uint32_t V_MTE2_EVENT1 = EVENT_ID2;
    static constexpr uint32_t V_MTE2_EVENT2 = EVENT_ID3;
    static constexpr uint32_t V_MTE2_EVENT3 = EVENT_ID5;

private:
    __aicore__ inline void GetKeyScale(const QLILdCommon::RunInfo &runInfo, LocalTensor<float> &kScaleUB,
                                       int64_t batchId, int64_t startS2, int64_t getLen);
    // classify+publish 复用层（同 payload 引擎 FinalizePayloadUpdate 的 MTP 块）
    __aicore__ inline void BuildSortedMissPairs(
        const LocalTensor<int32_t> &classifiedIndex, uint32_t missCount);
    __aicore__ inline void ExtractSortedMissSourceIds(
        const LocalTensor<int32_t> &output, uint32_t missCount);
    // ================================Local Buffer区====================================

    // tmp buff for vector
    TBuf<TPosition::VECCALC> resMm1Buf_;
    LocalTensor<QK_T> resMm1UB_;
    //tmp buff for weight（MTP 输入为 bf16，先入 weightUB_ 再 Cast 到 weightFloatUB_）
    TBuf<TPosition::VECCALC> weightBuf_;
    LocalTensor<bfloat16_t> weightUB_;
    LocalTensor<float> weightFloatUB_;  // 别名到 topkSharedTmpLocal_，ProcessVec1 Cast 的目标
    //tmp buff for kScale
    TBuf<TPosition::VECCALC> kScaleBuf_;
    LocalTensor<float> kScaleUB_;
    //tmp buff for qScale
    TBuf<TPosition::VECCALC> qScaleBuf_;
    LocalTensor<float> qScaleUB_;
    //tmp buff for out
    TBuf<TPosition::VECCALC> outBuf_;
    LocalTensor<SCORE_T> vec1OutUB_;
    // tmp buff for LD
    TBuf<TPosition::VECCALC> ldValueBuf_;
    LocalTensor<uint32_t> ldValueLocal_;  // SCORE_T

    TBuf<TPosition::VECCALC> topkIndexBuf_;
    LocalTensor<uint32_t> topkIndexLocal_;

    TBuf<TPosition::VECCALC> topkValueBuf_;
    LocalTensor<uint32_t> topkValueLocal_;

    // tmp buff for topk
    TBuf<TPosition::VECCALC> mrgValueBuf_;
    LocalTensor<SCORE_T> mrgValueLocal_;

    TBuf<TPosition::VECCALC> indicesOutBuf_;
    LocalTensor<uint32_t> indicesOutLocal_;

    TBuf<TPosition::VECCALC> scoreOutBuf_;
    LocalTensor<SCORE_T> scoreOutLocal_;

    TBuf<TPosition::VECCALC> topkSharedTmpBuf_;
    LocalTensor<uint32_t> topkSharedTmpLocal_;

    TBuf<TPosition::VECCALC> outInvalidBuf_;
    LocalTensor<int32_t> outInvalidLocal_;

    LocalTensor<uint32_t> ldIndexLocal_;

    int32_t blockId_ = -1;
    // para for vector
    int32_t groupInner_ = 0;
    int32_t globalTopkNum_ = 0;
    int64_t blockS2StartIdx_ = 0;
    int32_t gSize_ = 0;
    int32_t kSeqSize_ = 0;
    int32_t kHeadNum_ = 0;
    int32_t qHeadNum_ = 0;
    int32_t s1BaseSize_ = 0;
    int32_t s2BaseSize_ = 0;
    int32_t kCacheBlockSize_ = 0;
    int32_t maxBlockNumPerBatch_ = 0;
    uint32_t topkCount_ = 0;
    uint32_t topkCountAlign256_ = 0; // topkCount对齐到256(直方图需要)，支持topk泛化
    uint32_t topkCountAlign16_ = 0; // LD读取到UB，需要满足32B对齐
    uint32_t trunkLen_ = 0;
    bool isWholeRowGreedy_ = false; // Branch 2 走官方 ProcessVec1（load-once + kScale 16块批量）
    // MTP request-major score 布局：scoreRowBase_ = bN2Idx*qSeqSize*sourceCapacity_
    // + gS1Idx*s1BaseSize_*sourceCapacity_；行 r 的 GM 地址 = scoreRowBase_ + r*sourceCapacity_。
    int64_t scoreRowBase_ = 0;
    int64_t sourceCapacity_ = 0;    // Align(kSeqSize, 128)，request-major 行步长（SCORE_T 元素）
    // LD 归并行有效 S2 长度 = ldActS2SizeOrig_ - ldActS1Size_ + row（每行候选尾部 route 偏移）
    int32_t ldActS1Size_ = 0;
    int32_t ldActS2SizeOrig_ = 0;

    struct QLILdCommon::ConstInfo constInfo_;
    struct QLILdCommon::LdSplitCoreInfo ldInfo_;
    topk::LITopk<SCORE_T> topkOp_;
};

template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::InitBuffers(TPipe *pipe)
{
    pipe->InitBuffer(resMm1Buf_, 2 * CeilDiv(constInfo_.mBaseSize, 2) * s2BaseSize_ * sizeof(QK_T));   //大小：2(开dB) * 2 * 64 * 128 * 4 = 128KB
    resMm1UB_ = resMm1Buf_.Get<QK_T>();//qk
    // pingpong load 偏移固定 UB_BANK_STRIDE/4=256B，缓冲须≥256B+块内数据；2×CeilDiv×gSize×4 在 s1BaseSize=2 仅 256B 会写穿 kScale/out
    pipe->InitBuffer(weightBuf_, 2 * CeilDiv(s1BaseSize_, 2) * UB_BANK_DEPTH_STRIDE);
    weightUB_ = weightBuf_.Get<bfloat16_t>();//weight
    // 官方整行路径 kScale 每 16 块批量加载需 2*16*s2BaseSize 槽位（16KB）；Branch 1 逐块加载 1KB 即可（UB 预算不放大）
    pipe->InitBuffer(kScaleBuf_, isWholeRowGreedy_ ? (2 * s2BaseSize_ * 16 * sizeof(float)) : (2 * s2BaseSize_ * sizeof(float)));
    kScaleUB_ = kScaleBuf_.Get<float>();//kScale
    pipe->InitBuffer(qScaleBuf_, 2 * CeilDiv(s1BaseSize_, 2) * UB_BANK_DEPTH_STRIDE);
    qScaleUB_ = qScaleBuf_.Get<float>();//qScale
    pipe->InitBuffer(outBuf_, 2 * CeilDiv(s1BaseSize_, 2) * s2BaseSize_ * sizeof(SCORE_T));      // gS1=4: 2×2×128×2B = 1KB。两行(gSize/2)按 outStride=256elements 交错排进 [0,512)（槽0行0+槽1行0在[0,256)，两槽 row1 在[256,512)），与 OUT_PINGPONG_DEPTH=2 恰合
    vec1OutUB_ = outBuf_.Get<SCORE_T>();//out

    // Topk
    pipe->InitBuffer(mrgValueBuf_, (topkCountAlign256_ + trunkLen_) * sizeof(SCORE_T));     // 大小：(topkCountAlign256_ + 每次排序长度) * sizeof(SCORE_T)
    mrgValueLocal_ = mrgValueBuf_.Get<SCORE_T>();
    
    pipe->InitBuffer(indicesOutBuf_, (topkCountAlign256_ + 64) * sizeof(uint32_t));         // 大小：(topkCountAlign256_ + 64) * 4  64:duplicate刷-1需要额外空间
    indicesOutLocal_ = indicesOutBuf_.Get<uint32_t>();

    pipe->InitBuffer(scoreOutBuf_, (topkCountAlign256_ + 64) * sizeof(SCORE_T));            // (topkCountAlign256_ + 64) * sizeof(SCORE_T) 64:duplicate刷-1额外空间
    scoreOutLocal_ = scoreOutBuf_.Get<SCORE_T>();

    uint64_t topkSharedTmpSize = topkOp_.GetSharedTmpBufferSize();
    pipe->InitBuffer(topkSharedTmpBuf_, topkSharedTmpSize);
    topkSharedTmpLocal_ = topkSharedTmpBuf_.Get<uint32_t>();
    topkOp_.InitBuffers(topkSharedTmpLocal_);
    // weightFloatUB_ 复用 topkSharedTmpLocal_ 空闲区（ProcessVec1 与 topk 串行，
    // 无并发覆盖）；每行 512B（float 128 元素），与 qScale 预乘前的 Cast 目标。
    weightFloatUB_ = topkSharedTmpLocal_.template ReinterpretCast<float>();

    //刷-1
    pipe->InitBuffer(outInvalidBuf_, topkCount_ * sizeof(int32_t));
    outInvalidLocal_ = outInvalidBuf_.Get<int32_t>();
    Duplicate(kScaleUB_, float(0), isWholeRowGreedy_ ? (2 * s2BaseSize_ * 16) : (2 * s2BaseSize_));
}

template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::InitLDBuffers(TPipe *pipe, const struct QLILdCommon::LdSplitCoreInfo &ldInfo)
{
    ldIndexLocal_ = resMm1UB_.template ReinterpretCast<uint32_t>();
}

template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::InitParams(const struct QLILdCommon::ConstInfo &constInfo,
                                                   const struct QLILdCommon::LdSplitCoreInfo &ldInfo,
                                                   const QLITilingData *__restrict tilingData)
{
    this->constInfo_ = constInfo;
    this->ldInfo_ = ldInfo;
    blockS2StartIdx_ = 0;
    // DIAG(2026-09-04)：强制关 whole-row greedy（Branch1 逐块加载）。混合 batch=17>16 原走
    // Branch2；奇行(q1/q3)深部 qK≈0 疑在 Branch2 procNum=2 行处理。若强制 Branch1 后奇行恢复
    // → 定位 Branch2；仍坏 → cube 侧。定位后还原 constInfo.isWholeRowGreedy。
    isWholeRowGreedy_ = false;
    gSize_ = constInfo.gSize;
    kSeqSize_ = constInfo.kSeqSize;
    // define N2 para
    kHeadNum_ = constInfo.kHeadNum;
    qHeadNum_ = constInfo.qHeadNum;
    // define MMBase para
    s1BaseSize_ = constInfo.s1BaseSize;  // 4
    s2BaseSize_ = constInfo.s2BaseSize;  // 128
    sourceCapacity_ = QLILdCommon::Align((uint64_t)constInfo.kSeqSize, (uint64_t)s2BaseSize_);
    kCacheBlockSize_ = constInfo.kCacheBlockSize;
    maxBlockNumPerBatch_ = constInfo.maxBlockNumPerBatch;
    blockId_ = GetBlockIdx();
    trunkLen_ = TRUNK_LEN_16K;
    topkCount_ = constInfo.sparseCount;
    topkCountAlign256_ = QLILdCommon::Align(constInfo.sparseCount, (uint64_t)256); // topkCount对齐到256
    topkCountAlign16_ = QLILdCommon::Align(constInfo.sparseCount, (uint64_t)16); // topkCount对齐到16
    topkOp_.Init(topkCount_, trunkLen_);
}

template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::InitVecInputTensor(GlobalTensor<bfloat16_t> weightsGm, GlobalTensor<float> qScaleGm,
                                                           GlobalTensor<float> kScaleGm,
                                                           GlobalTensor<int32_t> indiceOutGm,
                                                           GlobalTensor<int32_t> blockTableGm)
{
    this->weightsGm = weightsGm;
    this->qScaleGm = qScaleGm;
    this->kScaleGm = kScaleGm;
    this->indiceOutGm = indiceOutGm;
    this->blockTableGm = blockTableGm;
}

template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::InitMtpOutputTensor(
    GlobalTensor<int32_t> routePairsGm, GlobalTensor<int32_t> topkSlotsGm,
    GlobalTensor<int32_t> missCountGm, GlobalTensor<uint16_t> thresholdGm,
    GlobalTensor<int32_t> cacheSlotsGm, GlobalTensor<int32_t> reqPoolEntriesGm,
    GlobalTensor<int32_t> cacheTokensGm)
{
    this->routePairsGm = routePairsGm;
    this->topkSlotsGm = topkSlotsGm;
    this->missCountGm = missCountGm;
    this->thresholdGm = thresholdGm;
    this->cacheSlotsGm = cacheSlotsGm;
    this->reqPoolEntriesGm = reqPoolEntriesGm;
    this->cacheTokensGm = cacheTokensGm;
}

template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::SetScoreRowBase(int64_t rowBase)
{
    scoreRowBase_ = rowBase;
}

template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::SetLdSeqLen(int32_t actS1Size, int32_t actS2SizeOrig)
{
    ldActS1Size_ = actS1Size;
    ldActS2SizeOrig_ = actS2SizeOrig;
}

template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::InitVecWorkspaceTensor(GlobalTensor<SCORE_T> scoreGm,
                                                            GlobalTensor<SCORE_T> ldScoreGm,
                                                            GlobalTensor<int32_t> ldIndexGm)
{
    this->scoreGm = scoreGm;//resucesum*k
    this->ldScoreGm = ldScoreGm;
    this->ldIndexGm = ldIndexGm;
}

template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::AllocEventID()
{
    SetFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT_KSCALE + 0);
    SetFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT_KSCALE + 1);
    SetFlag<HardEvent::MTE3_V>(VEC1_MTE3_V_EVENT + 0);
    SetFlag<HardEvent::MTE3_V>(VEC1_MTE3_V_EVENT + 1);
    SetFlag<HardEvent::MTE3_V>(VEC1_MTE3_V_EVENT + 2);
    SetFlag<HardEvent::MTE3_V>(VEC1_MTE3_V_EVENT + 3);
    // QSCALE 事件仅 Branch 2（整行 load-once）使用。Branch 1 若预置则 QSCALE+1=EVENT_ID7
    // 会把陈旧置位泄漏进 ProcessTopK 的 V_MTE2_EVENT 自屏障（Set/Wait 立即通过，破坏 V→MTE2 排序）。
    if (isWholeRowGreedy_) {
        SetFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT_QSCALE + 0);
        SetFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT_QSCALE + 1);
    }

    SetFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
    SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
    SetFlag<HardEvent::V_MTE2>(V_MTE2_EVENT1);
}

template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::FreeEventID()
{
    WaitFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT_KSCALE + 0);
    WaitFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT_KSCALE + 1);
    WaitFlag<HardEvent::MTE3_V>(VEC1_MTE3_V_EVENT + 0);
    WaitFlag<HardEvent::MTE3_V>(VEC1_MTE3_V_EVENT + 1);
    WaitFlag<HardEvent::MTE3_V>(VEC1_MTE3_V_EVENT + 2);
    WaitFlag<HardEvent::MTE3_V>(VEC1_MTE3_V_EVENT + 3);
    if (isWholeRowGreedy_) {
        WaitFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT_QSCALE + 0);
        WaitFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT_QSCALE + 1);
    }

    WaitFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
    WaitFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
    WaitFlag<HardEvent::V_MTE2>(V_MTE2_EVENT1);
}

template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::CleanInvalidOutput(int64_t invalidS1Offset)
{
    // init -1 and copy to output
    Duplicate(outInvalidLocal_, constInfo_.INVALID_IDX, constInfo_.sparseCount);

    SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
    WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);

    AscendC::DataCopyParams dataCopyOutParams;
    dataCopyOutParams.blockCount = 1;
    dataCopyOutParams.blockLen = constInfo_.sparseCount * sizeof(int32_t);
    dataCopyOutParams.srcStride = 0;
    dataCopyOutParams.dstStride = 0;
    AscendC::DataCopyPad(indiceOutGm[invalidS1Offset], outInvalidLocal_, dataCopyOutParams);
}

template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::GetKeyScale(const QLILdCommon::RunInfo &runInfo, LocalTensor<float> &kScaleUB,
                                                    int64_t batchId, int64_t startS2, int64_t getLen)
{
    // startS2一定能整除kCacheBlockSize_
    AscendC::DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
    AscendC::DataCopyExtParams copyInParams;
    // 官方整行路径按 kScaleLoop（每16块+1）选双缓冲槽（每槽16*s2BaseSize 元素）；
    // Branch 1 逐块加载按 loop%2 选 1 块槽。
    int32_t kScaleDstBase = isWholeRowGreedy_
        ? (16 * (int32_t)(runInfo.kScaleLoop % 2) * s2BaseSize_)
        : ((int32_t)(runInfo.loop % 2) * s2BaseSize_);
    if constexpr (PAGE_ATTENTION) {
        int32_t startBlockTableIdx = startS2 / kCacheBlockSize_;
        int32_t startBlockTableOffset = startS2 % kCacheBlockSize_;
        int32_t blockTableBatchOffset = batchId * maxBlockNumPerBatch_;
        copyInParams.blockCount = 1;
        copyInParams.srcStride = 0;
        copyInParams.dstStride = 0;
        copyInParams.rsv = 0;
        int32_t resUbBaseOffset = 0;
        if (startBlockTableOffset > 0) {
            int32_t firstPartLen =
                kCacheBlockSize_ - startBlockTableOffset > getLen ? getLen : kCacheBlockSize_ - startBlockTableOffset;
            copyInParams.blockLen = firstPartLen * sizeof(float);
            int32_t blockId = blockTableGm.GetValue(blockTableBatchOffset + startBlockTableIdx);
            SetFlag<HardEvent::S_MTE2>(KSCALE_S_MTE2_EVENT);
            WaitFlag<HardEvent::S_MTE2>(KSCALE_S_MTE2_EVENT);
            AscendC::DataCopyPad(kScaleUB[kScaleDstBase], kScaleGm[blockId * constInfo_.keyDequantScaleStride0 + startBlockTableOffset],
                                 copyInParams, padParams);
            startBlockTableIdx++;
            getLen = getLen - firstPartLen;
            resUbBaseOffset = firstPartLen;
        }
        int32_t getLoopNum = CeilDiv(getLen, kCacheBlockSize_);
        copyInParams.blockLen = kCacheBlockSize_ * sizeof(float);
        for (int32_t i = 0; i < getLoopNum; i++) {
            if (i == getLoopNum - 1) {
                copyInParams.blockLen = (getLen - i * kCacheBlockSize_) * sizeof(float);
            }
            int32_t blockId = blockTableGm.GetValue(blockTableBatchOffset + startBlockTableIdx + i);
            SetFlag<HardEvent::S_MTE2>(KSCALE_S_MTE2_EVENT);
            WaitFlag<HardEvent::S_MTE2>(KSCALE_S_MTE2_EVENT);
            AscendC::DataCopyPad(kScaleUB[kScaleDstBase + resUbBaseOffset + i * kCacheBlockSize_],
                                 kScaleGm[blockId * constInfo_.keyDequantScaleStride0],
                                 copyInParams, padParams);
        }
    } else {
        copyInParams.blockCount = 1;
        copyInParams.blockLen = getLen * sizeof(float);
        copyInParams.srcStride = 0;
        copyInParams.dstStride = 0;
        copyInParams.rsv = 0;
        AscendC::DataCopyPad(kScaleUB[kScaleDstBase], kScaleGm[runInfo.tensorKeyScaleOffset], copyInParams, padParams);
    }
}

template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::ProcessVec1(const QLILdCommon::RunInfo &info)
{
    auto pingpong = (info.loop % 2);
    auto outSlot = (info.loop % OUT_PINGPONG_DEPTH);  // outBase 槽 2 深：row0/row1 两行只占一槽，见 OUT_PINGPONG_DEPTH 注释
    auto qScalepingpong = (info.qScaleLoop % 2);  // 每行+1：weight/qScale 槽位（官方同式）
    auto kScalepingpong = (info.kScaleLoop % 2);  // 每16块+1：kScale 槽位（官方同式）
    auto s1BaseSizePerAIV = CeilDiv(s1BaseSize_, 2);
    int64_t curS1Idx = info.gS1Idx * s1BaseSize_;
    int64_t curS2Idx = info.s2Idx * s2BaseSize_;	 
    int64_t curS1ProcNum = curS1Idx + s1BaseSize_ > info.actS1Size ? info.actS1Size % s1BaseSize_ : s1BaseSize_;	 
    int64_t curAivS1Idx = curS1Idx + (blockId_ % 2) * CeilDiv(curS1ProcNum, 2);
    int64_t curAivS1ProcNum = (blockId_ % 2 == 0) ? CeilDiv(curS1ProcNum, 2) : curS1ProcNum / 2; 

    if (curAivS1ProcNum == 0) {
        CrossCoreWaitFlag<QLILdCommon::ConstInfo::QLI_SYNC_MODE4, PIPE_V>(QLILdCommon::ConstInfo::CROSS_CV_EVENT + pingpong);  // V核等C核计算完mm1，mm1Res已搬运到UB
        CrossCoreSetFlag<QLILdCommon::ConstInfo::QLI_SYNC_MODE4, PIPE_V>(QLILdCommon::ConstInfo::CROSS_VC_EVENT + pingpong);   // V核处理完，通知C核可以把mm1Res搬运到UB
        return;
    }
    int64_t weightGmOffset = info.tensorWeightsOffset + curAivS1Idx * kHeadNum_ * gSize_;
    static_assert(std::is_same_v<SCORE_T, uint16_t>);
    auto outBase = vec1OutUB_[outSlot * (UB_BANK_STRIDE / sizeof(SCORE_T))];
    if (isWholeRowGreedy_) {
        // ===== 官方整行路径：weight/qScale 每行仅首块加载、kScale 每16块批量加载 =====
        if (info.isFirstS2InnerLoop) {
            WaitFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT_QSCALE + qScalepingpong);
            //weightsGm(bf16) --> weightUB_
            DataCopyPadExtParams<bfloat16_t> padWeightsParams{false, 0, 0, 0};
            DataCopyExtParams wDataCopyExtParams;
            wDataCopyExtParams.blockCount = curAivS1ProcNum;
            wDataCopyExtParams.blockLen = gSize_ * sizeof(bfloat16_t);
            wDataCopyExtParams.srcStride = 0;
            wDataCopyExtParams.dstStride = (UB_BANK_DEPTH_STRIDE - wDataCopyExtParams.blockLen) / 32;
            DataCopyPad(weightUB_[qScalepingpong * (UB_BANK_STRIDE / sizeof(bfloat16_t))],
                        weightsGm[weightGmOffset], wDataCopyExtParams, padWeightsParams);
            //qScaleGm --> qScaleUB_
            DataCopyPadExtParams<float> padQScaleParams{false, 0, 0, 0};
            DataCopyExtParams qDataCopyExtParams;
            qDataCopyExtParams.blockCount = curAivS1ProcNum;
            qDataCopyExtParams.blockLen = gSize_ * sizeof(float);
            qDataCopyExtParams.srcStride = 0;
            qDataCopyExtParams.dstStride = (UB_BANK_DEPTH_STRIDE - qDataCopyExtParams.blockLen) / 32;
            DataCopyPad(qScaleUB_[qScalepingpong * (UB_BANK_STRIDE / sizeof(float))],
                        qScaleGm[weightGmOffset], qDataCopyExtParams, padQScaleParams);
            SetFlag<HardEvent::MTE2_V>(VEC1_MTE2_V_EVENT_QSCALE + qScalepingpong);
            WaitFlag<HardEvent::MTE2_V>(VEC1_MTE2_V_EVENT_QSCALE + qScalepingpong);
            // bf16 weight 转 float；行 r 在 weightUB_ 为 512B 行距（256 bf16 元素），
            // weightFloatUB_ 为 512B 行距（128 float 元素），一一对应后交给
            // BatchMulWeightAndReduceSum（其内部完成 weight*qScale 预乘）。
            for (int64_t r = 0; r < curAivS1ProcNum; ++r) {
                Cast(weightFloatUB_[r * (UB_BANK_DEPTH_STRIDE / sizeof(float))],
                     weightUB_[qScalepingpong * (UB_BANK_STRIDE / sizeof(bfloat16_t)) +
                               r * (UB_BANK_DEPTH_STRIDE / sizeof(bfloat16_t))],
                     RoundMode::CAST_NONE, gSize_);
            }
        }
        if ((info.s2Idx - info.s2Start) % 16 == 0) {
            WaitFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT_KSCALE + kScalepingpong);
            //kScaleGm --> kScaleUB_：一次取 16 块（尾组截断）
            uint32_t remainLen = (info.actS2Size > info.s2Idx * (uint32_t)s2BaseSize_)
                                     ? (uint32_t)(info.actS2Size - info.s2Idx * (uint32_t)s2BaseSize_)
                                     : 0;
            uint32_t getLen = (16 * (uint32_t)s2BaseSize_ > remainLen) ? remainLen : 16 * (uint32_t)s2BaseSize_;
            GetKeyScale(info, kScaleUB_, info.bIdx, curS2Idx, getLen);
            SetFlag<HardEvent::MTE2_V>(VEC1_MTE2_V_EVENT_KSCALE + kScalepingpong);
            WaitFlag<HardEvent::MTE2_V>(VEC1_MTE2_V_EVENT_KSCALE + kScalepingpong);
        }
        WaitFlag<HardEvent::MTE3_V>(VEC1_MTE3_V_EVENT + outSlot);

        //CV同步
        CrossCoreWaitFlag<QLILdCommon::ConstInfo::QLI_SYNC_MODE4, PIPE_V>(QLILdCommon::ConstInfo::CROSS_CV_EVENT + info.loop % 2);   //V核等C核计算完mm1，mm1Res已搬运到UB

        auto weightFloatBase = weightFloatUB_;
        auto qScaleBase = qScaleUB_[qScalepingpong * (UB_BANK_STRIDE / sizeof(float))];
        auto kScaleBase = kScaleUB_[kScalepingpong * 16 * s2BaseSize_ +
                                    ((info.s2Idx - info.s2Start) % 16) * s2BaseSize_];

        auto qkBase = resMm1UB_[pingpong * (UB_BANK_STRIDE / sizeof(QK_T))];
        auto qkVLstride = (UB_BANK_DEPTH_STRIDE / sizeof(QK_T)) / 2 * constInfo_.mBaseSize;
        vec1ld::BatchMulWeightAndReduceSum(outBase, UB_BANK_DEPTH_STRIDE / sizeof(SCORE_T),
                                            qkBase, qkVLstride, (uint32_t)(gSize_ * UB_BANK_DEPTH_STRIDE / sizeof(QK_T)),
                                            weightFloatBase, UB_BANK_DEPTH_STRIDE / sizeof(float),
                                            kScaleBase, (uint32_t)0,
                                            qScaleBase, UB_BANK_DEPTH_STRIDE / sizeof(float),
                                            gSize_, curAivS1ProcNum);
        if (info.isFirstS2InnerLoop) {
            SetFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT_QSCALE + qScalepingpong);
        }
        if ((info.s2Idx - info.s2Start) % 16 == 0) {
            SetFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT_KSCALE + kScalepingpong);
        }
    } else {
        // ===== 小 batch 拆 S2 + LD 路径：逐块加载 weight/qScale/kScale（保持现状）=====
        WaitFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT_KSCALE + pingpong);
#if C8_MTP_BISECT_PRINT
        if (GetBlockIdx() == 12U && info.loop == 0U) {
            AscendC::printf("[C8BISECT] AIV12 PV1 loop0 enter curAivS1Idx=%lld curS2Idx=%lld proc=%lld\n",
                            (int64_t)curAivS1Idx, (int64_t)curS2Idx, (int64_t)curAivS1ProcNum);
        }
#endif
        //weightsGm(bf16) --> weightUB_
        DataCopyPadExtParams<bfloat16_t> padWeightsParams{false, 0, 0, 0};
        DataCopyExtParams wDataCopyExtParams;
        wDataCopyExtParams.blockCount = curAivS1ProcNum;
        wDataCopyExtParams.blockLen = gSize_ * sizeof(bfloat16_t);
        wDataCopyExtParams.srcStride = 0;
        wDataCopyExtParams.dstStride = (UB_BANK_DEPTH_STRIDE - wDataCopyExtParams.blockLen) / 32;
        DataCopyPad(weightUB_[pingpong * (UB_BANK_STRIDE / sizeof(bfloat16_t))],
                    weightsGm[weightGmOffset], wDataCopyExtParams, padWeightsParams);
        //qScaleGm --> qScaleUB_
        DataCopyPadExtParams<float> padQScaleParams{false, 0, 0, 0};
        DataCopyExtParams qDataCopyExtParams;
        qDataCopyExtParams.blockCount = curAivS1ProcNum;
        qDataCopyExtParams.blockLen = gSize_ * sizeof(float);
        qDataCopyExtParams.srcStride = 0;
        qDataCopyExtParams.dstStride = (UB_BANK_DEPTH_STRIDE - qDataCopyExtParams.blockLen) / 32;
        DataCopyPad(qScaleUB_[pingpong * (UB_BANK_STRIDE / sizeof(float))],
                    qScaleGm[weightGmOffset], qDataCopyExtParams, padQScaleParams);
        //kScaleGm --> kScaleUB_
        GetKeyScale(info, kScaleUB_, info.bIdx, curS2Idx, info.actualSingleProcessSInnerSize);
        SetFlag<HardEvent::MTE2_V>(VEC1_MTE2_V_EVENT_KSCALE + pingpong);
        WaitFlag<HardEvent::MTE2_V>(VEC1_MTE2_V_EVENT_KSCALE + pingpong);
        WaitFlag<HardEvent::MTE3_V>(VEC1_MTE3_V_EVENT + outSlot);
#if C8_MTP_BISECT_PRINT
        if (GetBlockIdx() == 12U && info.loop == 0U) {
            AscendC::printf("[C8BISECT] AIV12 PV1 loop0 loads done\n");
        }
#endif

        //CV同步
        CrossCoreWaitFlag<QLILdCommon::ConstInfo::QLI_SYNC_MODE4, PIPE_V>(QLILdCommon::ConstInfo::CROSS_CV_EVENT + info.loop % 2);   //V核等C核计算完mm1，mm1Res已搬运到UB
#if C8_MTP_BISECT_PRINT
        if (GetBlockIdx() == 12U && info.loop == 0U) {
            AscendC::printf("[C8BISECT] AIV12 PV1 loop0 cv wait done\n");
        }
#endif

#if C8_MTP_BISECT_SKIP_VEC1
        // BISECT：跳过 Cast+BatchMul 向量算术，仅保留加载/事件/score GM 写。
        (void)0;
#else
        for (int64_t r = 0; r < curAivS1ProcNum; ++r) {
            Cast(weightFloatUB_[r * (UB_BANK_DEPTH_STRIDE / sizeof(float))],
                 weightUB_[pingpong * (UB_BANK_STRIDE / sizeof(bfloat16_t)) +
                           r * (UB_BANK_DEPTH_STRIDE / sizeof(bfloat16_t))],
                 RoundMode::CAST_NONE, gSize_);
        }
#if C8_MTP_BISECT_PRINT
        if (GetBlockIdx() == 12U && info.loop == 0U) {
            AscendC::printf("[C8BISECT] AIV12 PV1 loop0 cast done\n");
        }
#endif

        auto weightFloatBase = weightFloatUB_;
        auto qScaleBase = qScaleUB_[pingpong * (UB_BANK_STRIDE / sizeof(float))];
        auto kScaleBase = kScaleUB_[pingpong * s2BaseSize_];

        auto qkBase = resMm1UB_[pingpong * (UB_BANK_STRIDE / sizeof(QK_T))];
        auto qkVLstride = (UB_BANK_DEPTH_STRIDE / sizeof(QK_T)) / 2 * constInfo_.mBaseSize;
        vec1ld::BatchMulWeightAndReduceSum(outBase, UB_BANK_DEPTH_STRIDE / sizeof(SCORE_T),
                                            qkBase, qkVLstride, (uint32_t)(gSize_ * UB_BANK_DEPTH_STRIDE / sizeof(QK_T)),
                                            weightFloatBase, UB_BANK_DEPTH_STRIDE / sizeof(float),
                                            kScaleBase, (uint32_t)0,
                                            qScaleBase, UB_BANK_DEPTH_STRIDE / sizeof(float),
                                            gSize_, curAivS1ProcNum);
#if C8_MTP_BISECT_PRINT
        if (GetBlockIdx() == 12U && info.loop == 0U) {
            AscendC::printf("[C8BISECT] AIV12 PV1 loop0 batchmul done\n");
        }
#endif
#endif  // C8_MTP_BISECT_SKIP_VEC1
        SetFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT_KSCALE + pingpong);
    }
    SetFlag<HardEvent::V_MTE3>(VEC1_V_MTE3_EVENT + outSlot);
    WaitFlag<HardEvent::V_MTE3>(VEC1_V_MTE3_EVENT + outSlot);
#if C8_MTP_BISECT_PRINT
    if (GetBlockIdx() == 12U && info.loop == 0U) {
        AscendC::printf("[C8BISECT] AIV12 PV1 loop0 mte3 wait done\n");
    }
#endif
    //outUB_ --->  scoreGm（request-major：行 (curAivS1Idx+r) 落在
    // scoreRowBase_ + (curAivS1Idx+r)*sourceCapacity_ + curS2Idx）。
    // dstStride 为字节单位且 uint16（与已验证 standalone quant_lightning_indexer_c8
    // 同式：dstStride=(行跨度-块长) 字节）；sourceCapacity_=65536 时行跨度 131072 字节
    // 超 16bit 截断，单次多块 dstStride 无法编码行距。改逐行单块拷贝（blockCount=1），
    // dst 地址显式带行偏移，src 行距 = outBase 的 UB_BANK_DEPTH_STRIDE 行距（batchmul
    // 的行距参数 256 元素 = 512B，与 srcStride 原语义一致），任意 sourceCapacity 下成立。
    DataCopyExtParams copyOutParams;
    copyOutParams.blockCount = 1;
    copyOutParams.blockLen = s2BaseSize_ * sizeof(SCORE_T);
    copyOutParams.srcStride = 0;
    copyOutParams.dstStride = 0;
    for (int64_t row = 0; row < curAivS1ProcNum; ++row) {
        DataCopyPad(
            scoreGm[scoreRowBase_ + (curAivS1Idx + row) * sourceCapacity_ +
                    curS2Idx],
            outBase[row * (UB_BANK_DEPTH_STRIDE / sizeof(SCORE_T))],
            copyOutParams);
    }
#if C8_MTP_BISECT_PRINT
    // 逐行写唯一标记：确认本次运行编译到了 per-row 修复（request0 两个 AIV loop0）。
    if (GetBlockIdx() < 2U && info.loop == 0U) {
        AscendC::printf(
            "[C8BISECT] AIV%lld perrow marker cap=%lld proc=%lld aivIdx=%lld "
            "s2idx=%lld dstRow0=%lld dstRow1=%lld\n",
            (long long)GetBlockIdx(), (long long)sourceCapacity_,
            (long long)curAivS1ProcNum, (long long)curAivS1Idx,
            (long long)curS2Idx,
            (long long)(scoreRowBase_ + curAivS1Idx * sourceCapacity_ +
                        curS2Idx),
            (long long)(scoreRowBase_ + (curAivS1Idx + 1) * sourceCapacity_ +
                        curS2Idx));
    }
#endif
#if C8_MTP_BISECT_PRINT
    if (GetBlockIdx() == 12U && info.loop == 0U) {
        AscendC::printf("[C8BISECT] AIV12 PV1 loop0 score copy done\n");
    }
#endif
    SetFlag<HardEvent::MTE3_V>(VEC1_MTE3_V_EVENT + outSlot);
    CrossCoreSetFlag<QLILdCommon::ConstInfo::QLI_SYNC_MODE4, PIPE_V>(QLILdCommon::ConstInfo::CROSS_VC_EVENT + pingpong);   //V核处理完，通知C核可以把mm1Res搬运到UB
}

template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::ProcessTopK(const QLILdCommon::RunInfo &info)
{
    SetFlag<HardEvent::MTE3_MTE2>(MTE3_MTE2_EVENT);
    WaitFlag<HardEvent::MTE3_MTE2>(MTE3_MTE2_EVENT);

    int64_t curS1Idx = info.gS1Idx * s1BaseSize_;	 
    int64_t curS2StartIdx = info.s2Start * s2BaseSize_;
    int64_t curS2EndIdx = (info.s2LoopEnd + 1) * s2BaseSize_;
    int64_t curS1ProcNum = curS1Idx + s1BaseSize_ > info.actS1Size ? info.actS1Size % s1BaseSize_ : s1BaseSize_;	 
    int64_t curAivS1Idx = curS1Idx + (blockId_ % 2) * CeilDiv(curS1ProcNum, 2);
    int64_t curAivS1ProcNum = (blockId_ % 2 == 0) ? CeilDiv(curS1ProcNum, 2) : curS1ProcNum / 2; 

    // LD 需要搬运到的起始位置 32字节
    int64_t ldDstOffset = (info.isNeedLD) ? curS2StartIdx : 0;
    AscendC::DataCopyExtParams copyInParams;
    copyInParams.blockCount = 1;
    copyInParams.srcStride = 0;
    copyInParams.dstStride = 0;
    copyInParams.rsv = 0;

    AscendC::DataCopyParams copyOutParams;
    copyOutParams.blockCount = 1;
    copyOutParams.blockLen = topkCount_ * sizeof(uint32_t); // bytes
    copyOutParams.srcStride = 0;
    copyOutParams.dstStride = 0;

    AscendC::DataCopyParams ldCopyOutParams; // ldIndicesCopyOutParams
    ldCopyOutParams.blockCount = 1;
    ldCopyOutParams.blockLen = topkCountAlign16_ * sizeof(uint32_t); // bytes
    ldCopyOutParams.srcStride = 0;
    ldCopyOutParams.dstStride = 0;

    AscendC::DataCopyParams ldCopyScoreOutParams;
    ldCopyScoreOutParams.blockCount = 1;
    ldCopyScoreOutParams.blockLen = topkCountAlign16_ * sizeof(SCORE_T); // bytes
    ldCopyScoreOutParams.srcStride = 0;
    ldCopyScoreOutParams.dstStride = 0;

    int32_t cuRealAcSeq = info.actS2Size;
    if (constInfo_.attenMaskFlag) {
        cuRealAcSeq = info.actS2SizeOrig - info.actS1Size + curAivS1Idx + 1;
    } 

    int32_t validAllS2Len = cuRealAcSeq;
    for (uint32_t i = 0; i < curAivS1ProcNum; i++) {
        uint32_t rowIdx = blockId_ % 2 * CeilDiv(curS1ProcNum, 2) + i;

        SCORE_T zero = 0;
        int32_t neg = -1;
        uint32_t scoreAlign = sizeof(SCORE_T) == 4 ? 8 : 16; // int32对应UB对齐数为8，int16需要16
        if (constInfo_.attenMaskFlag) {
            validAllS2Len = ((int32_t)i + cuRealAcSeq) / static_cast<int32_t>(constInfo_.cmpRatio);
        }
        int32_t validS2Len = validAllS2Len;
        if (info.isNeedLD) {
            // 当前核处理的s2长度validS2Len
            validS2Len = Min((info.s2LoopEnd + 1) * s2BaseSize_, validAllS2Len) - curS2StartIdx;
        }
#if C8_MTP_DIAG_MC
        // DIAG[RQDIAG]：区分「validS2Len 每行算错(读侧)」还是「score 行深部=0(写侧)」。
        // 门不再要求 loop==0（gloop 是全核 s2-chunk 单调计数，ProcessTopK 只在 tile 最后
        // 一块跑，loop 必非 0），LD=0 行即 DIRECT（q0..q3），LD=1 为 split/merge 行。
        if (blockId_ <= 1U && validS2Len > 0) {
            AscendC::printf(
                "[RQDIAG] aiv=%lld i=%lld row=%lld s2Len=%lld acS2=%lld "
                "acS1=%lld mask=%lld cmp=%lld st=%lld end=%lld LD=%lld gS1=%lld\n",
                (int64_t)blockId_, (int64_t)i, (int64_t)rowIdx,
                (int64_t)validS2Len, (int64_t)info.actS2Size,
                (int64_t)info.actS1Size, (int64_t)constInfo_.attenMaskFlag,
                (int64_t)constInfo_.cmpRatio, (int64_t)info.s2Start,
                (int64_t)info.s2LoopEnd, (int64_t)info.isNeedLD,
                (int64_t)info.gS1Idx);
        }
        // DIAG[RQDEEP]：直读本行 scoreGm 在深源偏移的 SCORE 值，分辨 row(i=1) 深部是
        // 0(未算/被遮)、垃圾(写错位)、还是窜入别的行数据。仅 DIRECT 行(LD==0)安全读全窗。
        if (!info.isNeedLD && validS2Len > 0) {
            int64_t rowBase = scoreRowBase_ + (curAivS1Idx + i) * sourceCapacity_;
            AscendC::printf(
                "[RQDEEP] aiv=%lld i=%lld row=%lld rb=%lld "
                "s1000=%d s2100=%d s2200=%d s5000=%d s8000=%d s12200=%d\n",
                (int64_t)blockId_, (int64_t)i, (int64_t)rowIdx, (int64_t)rowBase,
                (int)scoreGm.GetValue(rowBase + 1000),
                (int)scoreGm.GetValue(rowBase + 2100),
                (int)scoreGm.GetValue(rowBase + 2200),
                (int)scoreGm.GetValue(rowBase + 5000),
                (int)scoreGm.GetValue(rowBase + 8000),
                (int)scoreGm.GetValue(rowBase + 12200));
        }
#endif

        uint64_t offset = info.saveWorkSpaceIdx * s1BaseSize_ * topkCountAlign16_ + rowIdx * topkCountAlign16_;
        if (validS2Len <= 0 && !info.isNeedLD) {
            WaitFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
            Duplicate(indicesOutLocal_.ReinterpretCast<int32_t>(), neg, topkCount_);
            SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
            WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
            AscendC::DataCopyPad(indiceOutGm[info.indiceOutOffset + (curS1Idx + rowIdx) * topkCount_], indicesOutLocal_.ReinterpretCast<int32_t>(), copyOutParams);
            SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
            continue;
        } else if (validS2Len <= 0 && info.isNeedLD) {
#if C8_MTP_DUMP_LD_SLOTS
            // 标量诊断（早退分支，tag=E）：isNeedLD 但本核认为窗口为空 → 向 slot 写全 -1。
            // 覆盖核若在此出现即 bug。slot 为全局行主序槽号，配合 bidx 与 python 对照。
            // r=rowIdx（LD 槽内行=全局 s1 行）。DIAG(mixed 2row)：q1/q3 全错，需看 row1
            // 各 slot 写入实况 → 放开到 even-AIV 的 rowIdx<=1。
            if ((rowIdx <= 1U) && (blockId_ % 2U) == 0U) {
                AscendC::printf(
                    "[DUMPW] aiv=%lld bidx=%lld slot=%lld r=%lld vS2=%lld "
                    "s2st=%lld s2end=%lld E\n",
                    (int64_t)blockId_, (int64_t)info.bIdx,
                    (int64_t)info.saveWorkSpaceIdx, (int64_t)rowIdx,
                    (int64_t)validS2Len,
                    (int64_t)info.s2Start, (int64_t)info.s2LoopEnd);
            }
#endif
            WaitFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
            Duplicate(indicesOutLocal_.ReinterpretCast<int32_t>(), neg, topkCountAlign16_);
            Duplicate(scoreOutLocal_, zero, topkCountAlign16_);
            SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
            WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
            // 将每一行S1的Topk结果存放在ldScoreGm和ldIndexGm中
            AscendC::DataCopyPad(ldScoreGm[offset], scoreOutLocal_, ldCopyScoreOutParams);
            AscendC::DataCopyPad(ldIndexGm[offset], indicesOutLocal_.ReinterpretCast<int32_t>(), ldCopyOutParams);
            SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
            continue;
        }

        WaitFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
        WaitFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);

        AscendC::DataCopyPadExtParams<SCORE_T> padParams{true, 0, 0, 0};
        if (validS2Len >= topkCount_) {
            uint32_t s2LoopNum = (validS2Len + trunkLen_ - 1) / trunkLen_;
            if (s2LoopNum == 1) {
                uint32_t validS2LenAlign = QLILdCommon::Align(validS2Len, (int32_t)256);
                Duplicate(mrgValueLocal_[validS2Len / 256 * 256], zero, validS2LenAlign - validS2Len / 256 * 256);
                SetFlag<HardEvent::V_MTE2>(V_MTE2_EVENT);
                WaitFlag<HardEvent::V_MTE2>(V_MTE2_EVENT);
                copyInParams.blockLen = validS2Len * sizeof(SCORE_T); // byte
                AscendC::DataCopyPadExtParams<SCORE_T> padParams{true, 0, 0, 0};
                AscendC::DataCopyPad(mrgValueLocal_, scoreGm[(scoreRowBase_ + (curAivS1Idx + i) * sourceCapacity_) + ldDstOffset], copyInParams, padParams);
                SetFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
                WaitFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
                topkOp_.TopK(mrgValueLocal_, indicesOutLocal_, scoreOutLocal_, validS2LenAlign, 0, 1, info.isNeedLD);
            } else {
                for (uint32_t loopIdx = 0; loopIdx < s2LoopNum; loopIdx++) {
                    if (loopIdx == 0) {
                        copyInParams.blockLen = trunkLen_ * sizeof(SCORE_T); // byte
                        AscendC::DataCopyPad(mrgValueLocal_, scoreGm[(scoreRowBase_ + (curAivS1Idx + i) * sourceCapacity_) + ldDstOffset], copyInParams, padParams);
                        SetFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
                        WaitFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
                        topkOp_.TopK(mrgValueLocal_, indicesOutLocal_, scoreOutLocal_, trunkLen_, loopIdx, s2LoopNum, info.isNeedLD);
                        continue;
                    }
                    SetFlag<HardEvent::V_MTE2>(V_MTE2_EVENT2);
                    WaitFlag<HardEvent::V_MTE2>(V_MTE2_EVENT2);
                    uint32_t validTrunkLen = (loopIdx * trunkLen_ + trunkLen_) > validS2Len ? validS2Len % trunkLen_ : trunkLen_;
                    uint32_t offset = (scoreRowBase_ + (curAivS1Idx + i) * sourceCapacity_) + loopIdx * trunkLen_ + ldDstOffset;
                    AscendC::DataCopy(mrgValueLocal_, scoreOutLocal_, topkCountAlign256_);
                    // topk如果没有对齐到256，则把topkCountAlign256_ - topkCount_部分刷0
                    if (topkCountAlign256_ != topkCount_) {
                        uint64_t mask[1];
                        mask[0] = ~0;
                        mask[0] = mask[0] << (topkCount_ % 64);
                        PipeBarrier<PIPE_V>();
                        // 把topkCount_对齐到64刷0，此处由于duplicate的限制mask[0]刷64个数
                        Duplicate(mrgValueLocal_[topkCount_ / 64 * 64], zero, mask, 1, 1, 0);
                        PipeBarrier<PIPE_V>();
                        // 把topk剩余对齐到256的部分刷0
                        Duplicate(mrgValueLocal_[topkCount_ / 64 * 64 + 64], zero, topkCountAlign256_ - (topkCount_ / 64 * 64 + 64));
                        SetFlag<HardEvent::V_MTE2>(V_MTE2_EVENT3);
                        WaitFlag<HardEvent::V_MTE2>(V_MTE2_EVENT3);
                    }
                    copyInParams.blockLen = validTrunkLen * sizeof(SCORE_T); // byte
                    // TOPK 直方图一次必须计算256，输入处理数据需要和256对齐
                    if ((topkCountAlign256_ + validTrunkLen) % 256 != 0) {
                        Duplicate(mrgValueLocal_[topkCountAlign256_ + validTrunkLen / 256 * 256], zero, QLILdCommon::Align(validTrunkLen, (uint32_t)256) - validTrunkLen / 256 * 256);
                        SetFlag<HardEvent::V_MTE2>(V_MTE2_EVENT);
                        WaitFlag<HardEvent::V_MTE2>(V_MTE2_EVENT);
                    }
                    WaitFlag<HardEvent::V_MTE2>(V_MTE2_EVENT1);
                    AscendC::DataCopyPad(mrgValueLocal_[topkCountAlign256_], scoreGm[offset], copyInParams, padParams);
                    SetFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
                    WaitFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
                    topkOp_.TopK(mrgValueLocal_, indicesOutLocal_, scoreOutLocal_, QLILdCommon::Align(topkCountAlign256_ + validTrunkLen, (uint32_t)256), loopIdx, s2LoopNum, info.isNeedLD);
                    SetFlag<HardEvent::V_MTE2>(V_MTE2_EVENT1);
                }   
            }
        } else {
            AscendC::CreateVecIndex(indicesOutLocal_.ReinterpretCast<int32_t>(), (int32_t)zero, validS2Len);
            // 如果需要LD 则将需要保存Value值
            if (info.isNeedLD) {
                copyInParams.blockLen = validS2Len * sizeof(SCORE_T);
                AscendC::DataCopyPad(scoreOutLocal_, scoreGm[(scoreRowBase_ + (curAivS1Idx + i) * sourceCapacity_) + ldDstOffset], copyInParams, padParams);
                SetFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
                WaitFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
            }
        }
        
        if (!info.isNeedLD) {
            if (validS2Len < topkCount_) {
                uint64_t mask[1];
                mask[0] = ~0;
                mask[0] = mask[0] << (validS2Len % 8); // 将`mask[0]`左移`validS2Len % 8`位 对齐
                PipeBarrier<PIPE_V>();
                // repeatTime  每次读取连续的8个datablock（每个block32Bytes，共256Bytes）
                // dstBlockStride 单次迭代内，矢量目的操作数不同datablock间地址步长。
                Duplicate(indicesOutLocal_.ReinterpretCast<int32_t>()[validS2Len / 8 * 8], neg, mask, 1, 1, 0);
            }
            
            if (validS2Len / 8 * 8 + 64 < topkCount_) {
                PipeBarrier<PIPE_V>();
                Duplicate(indicesOutLocal_.ReinterpretCast<int32_t>()[validS2Len / 8 * 8 + 64], neg, topkCount_ - (validS2Len / 8 * 8 + 64));
            }
            SetFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
            SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
            WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);

            // MTP：当前 s1 行（route）建 payload → classify → publish。PublishRoute
            // 自带头尾 V_MTE3/MTE3_V 屏障，产出 pairs/threshold/missCount/source/slots。
            uint32_t outputRow = static_cast<uint32_t>(info.indiceOutOffset / topkCount_) +
                                 static_cast<uint32_t>(curS1Idx + rowIdx);
            PublishRoute(outputRow, info.cacheRowIdx, info.cacheTokenCount);
        } else {
            PipeBarrier<PIPE_V>();
            AscendC::Adds(indicesOutLocal_.ReinterpretCast<int32_t>(), indicesOutLocal_.ReinterpretCast<int32_t>(), static_cast<int32_t>(curS2StartIdx), topkCountAlign16_);
            if (validS2Len < topkCount_) {
                uint64_t mask[1];
                mask[0] = ~0;
                mask[0] = mask[0] << (validS2Len % scoreAlign);
                PipeBarrier<PIPE_V>();
                Duplicate(scoreOutLocal_[validS2Len / scoreAlign * scoreAlign], zero, mask, 1, 1, 0);
                uint64_t maskI[1];
                maskI[0] = ~0;
                maskI[0] = maskI[0] << (validS2Len % 8);
                PipeBarrier<PIPE_V>();
                Duplicate(indicesOutLocal_.ReinterpretCast<int32_t>()[validS2Len / 8 * 8], neg, maskI, 1, 1, 0);
            }
            if (validS2Len / scoreAlign * scoreAlign + 64 < topkCount_) {
                PipeBarrier<PIPE_V>();
                Duplicate(scoreOutLocal_[validS2Len / scoreAlign * scoreAlign + 64], zero, topkCount_ - (validS2Len / scoreAlign * scoreAlign + 64));
            }
            if (validS2Len / 8 * 8 + 64 < topkCount_) {
                PipeBarrier<PIPE_V>();
                Duplicate(indicesOutLocal_.ReinterpretCast<int32_t>()[validS2Len / 8 * 8 + 64], neg, topkCount_ - (validS2Len / 8 * 8 + 64));
            }
            if (topkCountAlign16_ != topkCount_) {
                uint64_t mask[1];
                mask[0] = ~0;
                mask[0] = mask[0] << (topkCount_ % scoreAlign);
                PipeBarrier<PIPE_V>();
                Duplicate(scoreOutLocal_[topkCount_ / scoreAlign * scoreAlign], zero, mask, 1, 1, 0);
                uint64_t maskIndices[1];
                maskIndices[0] = ~0;
                maskIndices[0] = maskIndices[0] << (topkCount_ % 8);
                PipeBarrier<PIPE_V>();
                Duplicate(indicesOutLocal_.ReinterpretCast<int32_t>()[topkCount_ / 8 * 8], neg, maskIndices, 1, 1, 0);
            }

            SetFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
            SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
            WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
#if C8_MTP_DUMP_LD_SLOTS
            // 标量诊断（主 store 分支，tag=S）：报告该核窗口的写入实况。slot=全局行主序
            // 槽号；r=rowIdx（槽内行=全局 s1 行）。对照 route0/route1 各 slot 是否成对出现；
            // 若 route1 只写了部分 slot → 写侧空窗。若成对齐全仍 q1 错 → merge 读侧。
            if ((rowIdx <= 1U) && (blockId_ % 2U) == 0U) {
                AscendC::printf(
                    "[DUMPW] aiv=%lld bidx=%lld slot=%lld r=%lld vS2=%lld "
                    "s2st=%lld s2end=%lld S\n",
                    (int64_t)blockId_, (int64_t)info.bIdx,
                    (int64_t)info.saveWorkSpaceIdx, (int64_t)rowIdx,
                    (int64_t)validS2Len,
                    (int64_t)info.s2Start, (int64_t)info.s2LoopEnd);
            }
#endif
            // 将每一行S1的Topk结果存放在ldScoreGm和ldIndexGm中
            AscendC::DataCopyPad(ldScoreGm[offset], scoreOutLocal_, ldCopyScoreOutParams);
            AscendC::DataCopyPad(ldIndexGm[offset], indicesOutLocal_.ReinterpretCast<int32_t>(), ldCopyOutParams);
            SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
        }
    }
}

template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::ProcessLD()
{
    AscendC::DataCopyParams copyOutParams;
    copyOutParams.blockCount = 1;
    copyOutParams.blockLen = topkCount_ * sizeof(uint32_t); // bytes
    copyOutParams.srcStride = 0;
    copyOutParams.dstStride = 0;

    uint64_t ldProcessLen = ldInfo_.workspaceNum * topkCountAlign16_;
    if (ldProcessLen <= 0) {
        return;
    }
    uint64_t mrgValueLen = trunkLen_ + topkCountAlign256_;
    uint32_t ldWorkspaceNum = (ldProcessLen > mrgValueLen) ? (trunkLen_ / topkCountAlign16_) : ldInfo_.workspaceNum;
    uint32_t ldProcessNum = CeilDiv(ldInfo_.workspaceNum, ldWorkspaceNum);   // 搬运次数 一次搬运ldworkspaceNum块
    uint32_t ldProcessOffset = 0;

    SCORE_T zero = 0;
    int32_t neg = -1;
    uint32_t zero32 = 0;
    uint32_t scoreAlign = sizeof(SCORE_T) == 4 ? 8 : 16; // int32对应UB对齐数为8，int16需要16
    uint32_t ldProWorkspaceNum = ldInfo_.workspaceNum;
    SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
    for (uint32_t j = 0; j < ldInfo_.mNum; j++) {
        WaitFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
        for (uint32_t i = 0; i < ldProcessNum; i++) {
            // 读取数据段过长
            if (ldProcessNum > 1) {
                ldProWorkspaceNum = (i == ldProcessNum - 1) ? ldInfo_.workspaceNum - i * ldWorkspaceNum : ldWorkspaceNum; // 当前搬运块数
            }
            ldProcessOffset = (i != 0) ? topkCountAlign256_ : 0;
            PipeBarrier<PIPE_V>();
            // 索引全部刷-1 value全部刷0
            Duplicate(mrgValueLocal_[ldProcessOffset], zero, topkCountAlign256_ + trunkLen_ - ldProcessOffset);
            Duplicate(ldIndexLocal_.ReinterpretCast<int32_t>()[ldProcessOffset], neg, topkCountAlign256_ + trunkLen_ - ldProcessOffset);

            AscendC::DataCopyPadExtParams<SCORE_T> scorePadParams{true, 0, 0, 0};
            AscendC::DataCopyExtParams ldScoreParams;
            ldScoreParams.blockCount = ldProWorkspaceNum;
            ldScoreParams.blockLen = topkCountAlign16_ * sizeof(SCORE_T); // bytes
            ldScoreParams.srcStride = (s1BaseSize_ - 1) * topkCountAlign16_ * sizeof(SCORE_T);  // s1BaseSize-1 两个基本块之间的距离（slot 内行间为 T，跨 slot 步长 = s1BaseSize*T）
            ldScoreParams.dstStride = 0;

            AscendC::DataCopyPadExtParams<uint32_t> indexPadParams{true, 0, 0, 0};
            AscendC::DataCopyExtParams ldIndexParams;
            ldIndexParams.blockCount = ldProWorkspaceNum;
            ldIndexParams.blockLen = topkCountAlign16_ * sizeof(uint32_t); // bytes
            ldIndexParams.srcStride = (s1BaseSize_ - 1) * topkCountAlign16_ * sizeof(uint32_t);
            ldIndexParams.dstStride = 0;

            int32_t s2Len = topkCountAlign16_ * ldProWorkspaceNum + ldProcessOffset;
            uint32_t s2LenAlign = QLILdCommon::Align(s2Len, (int32_t)256); // 寄存器需要256对齐
            uint64_t LDGmOffset = ldInfo_.workspaceIdx * s1BaseSize_ * topkCountAlign16_ +
                                  topkCountAlign16_ * (ldInfo_.mStart + j) + i * ldWorkspaceNum * s1BaseSize_ * topkCountAlign16_; // 加入LD处理偏移
            SetFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
            WaitFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
            AscendC::DataCopyPad(mrgValueLocal_[ldProcessOffset], ldScoreGm[LDGmOffset], ldScoreParams, scorePadParams);
            AscendC::DataCopyPad(ldIndexLocal_[ldProcessOffset], ldIndexGm[LDGmOffset].ReinterpretCast<uint32_t>(),
                                 ldIndexParams, indexPadParams);

            SetFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
            WaitFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);

            // 对非对齐索引和值刷0 -1
            if (s2LenAlign != s2Len) {
                uint64_t mask[1];
                mask[0] = ~0;
                mask[0] = mask[0] << (s2Len % 64);
                Duplicate(mrgValueLocal_[s2Len / 64 * 64], zero, mask, 1, 1, 0);
                // 把s2Len对齐到64刷0，此处由于duplicate的限制mask[0]刷64个数
                Duplicate(ldIndexLocal_.ReinterpretCast<int32_t>()[s2Len / 64 * 64], neg, mask, 1, 1, 0);
            }
            if (s2Len / 64 * 64 + 64 < s2LenAlign) {
                PipeBarrier<PIPE_V>();
                Duplicate(mrgValueLocal_[s2Len / 64 * 64 + 64], zero, s2LenAlign - (s2Len / 64 * 64 + 64));
                PipeBarrier<PIPE_V>();
                Duplicate(ldIndexLocal_.ReinterpretCast<int32_t>()[s2Len / 64 * 64 + 64], neg, s2LenAlign - (s2Len / 64 * 64 + 64));
            }
            
            PipeBarrier<PIPE_V>();
            topkOp_.LdTopK(mrgValueLocal_, ldIndexLocal_, indicesOutLocal_, scoreOutLocal_, s2LenAlign, j, ldProcessNum);
            if (topkCountAlign256_ != topkCount_) {
                uint64_t mask[1];
                mask[0] = ~0;
                mask[0] = mask[0] << (topkCount_ % 64);
                PipeBarrier<PIPE_V>();
                Duplicate(scoreOutLocal_[topkCount_ / 64 * 64], zero, mask, 1, 1, 0);
                uint64_t maskIndices[1];
                maskIndices[0] = ~0;
                maskIndices[0] = maskIndices[0] << (topkCount_ % 64);
                PipeBarrier<PIPE_V>();
                Duplicate(indicesOutLocal_.ReinterpretCast<int32_t>()[topkCount_ / 64 * 64], neg, maskIndices, 1, 1, 0);
            }
            if (topkCount_ / 64 * 64 + 64 < topkCountAlign256_) {
                PipeBarrier<PIPE_V>();
                Duplicate(scoreOutLocal_[topkCount_ / 64 * 64 + 64], zero, topkCountAlign256_ - (topkCount_ / 64 * 64 + 64));
                PipeBarrier<PIPE_V>();
                Duplicate(ldIndexLocal_.ReinterpretCast<int32_t>()[topkCount_ / 64 * 64 + 64], neg, topkCountAlign256_ - (topkCount_ / 64 * 64 + 64));
            }
            
            PipeBarrier<PIPE_V>();
            uint32_t copyBytes = topkCountAlign256_; // 32B对齐 topkCountAlign256_
            AscendC::DataCopy(ldIndexLocal_, indicesOutLocal_, copyBytes);
            AscendC::DataCopy(mrgValueLocal_, scoreOutLocal_, copyBytes);
        }
        SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
        WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
        // MTP：LD 归并行 j（route = indiceOutCoreOffset 对应的首行 + mStart + j）
        // 建 payload → classify → publish。cacheRowIdx/cacheTokenCount 由本核负责的
        // 请求（ldInfo_.bIdx）标量读取；indicesOutLocal_ 已含该行的 request-local 源索引。
        uint32_t ldOutputRow =
            static_cast<uint32_t>(ldInfo_.indiceOutCoreOffset / topkCount_) +
            (ldInfo_.mStart + j);
        uint32_t ldCacheRowIdx =
            static_cast<uint32_t>(reqPoolEntriesGm.GetValue(ldInfo_.bIdx));
        uint32_t ldCacheTokenCount =
            static_cast<uint32_t>(cacheTokensGm.GetValue(ldInfo_.bIdx));
#if C8_MTP_DUMP_LD_SLOTS
        // 标量诊断（merge 归并行 j）：上报本 AIV 负责的 LD 任务与当前归并行。
        // 期望 wsIdx/slot 基址、wsNum==该行覆盖核数、out==该行归并后要写出的输出行。
        // r=mStart+j（槽内行=输出 route）。DIAG(mixed 2row)：若 out=r1 的 wsNum<route0 →
        // 归并只覆盖部分 slot → 截断到首窗（[0..2152] 症状）。
        AscendC::printf(
            "[DUMPL] aiv=%lld bidx=%lld wsIdx=%lld wsNum=%lld r=%lld "
            "out=%lld mStart=%lld mNum=%lld\n",
            (int64_t)blockId_, (int64_t)ldInfo_.bIdx,
            (int64_t)ldInfo_.workspaceIdx, (int64_t)ldInfo_.workspaceNum,
            (int64_t)(ldInfo_.mStart + j),
            (int64_t)ldOutputRow, (int64_t)ldInfo_.mStart,
            (int64_t)ldInfo_.mNum);
#endif
#if C8_MTP_DIAG_MC
        // DIAG[LDFILL]（一次性确认探针，验完即删）：core0 route0 在 classify 前标量
        // 统计 LdTopK 归并输出的 2048 个 winner（indicesOutLocal_ 仍是 request-local 源，
        // scoreOutLocal_ 为对应 uint16 分）。判定「0 分补位被 pad(idx=-1) 还是真实 0 分源
        // 占据」：若 numPad≈996 且 numRZ≈0 → kth==0 时前向 EQ 填充吃的是槽尾 pad0，真实
        // 0 分源被 pad 抢占（isolate 短窗 512 真 + 1536 pad 的证据）。期望 numPos≈1052。
        if (blockId_ >= 0 && blockId_ <= 3 && j == 0U) {
            SetFlag<HardEvent::V_S>(V_MTE2_EVENT3);
            WaitFlag<HardEvent::V_S>(V_MTE2_EVENT3);
            int64_t numPad = 0;
            int64_t numPos = 0;
            int64_t numRz = 0;
            int64_t firstPad = -1;
            int64_t firstRz = -1;
            int64_t lastPos = -1;
            int64_t rzCntBeforePad = 0;
            bool seenPad = false;
            for (uint32_t li = 0U; li < topkCount_; ++li) {
                int32_t sidx = (int32_t)indicesOutLocal_.GetValue(li);
                int32_t sval = (int32_t)scoreOutLocal_.GetValue(li);
                if (sidx < 0) {
                    numPad++;
                    if (firstPad < 0) {
                        firstPad = (int64_t)li;
                    }
                    seenPad = true;
                } else if (sval > 0) {
                    numPos++;
                    lastPos = (int64_t)li;
                } else {
                    numRz++;
                    if (firstRz < 0) {
                        firstRz = (int64_t)li;
                    }
                    if (!seenPad) {
                        rzCntBeforePad++;
                    }
                }
            }
            AscendC::printf(
                "[LDFILL] aiv=%lld out=%lld tot=%lld pad=%lld pos=%lld rz=%lld "
                "firstPad=%lld firstRz=%lld lastPos=%lld rzBeforePad=%lld\n",
                (int64_t)blockId_, (int64_t)ldOutputRow,
                (int64_t)(numPad + numPos + numRz), numPad, numPos, numRz,
                firstPad, firstRz, lastPos, rzCntBeforePad);
            SetFlag<HardEvent::S_V>(V_MTE2_EVENT3);
            WaitFlag<HardEvent::S_V>(V_MTE2_EVENT3);
        }
#endif
        PublishRoute(ldOutputRow, ldCacheRowIdx, ldCacheTokenCount);
    }
    WaitFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
}

template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::BuildSortedMissPairs(
    const LocalTensor<int32_t> &classifiedIndex, uint32_t missCount)
{
    if (missCount == 0U) {
        return;
    }

    // AscendC::Sort works on 32-element runs.  Its interleaved
    // (key, source) output is the exact private Stage-2 input, so keep that
    // representation instead of extracting source IDs and packing them
    // again after the global barrier.
    constexpr uint32_t SORT_RUN = 32U;
    constexpr uint32_t SORT_KEY_BASE_BITS = 0x40000000U;
    constexpr uint32_t SORT_TOKEN_MASK = (1U << 18U) - 1U;
    const uint32_t alignedCount =
        (missCount + SORT_RUN - 1U) / SORT_RUN * SORT_RUN;

    // UB reuse after histogram TopK has completed:
    //   topkSharedTmp: keys[2048] + payload[2048] + Sort scratch[4096]
    //   mrgValue:     sorted (key, payload) pairs[2048]
    LocalTensor<float> keyLocal =
        topkSharedTmpLocal_.template ReinterpretCast<float>();
    LocalTensor<uint32_t> payloadLocal = topkSharedTmpLocal_[topkCount_];
    LocalTensor<float> sortTmpLocal =
        topkSharedTmpLocal_[topkCount_ * 2U].template ReinterpretCast<float>();
    LocalTensor<float> sortedPairLocal =
        mrgValueLocal_.template ReinterpretCast<float>();

    DataCopy(payloadLocal,
             classifiedIndex.template ReinterpretCast<uint32_t>(),
             alignedCount);
    PipeBarrier<PIPE_V>();
    Muls(keyLocal.template ReinterpretCast<int32_t>(),
         payloadLocal.template ReinterpretCast<int32_t>(), -1,
         alignedCount);
    PipeBarrier<PIPE_V>();
    Adds(keyLocal.template ReinterpretCast<int32_t>(),
         keyLocal.template ReinterpretCast<int32_t>(),
         static_cast<int32_t>(SORT_KEY_BASE_BITS + SORT_TOKEN_MASK),
         alignedCount);
    PipeBarrier<PIPE_V>();

    // Padded lanes contain scratch data. Give them the smallest key so they
    // sort behind every valid miss. This scalar tail is bounded by 31 elements.
    SetFlag<HardEvent::V_S>(V_MTE2_EVENT3);
    WaitFlag<HardEvent::V_S>(V_MTE2_EVENT3);
    LocalTensor<uint32_t> keyBits = keyLocal.template ReinterpretCast<uint32_t>();
    for (uint32_t index = missCount; index < alignedCount; ++index) {
        keyBits.SetValue(index, 0U);
    }
    SetFlag<HardEvent::S_V>(V_MTE2_EVENT3);
    WaitFlag<HardEvent::S_V>(V_MTE2_EVENT3);

    // Sort is descending. key=base+mask-source therefore produces source IDs
    // in ascending order while retaining the pair layout Stage-2 consumes.
    AscendC::Sort<float, true>(
        sortedPairLocal, keyLocal, payloadLocal, sortTmpLocal,
        alignedCount / SORT_RUN);
    PipeBarrier<PIPE_V>();
}

template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::ExtractSortedMissSourceIds(
    const LocalTensor<int32_t> &output, uint32_t missCount)
{
    if (missCount == 0U) {
        return;
    }
    GatherMaskParams params;
    params.repeatTimes = (missCount * 2U * sizeof(uint32_t) + 255U) / 256U;
    params.src0BlockStride = 1;
    params.src0RepeatStride = 8;
    params.src1RepeatStride = 0;
    uint64_t reserved = 0U;
    // Sort emits interleaved (key, source-id) pairs. Pattern 2 extracts the
    // odd words, preserving the ascending source-ID order.
    GatherMask(
        output.template ReinterpretCast<uint32_t>(),
        mrgValueLocal_.template ReinterpretCast<uint32_t>(),
        static_cast<uint8_t>(2), false, static_cast<uint32_t>(0),
        params, reserved);
    PipeBarrier<PIPE_V>();
}

template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::PublishRoute(
    uint32_t outputRow, uint32_t cacheRowIdx, uint32_t cacheTokenCount)
{
    constexpr uint32_t CLASSIFY_CHUNK = TopkIndexerClassifyVF::CHUNK_SIZE;
    LocalTensor<int32_t> classifiedIndex =
        topkSharedTmpLocal_.template ReinterpretCast<int32_t>();
    LocalTensor<int32_t> slotStageLocal = outInvalidLocal_;

#if C8_MTP_BISECT_SKIP_PUBLISH
    // 调试二分：只把 top-2048 源索引（indicesOutLocal_）写 topkSourceIds，
    // 其余输出（slots/threshold/pairs/missCount）保持毒值。事件配对与完整版一致：
    // V_MTE3 自屏障（vector 完成）→ MTE3 拷贝 → MTE3_V 收尾。
    (void)cacheRowIdx;
    (void)cacheTokenCount;
    (void)CLASSIFY_CHUNK;
    (void)slotStageLocal;
    const uint64_t outOffsetBisect =
        static_cast<uint64_t>(outputRow) * topkCount_;
    AscendC::DataCopyParams outCopyBisect{
        1, static_cast<uint16_t>(topkCount_ * sizeof(int32_t)), 0, 0};
    SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
    WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
    DataCopyPad(indiceOutGm[outOffsetBisect],
                indicesOutLocal_.ReinterpretCast<int32_t>(), outCopyBisect);
    SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
    return;
#endif

#if C8_MTP_DUMP_LD_SLOTS
    // DIAG[CLSP]：发布该行的 AIV 在 classify 前标量采样 winner 源 S（head+tail）与 pool
    // 值，以及尾部 winner 的 score（tailSc）。sTail=-1 或 tailSc=0 → merge 读到空/脏 LD GM
    // 行（route r 读 {s*4+r} 行）；sTail/tailSc 正常但 mc 虚高 → 问题在 slot 编码/classify。
    // 本版放开到 block0..3（mixed 2row 下 route0/1=AIV0、route2/3=AIV1，q1/q3 全错）。
    if (blockId_ <= 3U) {
        const uint64_t clspBase =
            static_cast<uint64_t>(cacheRowIdx) * static_cast<uint64_t>(sourceCapacity_);
        const int32_t clspTail = static_cast<int32_t>(topkCount_) - 1;
        int32_t clspS0 =
            indicesOutLocal_.template ReinterpretCast<int32_t>().GetValue(0);
        int32_t clspSTail =
            indicesOutLocal_.template ReinterpretCast<int32_t>().GetValue(clspTail);
        int64_t clspP0 = (clspS0 >= 0 && (uint64_t)clspS0 < sourceCapacity_)
                             ? (int64_t)cacheSlotsGm.GetValue(clspBase + (uint64_t)clspS0)
                             : -999;
        int64_t clspPTail =
            (clspSTail >= 0 && (uint64_t)clspSTail < sourceCapacity_)
                ? (int64_t)cacheSlotsGm.GetValue(clspBase + (uint64_t)clspSTail)
                : -999;
        AscendC::printf("[CLSP0] aiv=%lld out=%lld ridx=%lld ntok=%lld cap=%lld\n",
                        (int64_t)blockId_, (int64_t)outputRow, (int64_t)cacheRowIdx,
                        (int64_t)cacheTokenCount, (int64_t)sourceCapacity_);
        AscendC::printf("[CLSP1] s0=%lld sTail=%lld tailSc=%lld\n",
                        (int64_t)clspS0, (int64_t)clspSTail,
                        (int64_t)scoreOutLocal_.GetValue(clspTail));
        AscendC::printf("[CLSP2] p0=%lld pTail=%lld\n", clspP0, clspPTail);
    }
#endif

    // 无缓存 token：发布全 -1 与 missCount 0（stage2 PublishInactiveRequest 兜底）。
    if (cacheTokenCount == 0U) {
        Duplicate(classifiedIndex, static_cast<int32_t>(-1), topkCount_);
        Duplicate(slotStageLocal, static_cast<int32_t>(-1), topkCount_);
        LocalTensor<int32_t> missCountLocal =
            topkSharedTmpLocal_[topkCountAlign256_].template ReinterpretCast<int32_t>();
        missCountLocal.SetValue(0, 0);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::S_MTE3>(EVENT_ID1);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID1);
        AscendC::DataCopyParams scalarCopy{
            1, static_cast<uint16_t>(sizeof(int32_t)), 0, 0};
        DataCopyPad(missCountGm[outputRow], missCountLocal, scalarCopy);
        SetFlag<HardEvent::MTE3_S>(EVENT_ID1);
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID1);
        SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
        WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
        AscendC::DataCopyParams outputCopy{
            1, static_cast<uint16_t>(topkCount_ * sizeof(int32_t)), 0, 0};
        const uint64_t outOffset = static_cast<uint64_t>(outputRow) * topkCount_;
        DataCopyPad(indiceOutGm[outOffset], classifiedIndex, outputCopy);
        DataCopyPad(topkSlotsGm[outOffset], slotStageLocal, outputCopy);
        SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
        return;
    }

    // 1) 槽位查找：把 indicesOutLocal_（top-2048 request-local 源索引）原地打成
    //    payload = slot<<18|source。负 cache slot（-1）→ slot<<18=0xFFFC0000 → 天然 MISS。
    // printf-free 根因（2026-09-03 第二版）：仅靠 MTE2_V 事件让 gather 立即读
    // slotChunkBuf 不可靠（ProcessLD i-loop 同事件可用是因为消费前有大量 slack；
    // 此处 DataCopyPad 后紧贴 gather）。printf 的延时掩盖 → 无打印时 slotChunkBuf
    // 读到未装载/撕裂数据 → 大量 resident 源被判 miss（mc 虚高 1217 应 100、published
    // miss 前缀全是垃圾 token）。修复：每 chunk 装载后、gather 前用 PIPE_ALL 硬排空
    //（MTE2 数据必已落 UB），gather 后同样 PIPE_ALL 才放行下一 chunk 覆写 slotChunkBuf。
    // PIPE_ALL 也顺带排空 ProcessLD 尾部对 mrgValueLocal_（slotChunkBuf 别名）的 UB 写回
    // （DataCopy 累加器），消除装载前撕裂。性能非首要（isolate 仅 4 route×4 chunk）。
    const uint64_t cacheBase =
        static_cast<uint64_t>(cacheRowIdx) * static_cast<uint64_t>(sourceCapacity_);
    constexpr uint32_t SLOT_CHUNK = 4096U;
    LocalTensor<int32_t> slotChunkBuf =
        mrgValueLocal_.template ReinterpretCast<int32_t>();
    const uint32_t slotChunkNum = static_cast<uint32_t>(
        (sourceCapacity_ + SLOT_CHUNK - 1U) / SLOT_CHUNK);
    PipeBarrier<PIPE_ALL>();
    for (uint32_t c = 0U; c < slotChunkNum; ++c) {
        const uint32_t chunkBase = c * SLOT_CHUNK;
        const uint32_t chunkLen =
            Min(static_cast<uint32_t>(sourceCapacity_ - chunkBase), SLOT_CHUNK);
        AscendC::DataCopyExtParams slotChunkCopy{1, 0, 0, 0, 0};
        slotChunkCopy.blockLen = chunkLen * sizeof(int32_t);
        AscendC::DataCopyPadExtParams<int32_t> slotChunkPadParams{true, 0, 0, 0};
        DataCopyPad(slotChunkBuf,
                    cacheSlotsGm[cacheBase + chunkBase], slotChunkCopy,
                    slotChunkPadParams);
        PipeBarrier<PIPE_ALL>();
        LdTopkB16Gather::ApplyStreamingSlotChunkVFImpl32(
            (__ubuf__ uint32_t *)indicesOutLocal_.GetPhyAddr(),
            (__ubuf__ uint32_t *)indicesOutLocal_.GetPhyAddr(),
            (__ubuf__ uint32_t *)slotChunkBuf.GetPhyAddr(),
            chunkBase, chunkLen, static_cast<uint16_t>(topkCount_ / 64U));
        PipeBarrier<PIPE_ALL>();
    }

    // 2) classify：miss 源（token id）→ classifiedIndex[0:missCount]；hit slots →
    //    slotStageLocal[missCount:]；随后 sorted miss pairs + 重建 miss/hit 源。
    SetFlag<HardEvent::V_S>(V_MTE2_EVENT3);
    WaitFlag<HardEvent::V_S>(V_MTE2_EVENT3);
    uint16_t kthValue = topkOp_.GetLastKthValue();
    // classify 读已由 slot 循环尾部 PIPE_ALL（~1289）保证排空。此处再一道 PIPE_ALL：
    // ① ClearSpr<AR> 前的全排空，确保先前 POST_MODE_UPDATE 特存链（LD-topk/上一 route
    // classify）的 AR 落账已完成，ClearSpr 稳定归零（曾见残留 AR 使 store 位移、mc 虚高，
    // 无打印态必现、printf 延时掩盖）；② 保证 kthValue 标量读在 classify 覆写
    // topkSharedTmpLocal_（nkValueLocal 别名区）前完成。真根因在 slot 装载（见上注释），
    // 两道 PIPE_ALL 均为正确性兜底，代价可忽略。
    PipeBarrier<PIPE_ALL>();
    // miss 计数读 AR 绝对值。SqueezeIndexerMissTokenIds 内部 ClearSpr<AR> 后再经
    // POST_MODE_UPDATE 累加 miss 字节数：ClearSpr 把同核此前（LD-topk/上一 route
    // classify）的 AR 残留清零 → store 落在 classifiedIndex +0，且 AR 终值恰 = 本 VF
    // 存储字节数。曾为免 ClearSpr 改「基线差分」(VF 前读 baseAr、VF 后读 totalAr)：
    // 计数(差分)虽对，但 store 链以进链 AR 现值作写偏移 → miss token 全写偏到
    // classifiedIndex 之后，[0:mc] 留宿值浮点垃圾（CQ6 实测）。故恢复 ClearSpr +
    // 绝对值读（与 union.h CompactSafeVictims 同型，已验证可靠）。
    TopkIndexerClassifyVF::SqueezeIndexerMissTokenIds(
        (__ubuf__ uint32_t *)classifiedIndex.GetPhyAddr(),
        (__ubuf__ uint32_t *)indicesOutLocal_.GetPhyAddr(),
        topkCount_ / CLASSIFY_CHUNK);
    PipeBarrier<PIPE_ALL>();
    const int64_t totalAr =
        AscendC::GetSpr<AscendC::SpecialPurposeReg::AR>();
    uint32_t currentMissCount = static_cast<uint32_t>(
        totalAr / sizeof(uint32_t));
    PipeBarrier<PIPE_V>();
#if C8_MTP_DIAG_MC
    if (blockId_ <= 3U) {
    AscendC::printf("[MCDIAG] aiv=%lld out=%lld mc=%lld\n",
                    (int64_t)blockId_, (int64_t)outputRow,
                    (int64_t)currentMissCount);
    // DIAG[CQ6]：core0（route0）在 BuildSortedMissPairs 前 dump 原始 miss token
    // （classifiedIndex[0:mc]，winner-lane 序，未排序）。修复（ClearSpr+绝对值）后
    // 应看到正确小值（≈14/74/...）；若仍是宿值浮点垃圾 → store 仍未落在 dst+0。
    if (blockId_ == 0 && currentMissCount > 0U) {
        SetFlag<HardEvent::V_S>(V_MTE2_EVENT3);
        WaitFlag<HardEvent::V_S>(V_MTE2_EVENT3);
        const uint32_t cq6n = (currentMissCount > 8U) ? 8U : currentMissCount;
        int64_t cq6[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        for (uint32_t k = 0U; k < cq6n; ++k) {
            cq6[k] = (int64_t)(int32_t)classifiedIndex.GetValue(k);
        }
        AscendC::printf("[CQ6] preSortMiss0..7=%lld/%lld/%lld/%lld/%lld/%lld/%lld/%lld\n",
                        cq6[0], cq6[1], cq6[2], cq6[3],
                        cq6[4], cq6[5], cq6[6], cq6[7]);
        SetFlag<HardEvent::S_V>(V_MTE2_EVENT3);
        WaitFlag<HardEvent::S_V>(V_MTE2_EVENT3);
    }
    }  // if (blockId_ <= 3U)
#endif
    Duplicate(slotStageLocal, static_cast<int32_t>(-1), topkCount_);
    PipeBarrier<PIPE_V>();
    TopkIndexerClassifyVF::SqueezeIndexerHitSlots(
        (__ubuf__ uint32_t *)slotStageLocal[currentMissCount].GetPhyAddr(),
        (__ubuf__ uint32_t *)indicesOutLocal_.GetPhyAddr(),
        topkCount_ / CLASSIFY_CHUNK);
    PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_S>(V_MTE2_EVENT3);
    WaitFlag<HardEvent::V_S>(V_MTE2_EVENT3);
    BuildSortedMissPairs(classifiedIndex, currentMissCount);
    ExtractSortedMissSourceIds(classifiedIndex, currentMissCount);
    TopkIndexerClassifyVF::SqueezeIndexerHitTokenIds(
        (__ubuf__ uint32_t *)classifiedIndex[currentMissCount].GetPhyAddr(),
        (__ubuf__ uint32_t *)indicesOutLocal_.GetPhyAddr(),
        topkCount_ / CLASSIFY_CHUNK);
    PipeBarrier<PIPE_V>();
#if C8_MTP_DIAG_MC
    // DIAG[CQ]：AIV0（isolate route0）标量复核 classify：
    //   missScalar = 逐 winner 直接读 cacheSlotsGm[cacheBase+src]，槽位<0 的真实 miss 数；
    //   与 VF 计数 currentMissCount 对比 → 判 classify 计数是否忠实。
    //   invalid = winner src >= sourceCapacity_ 数 → 判 LD winner 是否 request-local。
    //   CQ4/CQ5 = 发布在即的 topkSourceIds 首 6（sorted miss）与 miss 后缀首 6（hit）。
    if (blockId_ <= 3U) {
        SetFlag<HardEvent::V_S>(V_MTE2_EVENT3);
        WaitFlag<HardEvent::V_S>(V_MTE2_EVENT3);
        uint32_t missScalar = 0U;
        uint32_t invalidSrc = 0U;
        uint32_t srcMin = 0xFFFFFFFFU;
        uint32_t srcMax = 0U;
        const uint32_t SAMPLE_N = 6U;
        const uint32_t sampleIdx[6] = {0U, 1U, 64U, 1024U, 2045U, 2047U};
        int64_t sampleS[6] = {0, 0, 0, 0, 0, 0};
        int64_t sampleG[6] = {0, 0, 0, 0, 0, 0};
        for (uint32_t i = 0U; i < topkCount_; ++i) {
            uint32_t payload = indicesOutLocal_.GetValue(i);
            uint32_t src = payload & 0x3ffffU;
            int64_t gm = 0;
            if (src < sourceCapacity_) {
                gm = (int64_t)cacheSlotsGm.GetValue(cacheBase + (uint64_t)src);
                if (gm < 0) {
                    missScalar++;
                }
            } else {
                invalidSrc++;
                gm = -777;
            }
            if (src < srcMin) {
                srcMin = src;
            }
            if (src > srcMax) {
                srcMax = src;
            }
            for (uint32_t k = 0U; k < SAMPLE_N; ++k) {
                if (i == sampleIdx[k]) {
                    sampleS[k] = (int64_t)src;
                    sampleG[k] = gm;
                }
            }
        }
        AscendC::printf("[CQ0] aiv=%lld out=%lld cRow=%lld cTok=%lld cap=%lld mc=%lld\n",
                        (int64_t)blockId_, (int64_t)outputRow, (int64_t)cacheRowIdx,
                        (int64_t)cacheTokenCount, (int64_t)sourceCapacity_,
                        (int64_t)currentMissCount);
        AscendC::printf("[CQ1] missScalar=%lld invalid=%lld sMin=%lld sMax=%lld\n",
                        (int64_t)missScalar, (int64_t)invalidSrc, (int64_t)srcMin,
                        (int64_t)srcMax);
        AscendC::printf("[CQ2] src i0/1/64/1024/2045/2047 = %lld/%lld/%lld/%lld/%lld/%lld\n",
                        sampleS[0], sampleS[1], sampleS[2], sampleS[3], sampleS[4],
                        sampleS[5]);
        AscendC::printf("[CQ3] gmSlot same-idx = %lld/%lld/%lld/%lld/%lld/%lld\n",
                        sampleG[0], sampleG[1], sampleG[2], sampleG[3], sampleG[4],
                        sampleG[5]);
        SetFlag<HardEvent::V_S>(V_MTE2_EVENT3);
        WaitFlag<HardEvent::V_S>(V_MTE2_EVENT3);
        int64_t cf[6];
        int64_t cs[6];
        for (int32_t k = 0; k < 6; ++k) {
            cf[k] = (int64_t)(int32_t)classifiedIndex.GetValue(k);
            cs[k] = (int64_t)(int32_t)classifiedIndex.GetValue(currentMissCount + k);
        }
        AscendC::printf("[CQ4] srcPrefix0..5=%lld/%lld/%lld/%lld/%lld/%lld\n",
                        cf[0], cf[1], cf[2], cf[3], cf[4], cf[5]);
        AscendC::printf("[CQ5] hitSuffix(mc..)0..5=%lld/%lld/%lld/%lld/%lld/%lld\n",
                        cs[0], cs[1], cs[2], cs[3], cs[4], cs[5]);
    }
#endif

    // 3) threshold（每 route 16 uint16）与 missCount（标量）。阈值必须在
    //    BuildSortedMissPairs clobber topkSharedTmpLocal_ 前取出（nkValue 在该区）。
    constexpr uint32_t MTP_THRESHOLD_STRIDE = 16U;
    Duplicate(scoreOutLocal_, kthValue, MTP_THRESHOLD_STRIDE);
    PipeBarrier<PIPE_V>();
    LocalTensor<int32_t> routeMissCountLocal =
        topkSharedTmpLocal_[topkCountAlign256_ * 4U].template ReinterpretCast<int32_t>();
    routeMissCountLocal.SetValue(0, static_cast<int32_t>(currentMissCount));
    SetFlag<HardEvent::S_MTE3>(EVENT_ID1);
    WaitFlag<HardEvent::S_MTE3>(EVENT_ID1);
    AscendC::DataCopyParams scalarCopy{
        1, static_cast<uint16_t>(sizeof(int32_t)), 0, 0};
    DataCopyPad(missCountGm[outputRow], routeMissCountLocal, scalarCopy);
    SetFlag<HardEvent::MTE3_S>(EVENT_ID1);
    WaitFlag<HardEvent::MTE3_S>(EVENT_ID1);
    SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
    WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);

    // 4) GM 发布：pairs（(key,source) 交织，missCount*2 int32）、threshold、
    //    topkSourceIds（indiceOutGm 绑定该区）、topkSlots。
    const uint64_t outOffset = static_cast<uint64_t>(outputRow) * topkCount_;
    if (currentMissCount > 0U) {
        AscendC::DataCopyParams pairOutputCopy{
            1, static_cast<uint16_t>(currentMissCount * 2U * sizeof(int32_t)),
            0, 0};
        const uint64_t pairOutOffset =
            static_cast<uint64_t>(outputRow) * topkCount_ * 2U;
        DataCopyPad(routePairsGm[pairOutOffset],
                    mrgValueLocal_.template ReinterpretCast<int32_t>(),
                    pairOutputCopy);
    }
    DataCopy(thresholdGm[static_cast<uint64_t>(outputRow) * MTP_THRESHOLD_STRIDE],
             scoreOutLocal_, MTP_THRESHOLD_STRIDE);
    AscendC::DataCopyParams outputCopy{
        1, static_cast<uint16_t>(topkCount_ * sizeof(int32_t)), 0, 0};
    DataCopyPad(indiceOutGm[outOffset], classifiedIndex, outputCopy);
    DataCopyPad(topkSlotsGm[outOffset], slotStageLocal, outputCopy);
    SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
}
}  // namespace QLILdKernel
#endif