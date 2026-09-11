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
 * \file quant_lightning_indexer_kernel.h
 * \brief
 */

#ifndef quant_lightning_indexer_KERNEL_H
#define quant_lightning_indexer_KERNEL_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "quant_lightning_indexer_common.h"
#include "quant_lightning_indexer_service_vector.h"
#include "quant_lightning_indexer_service_cube.h"

namespace QLILdKernel {
using namespace QLILdCommon;
using namespace matmul;
using AscendC::CacheMode;
using AscendC::CrossCoreSetFlag;
using AscendC::CrossCoreWaitFlag;

// 由于S2循环前，RunInfo还没有赋值，使用TempLoopInfo临时存放B、N、S1轴相关的信息；同时减少重复计算
struct TempLoopInfo {
    uint32_t bN2Idx = 0;
    uint32_t bIdx = 0U;
    uint32_t n2Idx = 0U;
    uint32_t gS1Idx = 0U;
    uint32_t gS1LoopEnd = 0U;   // gS1方向循环的结束Idx
    uint32_t s2LoopEnd = 0U;    // S2方向循环的结束Idx
    uint32_t actS1Size = 1ULL;  // 当前Batch循环处理的S1轴的实际大小
    uint32_t actS2Size = 0ULL;
    uint32_t actS2SizeOrig = 0ULL;//压缩前s2
    bool curActSeqLenIsZero = false;
    bool needDealActS1LessThanS1 = false;  // S1的实际长度小于shape的S1长度时，是否需要清理输出
    uint32_t actMBaseSize = 0U;            // m轴(gS1)方向实际大小
    uint32_t mBasicSizeTail = 0U;          // gS1方向循环的尾基本块大小
    uint32_t s2BasicSizeTail = 0U;         // S2方向循环的尾基本块大小
    bool isNeedLD = false;     // 该基本块是否需要LD
};

template <typename QLIT>
class QLIPreload {
public:
    __aicore__ inline QLIPreload(){};
    __aicore__ inline void Init(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *weights,
                                __gm__ uint8_t *queryScale, __gm__ uint8_t *keyScale, __gm__ uint8_t *actualSeqLengthsQ,
                                __gm__ uint8_t *actualSeqLengthsK, __gm__ uint8_t *blockTable,
                                __gm__ uint8_t *sparseIndices,
                                __gm__ uint8_t *routePairRows, __gm__ uint8_t *topkSlots,
                                __gm__ uint8_t *missCount, __gm__ uint8_t *threshold,
                                __gm__ uint8_t *cacheSlots, __gm__ uint8_t *reqPoolEntries,
                                __gm__ uint8_t *cacheTokens, __gm__ uint8_t *workspace,
                                __gm__ uint8_t *ldWorkspace, const QLITilingData *__restrict tiling, TPipe *tPipe);
    __aicore__ inline void Process();

    // =================================类型定义区=================================
    using Q_T = typename QLIT::queryType;
    using K_T = typename QLIT::keyType;
    using OUT_T = typename QLIT::outputType;
    static constexpr bool PAGE_ATTENTION = QLIT::pageAttention;
    static constexpr LI_LAYOUT Q_LAYOUT_T = QLIT::layout;
    static constexpr LI_LAYOUT K_LAYOUT_T = QLIT::keyLayout;

    using SCORE_T = typename QLIT::scoreType;

    QLIMatmul<QLIT> matmulService;
    QLIVector<QLIT> vectorService;

    // =================================常量区=================================
    static constexpr uint32_t SYNC_C1_V1_FLAG = 4;
    static constexpr uint32_t SYNC_V1_C1_FLAG = 5;

    static constexpr uint32_t M_BASE_SIZE = 256;
    // s1BaseSize 由 InitTilingData 按 host 公式 ceil(256/gSize) 决定（=8 at gSize=32），
    // 不要用 S1_BASE_SIZE=4 覆盖：mBaseSize 会降到 128，bufUB_ 减半到 64KB，而 decode 块的
    // fixpipe 写入足印含不随 mBaseSize 缩小的常数项 (mSize-1)*dstStride≈16KB，净溢出~8K元素
    // 写穿后续 UB 缓冲 → bs=1 topk 全错。decode 的 actMBaseSize 封顶 32，与 s1BaseSize 无关。
    static constexpr uint32_t S2_BASE_SIZE = 128;
    static constexpr uint32_t HEAD_DIM = 128;
    static constexpr uint32_t K_HEAD_NUM = 1;
    static constexpr uint32_t GM_ALIGN_BYTES = 512;

    static constexpr int64_t LD_PREFETCH_LEN = 2;
    // for workspace double
    static constexpr uint32_t WS_DOUBLE = 2;

    // MIX_AIC_1_2 下 AIC=24、AIV=48（编译期常量，与 GetBlockNum() 自洽）
    static constexpr uint32_t AIC_CORE_NUM = 24;
    static constexpr uint32_t AIV_CORE_NUM = 48;

protected:
    TPipe *pipe = nullptr;

    // offset
    uint64_t queryCoreOffset = 0ULL;
    uint64_t keyCoreOffset = 0ULL;
    uint64_t keyScaleCoreOffset = 0ULL;
    uint64_t weightsCoreOffset = 0ULL;
    uint64_t indiceOutCoreOffset = 0ULL;
    bool isUsedCoreEqZero = false;
    // ================================Global Buffer区=================================
    GlobalTensor<Q_T> queryGm;
    GlobalTensor<K_T> keyGm;
    GlobalTensor<bfloat16_t> weightsGm;  // MTP 输入为 bf16
    GlobalTensor<float> qScaleGm;
    GlobalTensor<float> kScaleGm;

    GlobalTensor<int32_t> indiceOutGm;  // 绑定 topkSourceIds 区，PublishRoute 复用为 source 输出
    GlobalTensor<int32_t> blockTableGm;

    GlobalTensor<uint32_t> actualSeqLengthsGmQ;  // query 前缀和（每 request 恒 4）
    GlobalTensor<uint32_t> actualSeqLengthsGm;   // MTP：candidateLens（actS2SizeOrig）

    // IsActiveRequest（ComputeSplitInfo，AIC/AIV 都要读）用
    GlobalTensor<int32_t> cacheTokensGm;
    GlobalTensor<int32_t> reqPoolEntriesGm;
    // AIV 侧 classify+publish 输出/查询张量
    GlobalTensor<int32_t> routePairsGm;
    GlobalTensor<int32_t> topkSlotsGm;
    GlobalTensor<int32_t> missCountGm;
    GlobalTensor<uint16_t> thresholdGm;
    GlobalTensor<int32_t> cacheSlotsGm;

    // ================================类成员变量====================================
    // aic、aiv核信息
    uint32_t tmpBlockIdx = 0U;
    uint32_t aiCoreIdx = 0U;
    uint32_t usedCoreNum = 0U;

    QLILdCommon::ConstInfo constInfo{};
    TempLoopInfo tempLoopInfo{};
    QLILdCommon::SplitCoreInfo splitCoreInfo{};
    QLILdCommon::LdSplitCoreInfo ldInfo{};

    // ================================Init functions==================================
    __aicore__ inline void InitTilingData(const QLITilingData *__restrict tilingData);
    __aicore__ inline void InitBuffers();
    __aicore__ inline void InitActualSeqLen(__gm__ uint8_t *actualSeqLengthsQ, __gm__ uint8_t *actualSeqLengthsK);
    // ================================Split Core================================
    // on-device 分核：复刻 AICPU metadata 的 BalanceSchedule + SplitFD，融合进内核。
    // 所有核（AIC/AIV）跑同一确定性算法，各自取自己的切片，无需跨核同步、不新增输入。
    __aicore__ inline void ComputeSplitInfo(uint32_t cubeCoreIdx, uint32_t vecCoreIdx);
    __aicore__ inline uint32_t GetS2BaseBlockNumOnMask(uint32_t s1gIdx, uint32_t actS1Size, uint32_t actS2SizeOrig);
    // (bN2, gS1) 行游标：按 bN2 主序/gS1 次序列举所有非空行
    __aicore__ inline void InitRowCursor(uint32_t &bN2, uint32_t &gS1);
    __aicore__ inline uint32_t RowCursorInfo(uint32_t bN2, uint32_t gS1, uint32_t &w, uint32_t &tailM);
    __aicore__ inline bool AdvanceRowCursor(uint32_t &bN2, uint32_t &gS1);
    // ================================Process functions================================
    __aicore__ inline void ProcessMain();
    __aicore__ inline void ProcessBaseBlock(uint32_t loop, uint64_t s2LoopIdx,
                                            QLILdCommon::RunInfo runInfo, uint32_t qScaleLoop,
                                            uint32_t kScaleLoop);
    __aicore__ inline void ProcessDecode();
    __aicore__ inline void ProcessInvalid();
    // ================================Params Calc=====================================
    __aicore__ inline void CalcGS1LoopParams(uint32_t bN2Idx);
    __aicore__ inline void GetBN2Idx(uint32_t bN2Idx);
    __aicore__ inline uint32_t GetActualSeqLen(uint32_t bIdx, uint32_t actualLenDims, bool isAccumSeq,
                                               GlobalTensor<uint32_t> &actualSeqLengthsGm, uint32_t defaultSeqLen);
    __aicore__ inline uint32_t GetActualSeqLenKey(uint32_t bIdx, uint32_t actualLenDims, bool isAccumSeq,
                                            GlobalTensor<uint32_t> &actualSeqLengthsGm, uint32_t defaultSeqLen, uint32_t cmpRatio);
    __aicore__ inline void GetS1S2ActualSeqLen(uint32_t bIdx, uint32_t &actS1Size, uint32_t &actS2Size, uint32_t &actS2SizeOrig);
    __aicore__ inline bool IsActiveRequest(uint32_t bIdx);
    __aicore__ inline void CalcS2LoopParams(uint32_t bN2LoopIdx, uint32_t gS1LoopIdx);
    __aicore__ inline void CalcRunInfo(uint32_t loop, uint32_t s2LoopIdx, QLILdCommon::RunInfo &runInfo,
                                       uint32_t qScaleLoop, uint32_t kScaleLoop);
    __aicore__ inline void DealActSeqLenIsZero(uint32_t bIdx, uint32_t n2Idx, uint32_t s1Start);
};

template <typename QLIT>
__aicore__ inline void QLIPreload<QLIT>::InitTilingData(const QLITilingData *__restrict tilingData)
{
    usedCoreNum = tilingData->usedCoreNum;
    constInfo.batchSize = tilingData->batchSize;  // requests（非 packed T）
    constInfo.qHeadNum = constInfo.gSize = tilingData->indexHeads;
    constInfo.kSeqSize = tilingData->maxCandidateLen;
    // MTP-4 绑定：每 request 固定 4 条 route，kHeadNum=1、qSeqSize=4。
    // bN2Idx=bIdx（每 request 一个 (bN2,gS1) 行，gS1SplitNum=1）。
    constInfo.qSeqSize = 4;
    constInfo.attenMaskFlag = true;
    constInfo.kCacheBlockSize = 128;
    constInfo.maxBlockNumPerBatch = tilingData->maxBlockNumPerBatch;
    constInfo.sparseCount = 2048;
    constInfo.cmpRatio = 1;
    constInfo.batchSupperFlag = false;
    constInfo.keyStride0 = tilingData->keyStride;
    constInfo.keyDequantScaleStride0 = tilingData->scaleStride;
    constInfo.outputLayout = Q_LAYOUT_T;  // 输出和输入形状一致
    if (Q_LAYOUT_T == LI_LAYOUT::TND) {
        constInfo.isAccumSeqS1 = true;
    }
    if (K_LAYOUT_T == LI_LAYOUT::TND) {
        constInfo.isAccumSeqS2 = true;
    }

    constInfo.kHeadNum = K_HEAD_NUM;
    constInfo.headDim = HEAD_DIM;

    constInfo.mBaseSize = M_BASE_SIZE;
    constInfo.s2BaseSize = S2_BASE_SIZE;
    constInfo.s1BaseSize = (constInfo.mBaseSize + constInfo.gSize - 1) / constInfo.gSize;
    // workspace 步长固定用 host 分配值，不随 Branch 2 的对齐覆盖变化
    constInfo.s1BaseSizeWs = constInfo.s1BaseSize;
    // MTP classify+publish 校验需要
    constInfo.poolSize = tilingData->poolSize;
    constInfo.cacheSlotsSize = tilingData->sourceCapacity;
    // 实验：大 batch 走官方 9.1.0 整行贪婪（Branch 2：weight/qScale 每 S1 组仅首块加载一次
    // + kScale 每 16 块批量），小 batch 保持 Branch 1 逐块加载（块池切分已验证路径）。
    // 阈值 >16 为初值，A5 全 sweep 正确性 + 性能定稿。
    constInfo.isWholeRowGreedy = (constInfo.batchSize > 16);
}

template <typename QLIT>
__aicore__ inline void QLIPreload<QLIT>::InitBuffers()
{
    if ASCEND_IS_AIV {
        vectorService.InitBuffers(pipe);
    } else {
        matmulService.InitBuffers(pipe);
    }
}

template <typename QLIT>
__aicore__ inline void QLIPreload<QLIT>::InitActualSeqLen(__gm__ uint8_t *actualSeqLengthsQ,
                                                          __gm__ uint8_t *actualSeqLengthsK)
{
    if (actualSeqLengthsQ == nullptr) {
        constInfo.actualLenQDims = 0;
    } else {
        constInfo.actualLenQDims = (constInfo.batchSupperFlag) ? constInfo.batchSize + 1 : constInfo.batchSize;
        actualSeqLengthsGmQ.SetGlobalBuffer((__gm__ uint32_t *)actualSeqLengthsQ, constInfo.actualLenQDims);
    }
    if (actualSeqLengthsK == nullptr) {
        constInfo.actualLenDims = 0;
    } else {
        constInfo.actualLenDims = constInfo.batchSize;
        actualSeqLengthsGm.SetGlobalBuffer((__gm__ uint32_t *)actualSeqLengthsK, constInfo.actualLenDims);
    }
}

template <typename QLIT>
__aicore__ inline uint32_t QLIPreload<QLIT>::GetActualSeqLen(uint32_t bIdx, uint32_t actualLenDims, bool isAccumSeq,
                                                             GlobalTensor<uint32_t> &actualSeqLengthsGm,
                                                             uint32_t defaultSeqLen)
{
    bIdx = (constInfo.batchSupperFlag)? bIdx + 1 : bIdx; // 如果为B+1情况，则向后移动一位
    if (actualLenDims == 0) {
        return defaultSeqLen;
    } else if (isAccumSeq && bIdx > 0) {
        return actualSeqLengthsGm.GetValue(bIdx) - actualSeqLengthsGm.GetValue(bIdx - 1);
    } else {
        return actualSeqLengthsGm.GetValue(bIdx);
    }
}

template <typename QLIT>
__aicore__ inline uint32_t QLIPreload<QLIT>::GetActualSeqLenKey(uint32_t bIdx, uint32_t actualLenDims, bool isAccumSeq,
                                                             GlobalTensor<uint32_t> &actualSeqLengthsGm,
                                                             uint32_t defaultSeqLen, uint32_t cmpRatio)
{
    if (actualLenDims == 0) {
        return defaultSeqLen * cmpRatio;
    } else if (isAccumSeq && bIdx > 0) {
        return actualSeqLengthsGm.GetValue(bIdx) - actualSeqLengthsGm.GetValue(bIdx - 1);
    } else {
        return actualSeqLengthsGm.GetValue(bIdx);
    }
}

template <typename QLIT>
__aicore__ inline void QLIPreload<QLIT>::GetS1S2ActualSeqLen(uint32_t bIdx, uint32_t &actS1Size, uint32_t &actS2Size, uint32_t &actS2SizeOrig)
{
    actS1Size = GetActualSeqLen(bIdx, constInfo.actualLenQDims, constInfo.isAccumSeqS1, actualSeqLengthsGmQ,
                                constInfo.qSeqSize);
    actS2SizeOrig =
        GetActualSeqLenKey(bIdx, constInfo.actualLenDims, constInfo.isAccumSeqS2, actualSeqLengthsGm, constInfo.kSeqSize, constInfo.cmpRatio); // 压缩前的actS2Size
    actS2Size = actS2SizeOrig / constInfo.cmpRatio;   // 真实使用的压缩后S2长度
    // MTP：非活跃 request（queryEnd!=（b+1)*4 / budget 越界 / candidate 非法 /
    // poolRow 越界）整行跳过，输出由 stage2 兜底
    if (!IsActiveRequest(bIdx)) {
        actS1Size = 0;
        actS2Size = 0;
        actS2SizeOrig = 0;
    }
}

template <typename QLIT>
__aicore__ inline bool QLIPreload<QLIT>::IsActiveRequest(uint32_t bIdx)
{
    const int32_t expectedQueryEnd = static_cast<int32_t>((bIdx + 1U) * constInfo.qSeqSize);
    const int32_t queryEnd = static_cast<int32_t>(actualSeqLengthsGmQ.GetValue(bIdx));
    const int32_t budget = static_cast<int32_t>(cacheTokensGm.GetValue(bIdx));
    const int32_t candidate = static_cast<int32_t>(actualSeqLengthsGm.GetValue(bIdx));  // candidateLens
    const int32_t poolRow = static_cast<int32_t>(reqPoolEntriesGm.GetValue(bIdx));
    return queryEnd == expectedQueryEnd && budget >= 8192 &&
        budget <= 16256 && candidate >= budget &&
        candidate <= static_cast<int32_t>(constInfo.kSeqSize) &&
        candidate % static_cast<int32_t>(constInfo.s2BaseSize) == 0 &&
        poolRow >= 0 && poolRow < static_cast<int32_t>(constInfo.poolSize);
}

template <typename QLIT>
__aicore__ inline uint32_t QLIPreload<QLIT>::GetS2BaseBlockNumOnMask(uint32_t s1gIdx, uint32_t actS1Size,
                                                                     uint32_t actS2SizeOrig)
{
    if (actS2SizeOrig / constInfo.cmpRatio == 0) {
        return 0;
    }
    uint32_t s1Offset = constInfo.s1BaseSize * s1gIdx;
    int32_t validS2LenBase = static_cast<int32_t>(actS2SizeOrig) - static_cast<int32_t>(actS1Size);    // 压缩前的validS2LenBase
    int32_t validS2Len = (static_cast<int32_t>(s1Offset) + validS2LenBase + static_cast<int32_t>(constInfo.s1BaseSize)) / static_cast<int32_t>(constInfo.cmpRatio);  
    validS2Len = Min(validS2Len, static_cast<int32_t>(actS2SizeOrig) / constInfo.cmpRatio);
    validS2Len = Max(validS2Len, 1);
    return (validS2Len + constInfo.s2BaseSize - 1) / constInfo.s2BaseSize;
}

template <typename QLIT>
__aicore__ inline void QLIPreload<QLIT>::InitRowCursor(uint32_t &bN2, uint32_t &gS1)
{
    bN2 = 0;
    gS1 = 0;
    uint32_t totalBN2 = (uint32_t)(constInfo.batchSize * constInfo.kHeadNum);
    while (bN2 < totalBN2) {
        uint32_t bIdx = bN2 / constInfo.kHeadNum;
        uint32_t actS1Size, actS2Size, actS2SizeOrig;
        GetS1S2ActualSeqLen(bIdx, actS1Size, actS2Size, actS2SizeOrig);
        if (actS1Size != 0 && actS2Size != 0) {
            return;
        }
        bN2++;
    }
    gS1 = 0;
}

template <typename QLIT>
__aicore__ inline uint32_t QLIPreload<QLIT>::RowCursorInfo(uint32_t bN2, uint32_t gS1, uint32_t &w, uint32_t &tailM)
{
    uint32_t bIdx = bN2 / constInfo.kHeadNum;
    uint32_t actS1Size, actS2Size, actS2SizeOrig;
    GetS1S2ActualSeqLen(bIdx, actS1Size, actS2Size, actS2SizeOrig);
    if (constInfo.attenMaskFlag) {
        w = GetS2BaseBlockNumOnMask(gS1, actS1Size, actS2SizeOrig);
    } else {
        w = (actS2Size + constInfo.s2BaseSize - 1) / constInfo.s2BaseSize;
    }
    // 该 gS1 行在 S1 轴的行数（与内核 curS1ProcNum 一致），LD 归约的 m 轴大小
    uint32_t gS1Num = (actS1Size * constInfo.gSize + constInfo.mBaseSize - 1) / constInfo.mBaseSize;
    uint32_t lastRow = gS1Num - 1;
    uint32_t mSize = (gS1 == lastRow) ? (actS1Size * constInfo.gSize - gS1 * constInfo.mBaseSize)
                                      : constInfo.mBaseSize;
    tailM = mSize / constInfo.gSize;
    return w;
}

template <typename QLIT>
__aicore__ inline bool QLIPreload<QLIT>::AdvanceRowCursor(uint32_t &bN2, uint32_t &gS1)
{
    gS1++;
    uint32_t totalBN2 = (uint32_t)(constInfo.batchSize * constInfo.kHeadNum);
    while (bN2 < totalBN2) {
        uint32_t bIdx = bN2 / constInfo.kHeadNum;
        uint32_t actS1Size, actS2Size, actS2SizeOrig;
        GetS1S2ActualSeqLen(bIdx, actS1Size, actS2Size, actS2SizeOrig);
        uint32_t gS1Num = (actS1Size == 0 || actS2Size == 0)
                              ? 0
                              : (actS1Size * constInfo.gSize + constInfo.mBaseSize - 1) / constInfo.mBaseSize;
        if (gS1 < gS1Num) {
            return true;
        }
        bN2++;
        gS1 = 0;
    }
    return false;
}

template <typename QLIT>
__aicore__ inline void QLIPreload<QLIT>::ComputeSplitInfo(uint32_t cubeCoreIdx, uint32_t vecCoreIdx)
{
    // s1BaseSize/mBaseSize 保持 InitTilingData 的 host 值（s1BaseSize=ceil(256/gSize)，mBaseSize=256）。
    // 曾试验 s1BaseSize=4→mBaseSize=128，bufUB_/resMm1Buf_ 减半，而 decode 块 fixpipe 足印
    // (mSize=64×N=128) 含常数项 ~16KB 不随 mBaseSize 缩小 → 溢出写穿 UB → bs=1 topk 全错。
    // 块池切分本身不依赖该覆盖（slot 步长走 constInfo.s1BaseSize=8），故保持 256。

    // ===== Pass A：统计总块数（行主序：bN2 主序/gS1 次序，行区间连续铺满 [0,totalBlocks)）=====
    uint32_t totalBlocks = 0;
    uint32_t totalBN2 = (uint32_t)(constInfo.batchSize * constInfo.kHeadNum);
    for (uint32_t bN2 = 0; bN2 < totalBN2; bN2++) {
        uint32_t bIdx = bN2 / constInfo.kHeadNum;
        uint32_t actS1Size, actS2Size, actS2SizeOrig;
        GetS1S2ActualSeqLen(bIdx, actS1Size, actS2Size, actS2SizeOrig);
        if (actS1Size == 0 || actS2Size == 0) {
            continue;
        }
        uint32_t gS1Num = (actS1Size * constInfo.gSize + constInfo.mBaseSize - 1) / constInfo.mBaseSize;
        for (uint32_t gS1 = 0; gS1 < gS1Num; gS1++) {
            uint32_t w;
            if (constInfo.attenMaskFlag) {
                w = GetS2BaseBlockNumOnMask(gS1, actS1Size, actS2SizeOrig);
            } else {
                w = (actS2Size + constInfo.s2BaseSize - 1) / constInfo.s2BaseSize;
            }
            totalBlocks += w;
        }
    }
    if (totalBlocks == 0) {
        // 全空 case：清理输出
        isUsedCoreEqZero = true;
        splitCoreInfo.isCoreEnable = false;
        return;
    }

    // ===== 块池切分：所有行的所有 S2 块汇成池子，按块数均衡分给每个核（arch22 同式）=====
    // 每核 minBlock=totalBlocks/coreNum 块，前 deal1More=totalBlocks%coreNum 核多 1 块；
    // 核范围在行主序块空间上连续，可跨多行。只有被 >1 核覆盖的行（内部有核边界）需要 LD。
    uint32_t coreNum = (totalBlocks < AIC_CORE_NUM) ? totalBlocks : AIC_CORE_NUM;
    uint32_t minBlock = totalBlocks / coreNum;
    uint32_t deal1More = totalBlocks % coreNum;
    if (cubeCoreIdx >= coreNum) {
        splitCoreInfo.isCoreEnable = false;
        if ASCEND_IS_AIV {
            ldInfo.isLdCoreEnable = false;
        }
        return;
    }
    splitCoreInfo.isCoreEnable = true;
    uint64_t cb = (uint64_t)cubeCoreIdx * minBlock + ((cubeCoreIdx < deal1More) ? cubeCoreIdx : deal1More);
    uint64_t ce = (uint64_t)(cubeCoreIdx + 1) * minBlock + ((cubeCoreIdx + 1 < deal1More) ? (cubeCoreIdx + 1) : deal1More);

    // ===== Pass B：流式二次枚举行，定位本核范围 [cb,ce) 覆盖的行并统计每行覆盖核数 =====
    // 每行 touches=覆盖核数，touches>=2 的行占 touches 个连续 slot（slot 布局按行主序）。
    // fd 归约列表同步收集（AIV 侧 SplitFD 用）；被切分行必含核边界 → fdCount ≤ coreNum-1 ≤ 23。
    uint32_t bN2 = 0;
    uint32_t gS1 = 0;
    InitRowCursor(bN2, gS1);
    uint32_t blockAcc = 0;   // 当前行的起始块下标（行区间连续，等于已累计块数）
    uint32_t coreIdxWalk = 0;
    uint32_t slotCtr = 0;
    bool isR0 = false;  // 当前行是否为本核范围首块 cb 所在行
    bool isR1 = false;  // 当前行是否为本核范围末块 ce-1 所在行
    uint32_t firstCore0 = 0, slot0 = 0, w0 = 0;
    uint32_t firstCore1 = 0, slot1 = 0, w1 = 0;
    uint32_t fdCount = 0;
    uint32_t fdBN2[AIC_CORE_NUM], fdM[AIC_CORE_NUM], fdW[AIC_CORE_NUM];
    uint32_t fdSplitNum[AIC_CORE_NUM], fdMSize[AIC_CORE_NUM];
    while (true) {
        uint32_t w, tailM;
        RowCursorInfo(bN2, gS1, w, tailM);
        uint32_t rs = blockAcc;
        uint32_t re = blockAcc + w;
        isR0 = (rs <= cb && cb < re);
        isR1 = (rs <= ce - 1 && ce - 1 < re);
        if (isR0) {
            splitCoreInfo.bN2Start = bN2;
            splitCoreInfo.gS1Start = gS1;
            splitCoreInfo.s2Start = (cb > rs) ? (uint32_t)(cb - rs) : 0;
        }
        if (isR1) {
            splitCoreInfo.bN2End = bN2;
            splitCoreInfo.gS1End = gS1;
            splitCoreInfo.s2End = (uint32_t)(ce - 1 - rs);
        }
        while (coreIdxWalk + 1 < coreNum &&
               (uint64_t)(coreIdxWalk + 1) * minBlock + ((coreIdxWalk + 1 < deal1More) ? (coreIdxWalk + 1) : deal1More) <= rs) {
            coreIdxWalk++;
        }
        uint32_t firstCore = coreIdxWalk;
        uint32_t touches = 0;
        uint32_t c = firstCore;
        while (c < coreNum && (uint64_t)c * minBlock + ((c < deal1More) ? c : deal1More) < re) {
            touches++;
            c++;
        }
        uint32_t rowSlotBase = 0;
        if (touches >= 2) {
            rowSlotBase = slotCtr;
            slotCtr += touches;
            if (fdCount < AIC_CORE_NUM) {
                fdBN2[fdCount] = bN2;
                fdM[fdCount] = gS1;
                fdW[fdCount] = rowSlotBase;
                fdSplitNum[fdCount] = touches;
                fdMSize[fdCount] = tailM;
                fdCount++;
            }
        }
        if (isR0) {
            firstCore0 = firstCore;
            slot0 = rowSlotBase;
            w0 = w;
        }
        if (isR1) {
            firstCore1 = firstCore;
            slot1 = rowSlotBase;
            w1 = w;
        }
        coreIdxWalk = c - 1;
        blockAcc = re;
        if (!AdvanceRowCursor(bN2, gS1)) {
            break;
        }
    }

    // ===== 本核的部分行 slot：首行部分（范围首块所在行）+ 末行部分（≠首行且未覆盖完整）=====
    // 被切分行的 slot 按覆盖核序连续（firstCore 得 base、后续核 +1），本核 slot = base + (本核-firstCore)。
    ldInfo.ldSlotCount = 0;
    bool isSingleRow = (splitCoreInfo.bN2Start == splitCoreInfo.bN2End) &&
                       (splitCoreInfo.gS1Start == splitCoreInfo.gS1End);
    bool firstPartial = (splitCoreInfo.s2Start > 0);
    if (isSingleRow) {
        firstPartial = (splitCoreInfo.s2Start > 0) || (splitCoreInfo.s2End < w0 - 1);
    }
    if (firstPartial) {
        ldInfo.ldSlotBN2[ldInfo.ldSlotCount] = splitCoreInfo.bN2Start;
        ldInfo.ldSlotGS1[ldInfo.ldSlotCount] = splitCoreInfo.gS1Start;
        ldInfo.ldSlot[ldInfo.ldSlotCount] = slot0 + (cubeCoreIdx - firstCore0);
        ldInfo.ldSlotCount++;
    }
    if (!isSingleRow && splitCoreInfo.s2End < w1 - 1) {
        ldInfo.ldSlotBN2[ldInfo.ldSlotCount] = splitCoreInfo.bN2End;
        ldInfo.ldSlotGS1[ldInfo.ldSlotCount] = splitCoreInfo.gS1End;
        ldInfo.ldSlot[ldInfo.ldSlotCount] = slot1 + (cubeCoreIdx - firstCore1);
        ldInfo.ldSlotCount++;
    }

    if ASCEND_IS_AIV {
        // ===== SplitFD：把 fd 归约任务负载均衡分给 48 个 AIV =====
        if (fdCount == 0) {
            ldInfo.isLdCoreEnable = false;
            return;
        }
        uint64_t totalFDLoad = 0;
        for (uint32_t i = 0; i < fdCount; i++) {
            totalFDLoad += (uint64_t)fdSplitNum[i] * fdMSize[i];
        }
        uint64_t averageLoad = (totalFDLoad + AIV_CORE_NUM - 1) / AIV_CORE_NUM;
        uint32_t fdIdx[AIV_CORE_NUM], fdMStart[AIV_CORE_NUM], fdMNum[AIV_CORE_NUM];
        uint32_t curCoreIndex = 0;
        for (uint32_t i = 0; i < fdCount; i++) {
            uint32_t curFDVectorNum = (uint32_t)((uint64_t)fdSplitNum[i] * fdMSize[i] / averageLoad);
            curFDVectorNum = Max(1U, curFDVectorNum);
            uint32_t curAveMSize = (fdMSize[i] + curFDVectorNum - 1) / curFDVectorNum;
            curFDVectorNum = (fdMSize[i] + curAveMSize - 1) / curAveMSize;
            for (uint32_t vid = 0; vid < curFDVectorNum; vid++) {
                if (curCoreIndex >= AIV_CORE_NUM) {
                    break;
                }
                fdIdx[curCoreIndex] = i;
                fdMStart[curCoreIndex] = vid * curAveMSize;
                fdMNum[curCoreIndex] =
                    (vid < curFDVectorNum - 1) ? curAveMSize : fdMSize[i] - vid * curAveMSize;
                curCoreIndex++;
            }
        }
        if (vecCoreIdx >= curCoreIndex) {
            ldInfo.isLdCoreEnable = false;
            return;
        }
        uint32_t fi = fdIdx[vecCoreIdx];
        ldInfo.isLdCoreEnable = true;
        ldInfo.bn2Idx = fdBN2[fi];
        ldInfo.bIdx = ldInfo.bn2Idx / constInfo.kHeadNum;
        ldInfo.n2Idx = ldInfo.bn2Idx % constInfo.kHeadNum;
        ldInfo.mIdx = fdM[fi];
        ldInfo.workspaceIdx = fdW[fi];
        ldInfo.workspaceNum = fdSplitNum[fi];
        ldInfo.mStart = fdMStart[vecCoreIdx];
        ldInfo.mNum = fdMNum[vecCoreIdx];
        uint64_t actualSeqQPrefixSum = 0;
        if constexpr (Q_LAYOUT_T == LI_LAYOUT::TND) {
            uint32_t actualSeqLengthsGmQIdx = (constInfo.batchSupperFlag) ? ldInfo.bIdx : ldInfo.bIdx - 1;
            actualSeqQPrefixSum = (ldInfo.bIdx <= 0) ? 0 : actualSeqLengthsGmQ.GetValue(actualSeqLengthsGmQIdx);
        } else {  // BSND
            actualSeqQPrefixSum = (ldInfo.bIdx <= 0) ? 0 : (uint64_t)ldInfo.bIdx * constInfo.qSeqSize;
        }
        ldInfo.indiceOutCoreOffset = actualSeqQPrefixSum * constInfo.kHeadNum * constInfo.sparseCount +
                                     ldInfo.n2Idx * constInfo.sparseCount +
                                     ldInfo.mIdx * constInfo.s1BaseSize * constInfo.kHeadNum * constInfo.sparseCount;
    }
}

template <typename QLIT>
__aicore__ inline void QLIPreload<QLIT>::DealActSeqLenIsZero(uint32_t bIdx, uint32_t n2Idx, uint32_t s1Start)
{
    if ASCEND_IS_AIV {
        if (constInfo.outputLayout == LI_LAYOUT::TND) {
            uint32_t tSizeIdx = (constInfo.batchSupperFlag) ? constInfo.batchSize : constInfo.batchSize - 1;
            uint32_t tBaseIdx = (constInfo.batchSupperFlag) ? bIdx : bIdx - 1;
            uint32_t tSize = actualSeqLengthsGmQ.GetValue(tSizeIdx);
            uint32_t tBase = bIdx == 0 ? 0 : actualSeqLengthsGmQ.GetValue(tBaseIdx);
            uint32_t s1Count = tempLoopInfo.actS1Size;

            for (uint32_t s1Idx = s1Start; s1Idx < s1Count; s1Idx++) {
                uint64_t indiceOutOffset =
                    (tBase + s1Idx) * constInfo.kHeadNum * constInfo.sparseCount +  // T轴、s1轴偏移
                    n2Idx * constInfo.sparseCount;                                  // N2轴偏移
                vectorService.CleanInvalidOutput(indiceOutOffset);
            }
        } else if (constInfo.outputLayout == LI_LAYOUT::BSND) {
            for (uint32_t s1Idx = s1Start; s1Idx < constInfo.qSeqSize; s1Idx++) {
                // B,S1,N2,K
                uint64_t indiceOutOffset = bIdx * constInfo.qSeqSize * constInfo.kHeadNum * constInfo.sparseCount +
                                           s1Idx * constInfo.kHeadNum * constInfo.sparseCount +  // B轴、S1轴偏移
                                           n2Idx * constInfo.sparseCount;                        // N2轴偏移
                vectorService.CleanInvalidOutput(indiceOutOffset);
            }
        }
    }
}

template <typename QLIT>
__aicore__ inline void QLIPreload<QLIT>::Init(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *weights,
                                              __gm__ uint8_t *queryScale, __gm__ uint8_t *keyScale,
                                              __gm__ uint8_t *actualSeqLengthsQ, __gm__ uint8_t *actualSeqLengthsK,
                                              __gm__ uint8_t *blockTable,
                                              __gm__ uint8_t *sparseIndices,
                                              __gm__ uint8_t *routePairRows, __gm__ uint8_t *topkSlots,
                                              __gm__ uint8_t *missCount, __gm__ uint8_t *threshold,
                                              __gm__ uint8_t *cacheSlots, __gm__ uint8_t *reqPoolEntries,
                                              __gm__ uint8_t *cacheTokens, __gm__ uint8_t *workspace,
                                              __gm__ uint8_t *ldWorkspace,
                                              const QLITilingData *__restrict tiling, TPipe *tPipe)
{
    if ASCEND_IS_AIV {
        tmpBlockIdx = GetBlockIdx();  // vec:0-47
        aiCoreIdx = tmpBlockIdx / 2;
    } else {
        tmpBlockIdx = GetBlockIdx();  // cube:0-23
        aiCoreIdx = tmpBlockIdx;
    }

    InitTilingData(tiling);
    InitActualSeqLen(actualSeqLengthsQ, actualSeqLengthsK);

    // ComputeSplitInfo 的 IsActiveRequest 需要读 cacheTokens/reqPoolEntries，
    // 必须先绑定再分核，否则读到未初始化 GM（budget=0 → 全 request 判不活跃 →
    // totalBlocks=0 → isUsedCoreEqZero → 整个 kernel 只走 ProcessInvalid、不 publish）。
    cacheTokensGm.SetGlobalBuffer((__gm__ int32_t *)cacheTokens);
    reqPoolEntriesGm.SetGlobalBuffer((__gm__ int32_t *)reqPoolEntries);

    // 获取分核信息（on-device，无 metadata 输入）
    ComputeSplitInfo(aiCoreIdx, tmpBlockIdx);

    pipe = tPipe;

    uint32_t topkCountAlign16_ = QLILdCommon::Align(constInfo.sparseCount, (uint64_t)16); // topkCount对齐到16
    // request-major score：workspace 即 request-major 区（batch*4*sourceCapacity 行主序，
    // 行 r 在 requestBase + r*sourceCapacity）。scoreRowBase_ 由 ProcessMain 按 (bN2,gS1) 设。
    GlobalTensor<SCORE_T> scoreGm; // 存放vec核写出的score
    scoreGm.SetGlobalBuffer((__gm__ SCORE_T *)workspace);
    // vec 存储需要LD的s1对应的s2的score与index，大小为s1BaseSize * sparseCount * 2，一个核内最多有两个s1BaseSize需要LD
    // LD 区独立于 request-major scores，host 按 aicNum=GetBlockNum() 分配
    GlobalTensor<SCORE_T> ldScoreGm; // 存放进行LD的s2 score
    ldScoreGm.SetGlobalBuffer((__gm__ SCORE_T *)ldWorkspace);
    GlobalTensor<int32_t> ldIndexGm; // 存放进行LD的s2 Index
    ldIndexGm.SetGlobalBuffer((__gm__ int32_t *)(ldWorkspace +
        GetBlockNum() * constInfo.s1BaseSizeWs * topkCountAlign16_ * 2 * sizeof(SCORE_T)));

    if ASCEND_IS_AIV {
        vectorService.InitParams(constInfo, ldInfo, tiling);
        indiceOutGm.SetGlobalBuffer((__gm__ int32_t *)sparseIndices);
        weightsGm.SetGlobalBuffer((__gm__ bfloat16_t *)weights);
        qScaleGm.SetGlobalBuffer((__gm__ float *)queryScale);
        kScaleGm.SetGlobalBuffer((__gm__ float *)keyScale);
        blockTableGm.SetGlobalBuffer((__gm__ int32_t *)blockTable);
        cacheSlotsGm.SetGlobalBuffer((__gm__ int32_t *)cacheSlots);
        routePairsGm.SetGlobalBuffer((__gm__ int32_t *)routePairRows);
        topkSlotsGm.SetGlobalBuffer((__gm__ int32_t *)topkSlots);
        missCountGm.SetGlobalBuffer((__gm__ int32_t *)missCount);
        thresholdGm.SetGlobalBuffer((__gm__ uint16_t *)threshold);
        vectorService.InitVecInputTensor(weightsGm, qScaleGm, kScaleGm, indiceOutGm, blockTableGm);
        vectorService.InitMtpOutputTensor(routePairsGm, topkSlotsGm, missCountGm, thresholdGm,
                                          cacheSlotsGm, reqPoolEntriesGm, cacheTokensGm);
        vectorService.InitVecWorkspaceTensor(scoreGm, ldScoreGm, ldIndexGm);
    } else {
        matmulService.InitParams(constInfo);
        queryGm.SetGlobalBuffer((__gm__ Q_T *)query);
        if constexpr (PAGE_ATTENTION) {
            blockTableGm.SetGlobalBuffer((__gm__ int32_t *)blockTable);
        }
        keyGm.SetGlobalBuffer((__gm__ K_T *)key);
        matmulService.InitMm1GlobalTensor(blockTableGm, keyGm, queryGm);
    }
    InitBuffers();
}

template <typename QLIT>
__aicore__ inline void QLIPreload<QLIT>::GetBN2Idx(uint32_t bN2Idx)
{
    tempLoopInfo.bN2Idx = bN2Idx;
    tempLoopInfo.bIdx = bN2Idx / constInfo.kHeadNum;
    tempLoopInfo.n2Idx = bN2Idx % constInfo.kHeadNum;
}

template <typename QLIT>
__aicore__ inline void QLIPreload<QLIT>::CalcS2LoopParams(uint32_t bN2LoopIdx, uint32_t gS1LoopIdx)
{
    tempLoopInfo.gS1Idx = gS1LoopIdx;
    tempLoopInfo.actMBaseSize = constInfo.mBaseSize;
    uint32_t remainedGS1Size = tempLoopInfo.actS1Size * constInfo.gSize - tempLoopInfo.gS1Idx * constInfo.mBaseSize;
    if (remainedGS1Size <= constInfo.mBaseSize && remainedGS1Size > 0) {
        tempLoopInfo.actMBaseSize = tempLoopInfo.mBasicSizeTail;
    }

    bool isEnd = (bN2LoopIdx == splitCoreInfo.bN2End) && (gS1LoopIdx == splitCoreInfo.gS1End);
    uint32_t s2BlockNum;
    if (constInfo.attenMaskFlag) {
        s2BlockNum = GetS2BaseBlockNumOnMask(gS1LoopIdx, tempLoopInfo.actS1Size, tempLoopInfo.actS2SizeOrig);
    } else {
        s2BlockNum = (tempLoopInfo.actS2Size + constInfo.s2BaseSize - 1) / constInfo.s2BaseSize;
    }
    tempLoopInfo.s2LoopEnd = isEnd ? splitCoreInfo.s2End : s2BlockNum - 1;
    if (splitCoreInfo.s2Start > 0 || tempLoopInfo.s2LoopEnd < s2BlockNum - 1) {
        tempLoopInfo.isNeedLD = true;
    } else {
        tempLoopInfo.isNeedLD = false;
    }
}

template <typename QLIT>
__aicore__ inline void QLIPreload<QLIT>::CalcGS1LoopParams(uint32_t bN2LoopIdx)
{
    GetBN2Idx(bN2LoopIdx);
    GetS1S2ActualSeqLen(tempLoopInfo.bIdx, tempLoopInfo.actS1Size, tempLoopInfo.actS2Size, tempLoopInfo.actS2SizeOrig);
    if ((tempLoopInfo.actS2Size == 0) || (tempLoopInfo.actS1Size == 0)) {
        tempLoopInfo.curActSeqLenIsZero = true;
        return;
    }
    tempLoopInfo.curActSeqLenIsZero = false;
    tempLoopInfo.s2BasicSizeTail = tempLoopInfo.actS2Size % constInfo.s2BaseSize;
    tempLoopInfo.s2BasicSizeTail =
        (tempLoopInfo.s2BasicSizeTail == 0) ? constInfo.s2BaseSize : tempLoopInfo.s2BasicSizeTail;
    tempLoopInfo.mBasicSizeTail = (tempLoopInfo.actS1Size * constInfo.gSize) % constInfo.mBaseSize;
    tempLoopInfo.mBasicSizeTail =
        (tempLoopInfo.mBasicSizeTail == 0) ? constInfo.mBaseSize : tempLoopInfo.mBasicSizeTail;

    uint32_t gS1SplitNum = (tempLoopInfo.actS1Size * constInfo.gSize + constInfo.mBaseSize - 1) / constInfo.mBaseSize;
    tempLoopInfo.gS1LoopEnd = (bN2LoopIdx == splitCoreInfo.bN2End) ? splitCoreInfo.gS1End : gS1SplitNum - 1;
    if constexpr (Q_LAYOUT_T == LI_LAYOUT::BSND) {
        if (tempLoopInfo.gS1LoopEnd == gS1SplitNum - 1 && constInfo.qSeqSize > tempLoopInfo.actS1Size) {
            tempLoopInfo.needDealActS1LessThanS1 = true;
        }
    }
}

template <typename QLIT>
__aicore__ inline void QLIPreload<QLIT>::CalcRunInfo(uint32_t loop, uint32_t s2LoopIdx, QLILdCommon::RunInfo &runInfo,
                                                     uint32_t qScaleLoop, uint32_t kScaleLoop)
{
    runInfo.loop = loop;
    runInfo.qScaleLoop = qScaleLoop;
    runInfo.kScaleLoop = kScaleLoop;
    runInfo.bIdx = tempLoopInfo.bIdx;
    runInfo.gS1Idx = tempLoopInfo.gS1Idx;
    runInfo.s2Idx = s2LoopIdx;
    runInfo.bN2Idx = tempLoopInfo.bN2Idx;
    runInfo.isValid = s2LoopIdx <= tempLoopInfo.s2LoopEnd;
    runInfo.isNeedLD = tempLoopInfo.isNeedLD;
    // 块池切分下本核可能跨多行（至多 2 个部分行），部分行的 LD slot 按 (bN2,gS1) 查
    // ldInfo.ldSlot* 得到（被切分行的 slot 按行主序连续），不能再用逐行递增。
    if (runInfo.isNeedLD && s2LoopIdx == tempLoopInfo.s2LoopEnd) {
        runInfo.saveWorkSpaceIdx = 0;
        for (uint32_t i = 0; i < ldInfo.ldSlotCount; i++) {
            if (ldInfo.ldSlotBN2[i] == tempLoopInfo.bN2Idx && ldInfo.ldSlotGS1[i] == tempLoopInfo.gS1Idx) {
                runInfo.saveWorkSpaceIdx = ldInfo.ldSlot[i];
                break;
            }
        }
    }

    if (!runInfo.isValid) {
        return;  // 需要验证， v1 时候需要runInfo
    }

    runInfo.actS1Size = tempLoopInfo.actS1Size;
    runInfo.actS2Size = tempLoopInfo.actS2Size;
    runInfo.actS2SizeOrig = tempLoopInfo.actS2SizeOrig;
    // 计算实际基本块size
    runInfo.actMBaseSize = tempLoopInfo.actMBaseSize;
    runInfo.actualSingleProcessSInnerSize = constInfo.s2BaseSize;
    uint32_t s2SplitNum = (tempLoopInfo.actS2Size + constInfo.s2BaseSize - 1) / constInfo.s2BaseSize;
    if (runInfo.s2Idx == s2SplitNum - 1) {
        runInfo.actualSingleProcessSInnerSize = tempLoopInfo.s2BasicSizeTail;
    }
    runInfo.actualSingleProcessSInnerSizeAlign =
        QLILdCommon::Align((uint32_t)runInfo.actualSingleProcessSInnerSize, QLILdCommon::ConstInfo::BUFFER_SIZE_BYTE_32B);

    runInfo.isFirstS2InnerLoop = s2LoopIdx == splitCoreInfo.s2Start;
    runInfo.isLastS2InnerLoop = s2LoopIdx == tempLoopInfo.s2LoopEnd;
    runInfo.isAllLoopEnd = (runInfo.bN2Idx == splitCoreInfo.bN2End) && (runInfo.gS1Idx == splitCoreInfo.gS1End) &&
                           (runInfo.s2Idx == splitCoreInfo.s2End);

    if (runInfo.isFirstS2InnerLoop) {
        uint64_t actualSeqQPrefixSum;
        if constexpr (Q_LAYOUT_T == LI_LAYOUT::TND) {
            uint32_t actualSeqLengthsGmQIdx = (constInfo.batchSupperFlag) ? runInfo.bIdx : runInfo.bIdx - 1;
            actualSeqQPrefixSum = (runInfo.bIdx <= 0) ? 0 : actualSeqLengthsGmQ.GetValue(actualSeqLengthsGmQIdx);
        } else {  // BSND
            actualSeqQPrefixSum = (runInfo.bIdx <= 0) ? 0 : runInfo.bIdx * constInfo.qSeqSize;
        }
        uint64_t tndBIdxOffset = actualSeqQPrefixSum * constInfo.qHeadNum * constInfo.headDim;
        // B,S1,N1(N2,G),D
        queryCoreOffset = tndBIdxOffset + runInfo.gS1Idx * constInfo.mBaseSize * constInfo.headDim;
        // B,S1,N1(N2,G)/T,N1(N2,G)
        weightsCoreOffset = actualSeqQPrefixSum * constInfo.qHeadNum + runInfo.n2Idx * constInfo.gSize;
        // B,S1,N2,k/T,N2,k
        indiceOutCoreOffset =
            actualSeqQPrefixSum * constInfo.kHeadNum * constInfo.sparseCount + runInfo.n2Idx * constInfo.sparseCount;
    }
    uint64_t actualSeqKPrefixSum;
    if constexpr (K_LAYOUT_T == LI_LAYOUT::TND) { // T N2 D
        actualSeqKPrefixSum = (runInfo.bIdx <= 0) ? 0 : actualSeqLengthsGm.GetValue(runInfo.bIdx - 1);
        actualSeqKPrefixSum = actualSeqKPrefixSum / constInfo.cmpRatio;
    } else {
        actualSeqKPrefixSum = (runInfo.bIdx <= 0) ? 0 : runInfo.bIdx * constInfo.kSeqSize;
    }
    uint64_t tndBIdxOffsetForK = actualSeqKPrefixSum * constInfo.kHeadNum * constInfo.headDim;
    keyCoreOffset = tndBIdxOffsetForK + runInfo.s2Idx * constInfo.s2BaseSize * constInfo.kHeadNum * constInfo.headDim;
    keyScaleCoreOffset = (actualSeqKPrefixSum + runInfo.s2Idx * constInfo.s2BaseSize) * constInfo.kHeadNum;
    runInfo.tensorQueryOffset = queryCoreOffset;
    runInfo.tensorKeyOffset = keyCoreOffset;
    runInfo.tensorKeyScaleOffset = keyScaleCoreOffset;
    runInfo.tensorWeightsOffset = weightsCoreOffset;
    runInfo.indiceOutOffset = indiceOutCoreOffset;
    // MTP：非 LD publish 需要 cache 槽表基址（reqPoolEntries）与有效 token 数
    // （cacheTokens）。LD publish 在 ProcessLD 直接读 GM，此处只补非 LD 路径，
    // 每 route（bN2,gS1）内的 s2 循环取值恒定。
    runInfo.cacheRowIdx =
        static_cast<uint32_t>(reqPoolEntriesGm.GetValue(runInfo.bIdx));
    runInfo.cacheTokenCount =
        static_cast<uint32_t>(cacheTokensGm.GetValue(runInfo.bIdx));
}

template <typename QLIT>
__aicore__ inline void QLIPreload<QLIT>::Process()
{
#if C8_MTP_BISECT_PRINT
    if (GetBlockIdx() == 0U) {
        if ASCEND_IS_AIV {
            AscendC::printf("[C8BISECT] core0(AIV) Process enter\n");
        } else {
            AscendC::printf("[C8BISECT] core0(AIC) Process enter\n");
        }
    }
#endif
    if (isUsedCoreEqZero) {
        // 没有计算任务，直接清理输出
        ProcessInvalid();
        return;
    }

    ProcessMain();
#if C8_MTP_BISECT_PRINT
    if (GetBlockIdx() == 0U) {
        if ASCEND_IS_AIV {
            AscendC::printf("[C8BISECT] core0(AIV) ProcessMain done\n");
        } else {
            AscendC::printf("[C8BISECT] core0(AIC) ProcessMain done\n");
        }
    }
#endif

    ProcessDecode();
#if C8_MTP_BISECT_PRINT
    if (GetBlockIdx() == 0U) {
        if ASCEND_IS_AIV {
            AscendC::printf("[C8BISECT] core0(AIV) Process done\n");
        } else {
            AscendC::printf("[C8BISECT] core0(AIC) Process done\n");
        }
    }
#endif
}

template <typename QLIT>
__aicore__ inline void QLIPreload<QLIT>::ProcessInvalid()
{
    if ASCEND_IS_AIV {
        uint32_t aivCoreNum = GetBlockNum() * 2;  // 2 means c:v = 1:2
        uint64_t totalOutputSize =
            constInfo.batchSize * constInfo.qSeqSize * constInfo.kHeadNum * constInfo.sparseCount;
        uint64_t singleCoreSize =
            QLILdCommon::Align((totalOutputSize + aivCoreNum - 1) / aivCoreNum, GM_ALIGN_BYTES / sizeof(OUT_T));
        uint64_t baseSize = tmpBlockIdx * singleCoreSize;
        if (baseSize < totalOutputSize) {
            uint64_t dealSize =
                (baseSize + singleCoreSize <= totalOutputSize) ? singleCoreSize : totalOutputSize - baseSize;
            GlobalTensor<OUT_T> output = indiceOutGm[baseSize];
            AscendC::InitGlobalMemory(output, dealSize, constInfo.INVALID_IDX);
        }
    }
}

template <typename QLIT>
__aicore__ inline void QLIPreload<QLIT>::ProcessMain()
{
    if(!splitCoreInfo.isCoreEnable){
        return;
    }

    if ASCEND_IS_AIV {
        vectorService.AllocEventID();
        CrossCoreSetFlag<QLILdCommon::ConstInfo::QLI_SYNC_MODE4, PIPE_V>(QLILdCommon::ConstInfo::CROSS_VC_EVENT + 0);
        CrossCoreSetFlag<QLILdCommon::ConstInfo::QLI_SYNC_MODE4, PIPE_V>(QLILdCommon::ConstInfo::CROSS_VC_EVENT + 1);
    } else {
        matmulService.AllocEventID();
    }

    QLILdCommon::RunInfo runInfo;
    uint32_t gloop = 0;
    uint32_t qScaleLoop = 0;  // 每行 +1：weight/qScale 双缓冲槽位（官方同式）
#if C8_MTP_BISECT_SKIP_PROCESS_MAIN
    // BISECT：AIC/AIV 都跳过整个 bN2/gS1/s2 循环。AIC 走 FreeEventID + 等 CROSS_VC 0/1
    // （AIV 已在循环前 Set 0/1，满足）；AIV 走 FreeEventID。之后只剩 ProcessDecode 的
    // InitLDBuffers/ICachePreLoad/SyncAll + entry SyncAll。
    (void)runInfo;
    (void)gloop;
    (void)qScaleLoop;
#else
    for (uint32_t bN2LoopIdx = splitCoreInfo.bN2Start; bN2LoopIdx <= splitCoreInfo.bN2End; bN2LoopIdx++) {
        CalcGS1LoopParams(bN2LoopIdx);
        if (tempLoopInfo.curActSeqLenIsZero) {
            DealActSeqLenIsZero(tempLoopInfo.bIdx, tempLoopInfo.n2Idx, 0U);
            continue;
        }
        for (uint32_t gS1LoopIdx = splitCoreInfo.gS1Start; gS1LoopIdx <= tempLoopInfo.gS1LoopEnd; gS1LoopIdx++) {
            CalcS2LoopParams(bN2LoopIdx, gS1LoopIdx);
            if ASCEND_IS_AIV {
                // request-major score 行基址：bN2Idx*qSeqSize*sourceCapacity 为该 request 的
                // 4 条 route 起始（gS1=0），gS1 后续块再 +gS1*s1BaseSize*sourceCapacity。
                int64_t sourceCap =
                    QLILdCommon::Align((uint64_t)constInfo.kSeqSize, (uint64_t)constInfo.s2BaseSize);
                vectorService.SetScoreRowBase(
                    (int64_t)bN2LoopIdx * (int64_t)constInfo.qSeqSize * sourceCap +
                    (int64_t)gS1LoopIdx * (int64_t)constInfo.s1BaseSize * sourceCap);
            }
            runInfo.s2Start = splitCoreInfo.s2Start;
            runInfo.s2LoopEnd = tempLoopInfo.s2LoopEnd;
            uint32_t kScaleLoop = 0;  // 每 16 个 s2 块 +1：kScale 双缓冲槽位（官方同式）
            for (int s2LoopIdx = splitCoreInfo.s2Start; s2LoopIdx <= tempLoopInfo.s2LoopEnd; s2LoopIdx++) {
                if ((s2LoopIdx - (int)splitCoreInfo.s2Start) % 16 == 0) {
                    ++kScaleLoop;
                }
                ProcessBaseBlock(gloop, s2LoopIdx, runInfo, qScaleLoop, kScaleLoop);
                ++gloop;
            }
            ++qScaleLoop;
            splitCoreInfo.s2Start = 0;
        }
        if (tempLoopInfo.needDealActS1LessThanS1) {
            DealActSeqLenIsZero(tempLoopInfo.bIdx, tempLoopInfo.n2Idx, tempLoopInfo.actS1Size);
        }
        splitCoreInfo.gS1Start = 0;
    }
#endif  // C8_MTP_BISECT_SKIP_PROCESS_MAIN

    if ASCEND_IS_AIV {
        vectorService.FreeEventID();
    } else {
        matmulService.FreeEventID();
        CrossCoreWaitFlag<QLILdCommon::ConstInfo::QLI_SYNC_MODE4, PIPE_FIX>(QLILdCommon::ConstInfo::CROSS_VC_EVENT + 0); 
        CrossCoreWaitFlag<QLILdCommon::ConstInfo::QLI_SYNC_MODE4, PIPE_FIX>(QLILdCommon::ConstInfo::CROSS_VC_EVENT + 1); 
    }
}

template <typename QLIT>
__aicore__ inline void QLIPreload<QLIT>::ProcessBaseBlock(uint32_t loop, uint64_t s2LoopIdx,
                                                          QLILdCommon::RunInfo runInfo, uint32_t qScaleLoop,
                                                          uint32_t kScaleLoop)
{
    CalcRunInfo(loop, s2LoopIdx, runInfo, qScaleLoop, kScaleLoop);
#if C8_MTP_BISECT_PRINT
    if (loop == 0U && GetBlockIdx() == 0U) {
        if ASCEND_IS_AIV {
            AscendC::printf("[C8BISECT] core0(AIV) first block enter loop=0\n");
        } else {
            AscendC::printf("[C8BISECT] core0(AIC) first block enter loop=0\n");
        }
    }
#endif
    if ASCEND_IS_AIC {
        matmulService.ComputeMm1(runInfo);
    } else {
#if C8_MTP_BISECT_SKIP_PV1_ENTIRE
        // BISECT：AIV 完全跳过 ProcessVec1，仅保留跨核同步（等 C 核 mm1，
        // 通知 C 核可复用 mm1Res）。若仍崩 → 崩在 ProcessVec1 之外（setup/
        // ComputeMm1 loads/同步）。定位到根因后恢复为 0。
        CrossCoreWaitFlag<QLILdCommon::ConstInfo::QLI_SYNC_MODE4, PIPE_V>(
            QLILdCommon::ConstInfo::CROSS_CV_EVENT + runInfo.loop % 2);
        CrossCoreSetFlag<QLILdCommon::ConstInfo::QLI_SYNC_MODE4, PIPE_V>(
            QLILdCommon::ConstInfo::CROSS_VC_EVENT + runInfo.loop % 2);
#else
        vectorService.ProcessVec1(runInfo);
#if C8_MTP_BISECT_SKIP_TOPK
        // BISECT：跳过 ProcessTopK（非 LD publish + LD 部分写）。只剩 cube 打分 +
        // ProcessVec1 + 事件 + SyncAll。判定：
        //   仍 507015 → 崩在 cube/vec1/事件；不崩 → 崩在 ProcessTopK。
        (void)0;
#else
        if (runInfo.isLastS2InnerLoop) {   //本核s2last
            vectorService.ProcessTopK(runInfo);
        }
#endif  // C8_MTP_BISECT_SKIP_TOPK
#endif  // C8_MTP_BISECT_SKIP_PV1_ENTIRE
    }
#if C8_MTP_BISECT_PRINT
    if (loop == 0U && GetBlockIdx() == 0U) {
        if ASCEND_IS_AIV {
            AscendC::printf("[C8BISECT] core0(AIV) first block done loop=0\n");
        } else {
            AscendC::printf("[C8BISECT] core0(AIC) first block done loop=0\n");
        }
    }
#endif
}

 template <typename QLIT>
__aicore__ inline void QLIPreload<QLIT>::ProcessDecode()
{
    if ASCEND_IS_AIV {
        vectorService.InitLDBuffers(pipe, ldInfo);
        ICachePreLoad(LD_PREFETCH_LEN);
#if C8_MTP_BISECT_PRINT
        if (GetBlockIdx() == 0U) {
            AscendC::printf("[C8BISECT] core0(AIV) SyncAll enter\n");
        }
#endif
        SyncAll();
#if C8_MTP_BISECT_PRINT
        if (GetBlockIdx() == 0U) {
            AscendC::printf("[C8BISECT] core0(AIV) SyncAll done\n");
        }
#endif
#if C8_MTP_BISECT_SKIP_LD
        // BISECT：跳过 ProcessLD（LD 归并读 + LdTopK）。cube 打分、ProcessVec1、
        // ProcessTopK 部分写、事件、SyncAll 全保留。判定：
        //   仍 507015 → 崩在 cube/vec1/topk写/事件；不崩 → 崩在 ProcessLD。
        (void)0;
#else
        if (ldInfo.isLdCoreEnable) {
            // LD 归并行按 candidate-3+r 截断，需当前 request 的有效 S2 原始长度
            uint32_t ldActS1Size, ldActS2Size, ldActS2SizeOrig;
            GetS1S2ActualSeqLen(ldInfo.bIdx, ldActS1Size, ldActS2Size, ldActS2SizeOrig);
            vectorService.SetLdSeqLen((int32_t)ldActS1Size, (int32_t)ldActS2SizeOrig);
            vectorService.ProcessLD();
        }
#endif  // C8_MTP_BISECT_SKIP_LD
    }
}

}  // namespace QLILdKernel
#endif  // quant_lightning_indexer_KERNEL_H