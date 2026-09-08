/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file indexer_coarse_screen_kernel.h
 * \brief
 */

#ifndef INDEXER_COARSE_SCREEN_KERNEL_H
#define INDEXER_COARSE_SCREEN_KERNEL_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "../indexer_coarse_screen_common.h"
#include "indexer_coarse_screen_service_vector.h"
#include "indexer_coarse_screen_service_cube.h"

namespace LIKernel {
using namespace IndexerCoarseScreenCommon;
using namespace IndexerCoarseScreenServiceVec;
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
    uint32_t gS1LoopEnd = 0U;      // gS1方向循环的结束Idx
    uint32_t s2LoopEnd = 0U;       // S2方向循环的结束Idx
    uint32_t actS1Size = 1ULL;     // 当前Batch循环处理的S1轴的实际大小
    uint32_t actS2Size = 0ULL;
    bool curActSeqLenIsZero = false;
    bool needDealActS1LessThanS1 = false; // S1的实际长度小于shape的S1长度时，是否需要清理输出
    uint32_t actMBaseSize = 0U;    // m轴(gS1)方向实际大小
    uint32_t mBasicSizeTail = 0U;  // gS1方向循环的尾基本块大小
    uint32_t s2BasicSizeTail = 0U; // S2方向循环的尾基本块大小
};

template <typename LIT>
class IndexerCoarseScreenKernel {
public:
    __aicore__ inline IndexerCoarseScreenKernel(){};
    __aicore__ inline void Init(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *weights,
                                __gm__ uint8_t *rowWeights, __gm__ uint8_t *actualSeqLengthsQ,
                                __gm__ uint8_t *actualSeqLengths,
                                __gm__ uint8_t *blockTable, __gm__ uint8_t *sparseIndices, __gm__ uint8_t *workspace,
                                const IndexerCoarseScreenTilingData *__restrict tiling, TPipe *tPipe);
    __aicore__ inline void Process();

    // =================================类型定义区=================================
    static constexpr bool DT_W_FLAG = LIT::weightsTypeFlag;
    using Q_T = typename LIT::queryType;
    using K_T = typename LIT::keyType;
    using OUT_T = typename LIT::outputType;
    static constexpr bool PAGE_ATTENTION = LIT::pageAttention;
    static constexpr LI_LAYOUT LAYOUT_T = LIT::layout;
    static constexpr LI_LAYOUT K_LAYOUT_T = LIT::keyLayout;
    // 编译期条件选择模板第二个参数的类型，直接声明W_T
    // 第一个模板参数：固定为Q_T；第二个模板参数：编译期选float/void
    using W_T = typename IndexerCoarseScreenTypeTraits<Q_T,
                                            typename std::conditional<DT_W_FLAG, float, void>::type>::weightsType;

    using MM1_OUT_T = float;

    IndexerCoarseScreenServiceCube<LIT> matmulService;
    IndexerCoarseScreenServiceVector<LIT> vectorService;

    // =================================常量区=================================
    static constexpr uint32_t SYNC_C1_V1_FLAG = 4;
    static constexpr uint32_t SYNC_V1_C1_FLAG = 5;

    static constexpr uint32_t M_BASE_SIZE = 512;
    static constexpr uint32_t S2_BASE_SIZE = 512;
    static constexpr uint32_t HEAD_DIM = 128;
    static constexpr uint32_t K_HEAD_NUM = 1;
    static constexpr uint32_t GM_ALIGN_BYTES = 512;
    static constexpr uint32_t SPARSE_COUNT_8K = 8192;
    static constexpr uint32_t BLOCK_CUBE_SIZE = 16;

    static constexpr int64_t LD_PREFETCH_LEN = 2;
    // for workspace double
    static constexpr uint32_t WS_DOUBLE = 2;

protected:
    TPipe *pipe = nullptr;

    // offset
    uint64_t queryCoreOffset = 0ULL;
    uint64_t keyCoreOffset = 0ULL;
    uint64_t weightsCoreOffset = 0ULL;
    uint64_t indiceOutCoreOffset = 0ULL;

    // ================================Global Buffer区=================================
    GlobalTensor<Q_T> queryGm;       // Stage-1 输入 query [N,H,D] TND(池化读);Stage-2 AIC 侧源 = qBarGm
    GlobalTensor<W_T> weightsGm;     // Stage-1 输入 weights [N,H](池化读)
    GlobalTensor<W_T> rowWeightsGm;  // Stage-1 输入 row_weights [R,g](池化读)
    GlobalTensor<Q_T> qBarGm;        // workspace 代理 [K,H,D] bf16
    GlobalTensor<W_T> wBarGm;        // workspace 代理 [K,H] W_T

    GlobalTensor<int32_t> indiceOutGm;
    GlobalTensor<int32_t> blockTableGm;

    GlobalTensor<uint32_t> actualSeqLengthsGmQ;
    GlobalTensor<uint32_t> actualSeqLengthsGm;
    GlobalTensor<K_T> paKeyGm;            // 原始 PA key cache(cube 侧全前缀散行直读)
    // workspace
    GlobalTensor<MM1_OUT_T> mm1ResGm;  // 存放S
    GlobalTensor<float> vec1ResGm;     // 存放TopK计算中间结果
    GlobalTensor<int64_t> vec1ParamGm; // 存放LD参数信息

    // ================================类成员变量====================================
    // aic、aiv核信息
    uint32_t tmpBlockIdx = 0U;
    uint32_t aiCoreIdx = 0U;
    uint32_t usedCoreNum = 0U;

    IndexerCoarseScreenCommon::ConstInfo constInfo{};
    TempLoopInfo tempLoopInfo{};
    IndexerCoarseScreenCommon::SplitCoreInfo splitCoreInfo{};

    // ================================Init functions==================================
    __aicore__ inline void InitTilingData(const IndexerCoarseScreenTilingData *__restrict tilingData);
    __aicore__ inline void InitBuffers();
    __aicore__ inline void InitActualSeqLen(__gm__ uint8_t *actualSeqLengthsQ, __gm__ uint8_t *actualSeqLengths);
    // ================================Split Core================================
    __aicore__ inline void SplitCore(uint32_t curCoreIdx, uint32_t &coreNum, IndexerCoarseScreenCommon::SplitCoreInfo &info);
    __aicore__ inline uint32_t GetS2BaseBlockNumOnMask(uint32_t s1gIdx, uint32_t actS1Size, uint32_t actS2Size);
    __aicore__ inline uint32_t GetTotalBaseBlockNum();
    // ================================Process functions================================
    __aicore__ inline void ProcessMain();
    __aicore__ inline void ProcessBaseBlock(uint32_t loop, uint64_t s2LoopIdx, IndexerCoarseScreenCommon::RunInfo &runInfo);
    __aicore__ inline void ProcessDecode();
    __aicore__ inline void ProcessInvalid();
    // ================================Params Calc=====================================
    __aicore__ inline void CalcGS1LoopParams(uint32_t bN2Idx);
    __aicore__ inline void GetBN2Idx(uint32_t bN2Idx);
    __aicore__ inline uint32_t GetActualSeqLen(uint32_t bIdx, uint32_t actualLenDims, bool isAccumSeq,
                                               GlobalTensor<uint32_t> &actualSeqLengthsGm, uint32_t defaultSeqLen);
    __aicore__ inline void GetS1S2ActualSeqLen(uint32_t bIdx, uint32_t &actS1Size, uint32_t &actS2Size);
    __aicore__ inline void CalcS2LoopParams(uint32_t bN2LoopIdx, uint32_t gS1LoopIdx);
    __aicore__ inline void CalcRunInfo(uint32_t loop, uint32_t s2LoopIdx, IndexerCoarseScreenCommon::RunInfo &runInfo);
    __aicore__ inline void DealActSeqLenIsZero(uint32_t bIdx, uint32_t n2Idx, uint32_t s1Start);
};

template <typename LIT>
__aicore__ inline void IndexerCoarseScreenKernel<LIT>::InitTilingData(const IndexerCoarseScreenTilingData *__restrict tilingData)
{
    usedCoreNum = tilingData->usedCoreNum;
    constInfo.batchSize = tilingData->bSize;
    constInfo.qHeadNum = constInfo.gSize = tilingData->gSize;
    constInfo.kSeqSize = tilingData->s2Size;
    constInfo.qSeqSize = tilingData->s1Size;
    // coarse 语义:Stage-2 全前缀 topk,无窗口掩码、无 values 输出 → attenMaskFlag/returnValue 恒 false
    constInfo.attenMaskFlag = false;
    constInfo.kCacheBlockSize = tilingData->blockSize;
    constInfo.maxBlockNumPerBatch = tilingData->maxBlockNumPerBatch;
    constInfo.sparseCount = tilingData->sparseCount; // = coarse_count(输出 topk 宽度,4096)
    constInfo.gMax = tilingData->gMax;               // 本地窗 group 宽上界 = row_weights.dim1(窗宽 g,窗宽上界)
    constInfo.outRowWidth = tilingData->outRowWidth; // Align8(sparseCount + 2*gMax - 1), 输出单行宽(有效 + -1 pad)
    constInfo.preTokens = INT64_MAX;
    constInfo.nextTokens = INT64_MAX;
    constInfo.returnValue = false;

    constInfo.outputLayout = LAYOUT_T; // 输出和输入形状一致
    if (LAYOUT_T == LI_LAYOUT::TND) {
        constInfo.isAccumSeqS1 = true;
    }
    if (K_LAYOUT_T == LI_LAYOUT::TND) {
        constInfo.isAccumSeqS2 = true;
    }

    constInfo.kHeadNum = K_HEAD_NUM;
    constInfo.headDim = HEAD_DIM;
    constInfo.s2BaseSize = S2_BASE_SIZE;
    constInfo.isSparseCountOver2K = (constInfo.sparseCount <= BASE_TOPK) ? false : true;

    constInfo.s1BaseSize = constInfo.isSparseCountOver2K ? SPARSE_COUNT_8K / constInfo.sparseCount * 2 : 8;
    constInfo.mBaseSize = constInfo.s1BaseSize * constInfo.gSize;
    constInfo.mBaseSizeAlign = IndexerCoarseScreenCommon::Align(constInfo.mBaseSize, BLOCK_CUBE_SIZE);
}

template <typename LIT>
__aicore__ inline void IndexerCoarseScreenKernel<LIT>::InitBuffers()
{
    if ASCEND_IS_AIV {
        vectorService.InitBuffers(pipe);
    } else {
        matmulService.InitBuffers(pipe);
    }
}

template <typename LIT>
__aicore__ inline void IndexerCoarseScreenKernel<LIT>::InitActualSeqLen(__gm__ uint8_t *actualSeqLengthsQ,
                                                            __gm__ uint8_t *actualSeqLengths)
{
    if (actualSeqLengthsQ == nullptr) {
        constInfo.actualLenQDims = 0;
    } else {
        constInfo.actualLenQDims = constInfo.batchSize;
        actualSeqLengthsGmQ.SetGlobalBuffer((__gm__ uint32_t *)actualSeqLengthsQ, constInfo.actualLenQDims);
    }
    if (actualSeqLengths == nullptr) {
        constInfo.actualLenDims = 0;
    } else {
        constInfo.actualLenDims = constInfo.batchSize;
        actualSeqLengthsGm.SetGlobalBuffer((__gm__ uint32_t *)actualSeqLengths, constInfo.actualLenDims);
    }
}

template <typename LIT>
__aicore__ inline uint32_t IndexerCoarseScreenKernel<LIT>::GetActualSeqLen(uint32_t bIdx,
                                                               uint32_t actualLenDims, bool isAccumSeq,
                                                               GlobalTensor<uint32_t> &actualSeqLengthsGm,
                                                               uint32_t defaultSeqLen)
{
    if (actualLenDims == 0) {
        return defaultSeqLen;
    } else if (isAccumSeq && bIdx > 0) {
        return actualSeqLengthsGm.GetValue(bIdx) - actualSeqLengthsGm.GetValue(bIdx - 1);
    } else {
        return actualSeqLengthsGm.GetValue(bIdx);
    }
}

template <typename LIT>
__aicore__ inline void IndexerCoarseScreenKernel<LIT>::GetS1S2ActualSeqLen(uint32_t bIdx,
                                                             uint32_t &actS1Size, uint32_t &actS2Size)
{
    // coarse 融合语义(域 [0,L)):每请求 1 行代理(q_bar),S1 恒为 1。
    // 组宽 g_r = aslq cum 差分(与 PreprocessMean 同取法);L = 自然长度(aslk) − g_r。
    // S2 = 域长 L(非累加 PA_BSND):cube PA 读、子片数随 L 收缩,自有 token 永不入粗筛。
    actS1Size = 1;
    uint32_t seqLen =
        GetActualSeqLen(bIdx, constInfo.actualLenDims, constInfo.isAccumSeqS2, actualSeqLengthsGm, constInfo.kSeqSize);
    uint32_t gR = constInfo.gMax;
    if (constInfo.actualLenQDims != 0) {
        uint32_t cumPrev = (bIdx == 0) ? 0 : actualSeqLengthsGmQ.GetValue(bIdx - 1);
        uint32_t cumDiff = actualSeqLengthsGmQ.GetValue(bIdx) - cumPrev;
        if (cumDiff > 0 && cumDiff < gR) {
            gR = cumDiff;
        }
    }
    actS2Size = (seqLen > gR) ? (seqLen - gR) : 0;
}

template <typename LIT>
__aicore__ inline uint32_t IndexerCoarseScreenKernel<LIT>::GetS2BaseBlockNumOnMask(uint32_t s1gIdx, uint32_t actS1Size,
                                                                       uint32_t actS2Size)
{
    if (actS2Size == 0) {
        return 0;
    }
    uint32_t s1Offset = constInfo.s1BaseSize * s1gIdx;
    int32_t validS2LenBase = static_cast<int32_t>(actS2Size) - static_cast<int32_t>(actS1Size);
    int32_t validS2Len = s1Offset + validS2LenBase + constInfo.s1BaseSize;
    validS2Len = Min(validS2Len, static_cast<int32_t>(actS2Size));
    validS2Len = Max(validS2Len, 1);
    return (validS2Len + constInfo.s2BaseSize - 1) / constInfo.s2BaseSize;
}

template <typename LIT>
__aicore__ inline uint32_t IndexerCoarseScreenKernel<LIT>::GetTotalBaseBlockNum()
{
    uint32_t totalBlockNum = 0;
    uint32_t actS1Size, actS2Size;
    uint32_t s1GBaseNum, s2BaseNum;
    for (uint32_t bIdx = 0; bIdx < constInfo.batchSize; bIdx++) {
        GetS1S2ActualSeqLen(bIdx, actS1Size, actS2Size);
        s1GBaseNum = CeilDiv(actS1Size, constInfo.s1BaseSize);
        if (!constInfo.attenMaskFlag) {
            // 域 [0,L) 下 L 可能为 0(seq_lens==g_r):仍需计 1 块使该请求被某核拥有,
            // 主流水对 L==0 走 DealActSeqLenIsZero 发窗行,故 over2k 每请求恒 1 块。
            s2BaseNum = constInfo.isSparseCountOver2K ? 1 : CeilDiv(actS2Size, constInfo.s2BaseSize);
            totalBlockNum += s1GBaseNum * s2BaseNum * constInfo.kHeadNum;
            continue;
        }
        for (uint32_t s1gIdx = 0; s1gIdx < s1GBaseNum; s1gIdx++) {
            s2BaseNum = constInfo.isSparseCountOver2K
                      ? 1
                      : GetS2BaseBlockNumOnMask(s1gIdx, actS1Size, actS2Size);
            totalBlockNum += s2BaseNum * constInfo.kHeadNum;
        }
    }
    return totalBlockNum;
}

// 多核版本，双闭区间
template <typename LIT>
__aicore__ void inline IndexerCoarseScreenKernel<LIT>::SplitCore(uint32_t curCoreIdx,
                                                             uint32_t &coreNum, IndexerCoarseScreenCommon::SplitCoreInfo &info)
{
    // 计算每个核最少处理的块数, 剩余的部分前面的核每个核多处理一块
    uint32_t totalBlockNum = GetTotalBaseBlockNum();
    uint32_t minBlockPerCore = totalBlockNum / coreNum;
    uint32_t deal1MoreBlockCoreNum = totalBlockNum % coreNum;
    uint32_t coreIdx = 0;
    uint32_t lastGS1RemainBlockCnt = 0;
    uint32_t coreDealBlockCnt = coreIdx < deal1MoreBlockCoreNum ? minBlockPerCore + 1 : minBlockPerCore;
    coreNum = minBlockPerCore == 0 ? deal1MoreBlockCoreNum : coreNum;

    bool findLastCoreEnd = true;
    uint32_t actS1Size, actS2Size;
    uint32_t s1GBaseNum, s2BaseNum, s2Loop;
    for (uint32_t bN2Idx = 0; bN2Idx < constInfo.batchSize * constInfo.kHeadNum; bN2Idx++) {
        uint32_t bIdx = bN2Idx / constInfo.kHeadNum;
        if (bN2Idx % constInfo.kHeadNum == 0) {
            GetS1S2ActualSeqLen(bIdx, actS1Size, actS2Size);
            s1GBaseNum = CeilDiv(actS1Size, constInfo.s1BaseSize);
            s2BaseNum = CeilDiv(actS2Size, constInfo.s2BaseSize);
        }
        if constexpr (LAYOUT_T == LI_LAYOUT::BSND) {
            if (findLastCoreEnd && (s1GBaseNum == 0U || s2BaseNum == 0U)) {
                info.bN2Start = bN2Idx;
                info.gS1Start = 0;
                info.s2Start = 0;
                findLastCoreEnd = false;
            }
        }
        for (uint32_t gS1Idx = 0; gS1Idx < s1GBaseNum; gS1Idx++) {
            if (constInfo.attenMaskFlag) {
                s2BaseNum = GetS2BaseBlockNumOnMask(gS1Idx, actS1Size, actS2Size);
            }
            if (findLastCoreEnd && s2BaseNum == 0U) {
                info.bN2Start = bN2Idx;
                info.gS1Start = gS1Idx;
                info.s2Start = 0;
                findLastCoreEnd = false;
            }
            // over2k 每请求 1 块(L==0 也计 1,交由 DealActSeqLenIsZero 发窗行)
            s2Loop = constInfo.isSparseCountOver2K ? 1 : s2BaseNum;
            for (uint32_t s2Idx = 0; s2Idx < s2Loop;) {
                if (findLastCoreEnd) {
                    info.bN2Start = bN2Idx;
                    info.gS1Start = gS1Idx;
                    info.s2Start = s2Idx;
                    findLastCoreEnd = false;
                }
                uint32_t s2RemainBaseNum = s2Loop - s2Idx;
                if (lastGS1RemainBlockCnt + s2RemainBaseNum >= coreDealBlockCnt) {
                    info.bN2End = bN2Idx;
                    info.gS1End = gS1Idx;
                    // L==0 时 s2BaseNum==0,s2End 置 0(主流水短接不消费);正常请求 = 全子片末片
                    info.s2End = constInfo.isSparseCountOver2K
                               ? (s2BaseNum > 0 ? s2BaseNum - 1 : 0)
                               : s2Idx + coreDealBlockCnt - lastGS1RemainBlockCnt - 1;

                    if (coreIdx == curCoreIdx) {
                        // S2被切N核，那么只有第一个核需要处理LD，其他核不用
                        if (s2Idx == 0 && info.s2End + 1 < s2BaseNum) {
                            info.isLD = true;
                        }
                        // 最后一个核处理的不是最后一个Batch，表明后面的Batch为空块(S2=0), 调整终点坐标以便清理输出
                        if (coreIdx == coreNum - 1 && info.bN2End != constInfo.batchSize -1) {
                            info.bN2End = constInfo.batchSize -1;
                            info.gS1End = 0;
                            info.s2End = 0;
                        }
                        return;
                    }
                    coreIdx++;
                    findLastCoreEnd = true;
                    s2Idx = info.s2End + 1;
                    lastGS1RemainBlockCnt = 0;
                    coreDealBlockCnt = coreIdx < deal1MoreBlockCoreNum ? minBlockPerCore + 1 : minBlockPerCore;
                } else {
                    lastGS1RemainBlockCnt += s2RemainBaseNum;
                    break;
                }
            }
        }
    }
}

template <typename LIT>
__aicore__ inline void IndexerCoarseScreenKernel<LIT>::DealActSeqLenIsZero(uint32_t bIdx, uint32_t n2Idx, uint32_t s1Start)
{
    if ASCEND_IS_AIV {
        if (constInfo.outputLayout == LI_LAYOUT::TND) {
            // L==0(seq_lens==g_r,域空):输出该请求自有窗行 [0,g_r) 后 -1 pad 至 outRowWidth,
            // actS1Size 恒为 1 每请求仅 1 行(双 AIV 冗余写同内容,值一致,与旧 CleanInvalidOutput 同构)。
            for (uint32_t s1Idx = s1Start; s1Idx < tempLoopInfo.actS1Size; s1Idx++) {
                uint64_t indiceOutOffset =
                    (uint64_t)bIdx * constInfo.kHeadNum * constInfo.outRowWidth + // B轴(请求)偏移
                    (uint64_t)n2Idx * constInfo.outRowWidth;                      // N2轴偏移
                vectorService.WriteZeroRow(bIdx, indiceOutOffset);
            }
        } else if (constInfo.outputLayout == LI_LAYOUT::BSND) {
            for (uint32_t s1Idx = s1Start; s1Idx < constInfo.qSeqSize; s1Idx++) {
                // B,S1,N2,K
                uint64_t indiceOutOffset = (uint64_t)bIdx * constInfo.qSeqSize * constInfo.kHeadNum *
                                               constInfo.outRowWidth +
                                           (uint64_t)s1Idx * constInfo.kHeadNum * constInfo.outRowWidth + // B轴、S1轴偏移
                                           (uint64_t)n2Idx * constInfo.outRowWidth; // N2轴偏移
                vectorService.CleanInvalidOutput(indiceOutOffset);
            }
        }
    }
}

template <typename LIT>
__aicore__ inline void IndexerCoarseScreenKernel<LIT>::Init(__gm__ uint8_t *query,
                                                __gm__ uint8_t *key, __gm__ uint8_t *weights,
                                                __gm__ uint8_t *rowWeights,
                                                __gm__ uint8_t *actualSeqLengthsQ, __gm__ uint8_t *actualSeqLengths,
                                                __gm__ uint8_t *blockTable, __gm__ uint8_t *sparseIndices,
                                                __gm__ uint8_t *workspace, const IndexerCoarseScreenTilingData *__restrict tiling,
                                                TPipe *tPipe)
{
    if ASCEND_IS_AIV {
        tmpBlockIdx = GetBlockIdx(); // vec:0-47
        aiCoreIdx = tmpBlockIdx / 2;
    } else {
        tmpBlockIdx = GetBlockIdx(); // cube:0-23
        aiCoreIdx = tmpBlockIdx;
    }

    InitTilingData(tiling);
    InitActualSeqLen(actualSeqLengthsQ, actualSeqLengths);

    // 计算分核
    SplitCore(aiCoreIdx, usedCoreNum, splitCoreInfo);

    pipe = tPipe;
    // workspace 内存排布:与生产 lightning_indexer 完全一致
    // |mm1ResGm(存S)|vec1ResGm(存LD中间结果)|vec1ParamGm(存LD参数)|qBarGm(代理)|wBarGm(代理)
    // |Core0_mm1ResDB0-Core0_mm1ResDB1-Core1_mm1ResDB0....Core23_mm1ResDB0-Core23_mm1ResDB1|Core0_vec1Res...
    // |qBarGm[batchSize,gSize,headDim]|wBarGm[batchSize,gSize]| (Stage-1 代理,全局共享,非按核)
    uint64_t offset = 0;

    // mm1开DoubleBuffer
    uint64_t singleCoreMm1ResSize = WS_DOUBLE * constInfo.mBaseSizeAlign * constInfo.s2BaseSize * sizeof(MM1_OUT_T);
    mm1ResGm.SetGlobalBuffer((__gm__ MM1_OUT_T *)(workspace + offset + aiCoreIdx * singleCoreMm1ResSize));
    offset += GetBlockNum() * singleCoreMm1ResSize;

    // ld流程需要ws大小: [aicnum, 2, CeilDiv(constInfo.mBaseSize, constInfo.gSize), topkOut_*2]
    // (aic, 8, 2, 2, 2048)
    // (aic, s1_cube, 头尾, idx/value, K)
    vec1ResGm.SetGlobalBuffer((__gm__ float *)(workspace + offset));
    offset += GetBlockNum() * constInfo.s1BaseSize * WS_DOUBLE * WS_DOUBLE * BASE_TOPK * sizeof(float);

    // (aic, 8, 2, 16)
    // (aic, s1_cube, 头尾，16ele)
    vec1ParamGm.SetGlobalBuffer((__gm__ int64_t *)(workspace + offset));
    offset += GetBlockNum() * constInfo.s1BaseSize * WS_DOUBLE * LD_PARAM_NUM * sizeof(int64_t);

    // Stage-1 组均值代理 workspace(全局共享)
    qBarGm.SetGlobalBuffer((__gm__ Q_T *)(workspace + offset));
    offset += constInfo.batchSize * constInfo.gSize * constInfo.headDim * sizeof(Q_T);
    wBarGm.SetGlobalBuffer((__gm__ W_T *)(workspace + offset));
    offset += constInfo.batchSize * constInfo.gSize * sizeof(W_T);

    if ASCEND_IS_AIV {
        vectorService.InitParams(constInfo, tiling);
        indiceOutGm.SetGlobalBuffer((__gm__ int32_t *)sparseIndices);
        // Stage-1 池化输入(query 原值、逐行 weights、row_weights)
        queryGm.SetGlobalBuffer((__gm__ Q_T *)query);
        weightsGm.SetGlobalBuffer((__gm__ W_T *)weights);
        rowWeightsGm.SetGlobalBuffer((__gm__ W_T *)rowWeights);
        // Stage-2 weights 源 = wBarGm(代理),indiceOut 按请求写 [K,1,sparse_count]
        vectorService.InitVec1GlobalTensor(mm1ResGm, vec1ResGm, vec1ParamGm, wBarGm, indiceOutGm);
        vectorService.InitPreprocessTensor(queryGm, weightsGm, rowWeightsGm, qBarGm, wBarGm, actualSeqLengthsGmQ);
    } else {
        matmulService.InitParams(constInfo);
        // coarse key 输入 = 原始 PA key cache,cube 侧按全前缀逻辑位置经块表直读散行
        // Stage-2 query 源 = qBarGm(代理 [K,H,D])
        if constexpr (PAGE_ATTENTION) {
            blockTableGm.SetGlobalBuffer((__gm__ int32_t *)blockTable);
        }
        paKeyGm.SetGlobalBuffer((__gm__ K_T *)key);
        matmulService.InitMm1GlobalTensor(blockTableGm, paKeyGm, qBarGm, mm1ResGm);
    }
    InitBuffers();
}

template <typename LIT>
__aicore__ inline void IndexerCoarseScreenKernel<LIT>::GetBN2Idx(uint32_t bN2Idx)
{
    tempLoopInfo.bN2Idx = bN2Idx;
    tempLoopInfo.bIdx = bN2Idx / constInfo.kHeadNum;
    tempLoopInfo.n2Idx = bN2Idx % constInfo.kHeadNum;
}

template <typename LIT>
__aicore__ inline void IndexerCoarseScreenKernel<LIT>::CalcS2LoopParams(uint32_t bN2LoopIdx, uint32_t gS1LoopIdx)
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
        s2BlockNum = GetS2BaseBlockNumOnMask(gS1LoopIdx, tempLoopInfo.actS1Size, tempLoopInfo.actS2Size);
    } else {
        s2BlockNum = (tempLoopInfo.actS2Size + constInfo.s2BaseSize - 1) / constInfo.s2BaseSize;
    }
    tempLoopInfo.s2LoopEnd = isEnd ? splitCoreInfo.s2End : s2BlockNum - 1;
}

template <typename LIT>
__aicore__ inline void IndexerCoarseScreenKernel<LIT>::CalcGS1LoopParams(uint32_t bN2LoopIdx)
{
    GetBN2Idx(bN2LoopIdx);
    GetS1S2ActualSeqLen(tempLoopInfo.bIdx, tempLoopInfo.actS1Size, tempLoopInfo.actS2Size);
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
    if constexpr (LAYOUT_T == LI_LAYOUT::BSND) {
        if (tempLoopInfo.gS1LoopEnd == gS1SplitNum - 1 && constInfo.qSeqSize > tempLoopInfo.actS1Size) {
            tempLoopInfo.needDealActS1LessThanS1 = true;
        }
    }
}

template <typename LIT>
__aicore__ inline void IndexerCoarseScreenKernel<LIT>::CalcRunInfo(uint32_t loop,
                                                             uint32_t s2LoopIdx, IndexerCoarseScreenCommon::RunInfo &runInfo)
{
    runInfo.loop = loop;
    runInfo.bIdx = tempLoopInfo.bIdx;
    runInfo.gS1Idx = tempLoopInfo.gS1Idx;
    runInfo.s2Idx = s2LoopIdx;
    runInfo.bN2Idx = tempLoopInfo.bN2Idx;

    runInfo.actS1Size = tempLoopInfo.actS1Size;
    runInfo.actS2Size = tempLoopInfo.actS2Size;
    // 计算实际基本块size
    runInfo.actMBaseSize = tempLoopInfo.actMBaseSize;
    runInfo.actualSingleProcessSInnerSize = constInfo.s2BaseSize;
    uint32_t s2SplitNum = (tempLoopInfo.actS2Size + constInfo.s2BaseSize - 1) / constInfo.s2BaseSize;
    if (runInfo.s2Idx == s2SplitNum - 1) {
        runInfo.actualSingleProcessSInnerSize = tempLoopInfo.s2BasicSizeTail;
    }
    runInfo.actualSingleProcessSInnerSizeAlign =
        IndexerCoarseScreenCommon::Align((uint32_t)runInfo.actualSingleProcessSInnerSize, IndexerCoarseScreenCommon::ConstInfo::BUFFER_SIZE_BYTE_32B);

    runInfo.isFirstS2InnerLoop = s2LoopIdx == splitCoreInfo.s2Start;
    runInfo.isLastS2InnerLoop = s2LoopIdx == tempLoopInfo.s2LoopEnd;
    runInfo.isAllLoopEnd = (runInfo.bN2Idx == splitCoreInfo.bN2End) && (runInfo.gS1Idx == splitCoreInfo.gS1End) &&
                           (runInfo.s2Idx == splitCoreInfo.s2End);

    if (runInfo.isFirstS2InnerLoop) {
        uint64_t actualSeqQPrefixSum;
        uint64_t actualSeqKPrefixSum;
        if constexpr (LAYOUT_T == LI_LAYOUT::TND) {
            actualSeqQPrefixSum = (runInfo.bIdx <= 0) ? 0 : actualSeqLengthsGmQ.GetValue(runInfo.bIdx - 1);
            actualSeqKPrefixSum = (runInfo.bIdx <= 0) ? 0 : actualSeqLengthsGm.GetValue(runInfo.bIdx - 1);
        } else { // BSND
            actualSeqQPrefixSum = (runInfo.bIdx <= 0) ? 0 : runInfo.bIdx * constInfo.qSeqSize;
            actualSeqKPrefixSum = (runInfo.bIdx <= 0) ? 0 : runInfo.bIdx * constInfo.kSeqSize;
        }
        uint64_t tndBIdxOffset = actualSeqQPrefixSum * constInfo.qHeadNum * constInfo.headDim;
        uint64_t tndKeyBIdxOffset = actualSeqKPrefixSum * constInfo.kHeadNum * constInfo.headDim;
        // coarse 语义:Stage-2 源 = 代理 workspace(qBarGm/wBarGm),按请求 bIdx 索引
        queryCoreOffset = runInfo.bIdx * constInfo.qHeadNum * constInfo.headDim;
        keyCoreOffset = tndKeyBIdxOffset + runInfo.n2Idx * constInfo.headDim;
        // B,S1,N1(N2,G)
        weightsCoreOffset = runInfo.bIdx * constInfo.qHeadNum;
        // B,S1,N2,W8(行宽 = 8 对齐 outRowWidth,含有效 + -1 pad)
        indiceOutCoreOffset = (uint64_t)runInfo.bIdx * constInfo.kHeadNum * constInfo.outRowWidth +
                              (uint64_t)runInfo.n2Idx * constInfo.outRowWidth;
    }
    runInfo.tensorQueryOffset = queryCoreOffset;
    runInfo.tensorKeyOffset = keyCoreOffset + runInfo.s2Idx * constInfo.s2BaseSize * constInfo.kHeadNum
    * constInfo.headDim;
    runInfo.tensorWeightsOffset = weightsCoreOffset;
    runInfo.indiceOutOffset = indiceOutCoreOffset;
}

template <typename LIT>
__aicore__ inline void IndexerCoarseScreenKernel<LIT>::Process()
{
    if (usedCoreNum == 0) {
        // 没有计算任务，直接清理输出
        ProcessInvalid();
        return;
    }
    ProcessMain();
    ProcessDecode();
}

template <typename LIT>
__aicore__ inline void IndexerCoarseScreenKernel<LIT>::ProcessInvalid()
{
    if ASCEND_IS_AIV {
        uint32_t aivCoreNum = GetBlockNum() * 2; // 2 means c:v = 1:2
        // coarse 语义:输出 [K,1,outRowWidth],每请求仅 1 行
        uint64_t totalOutputSize =
            constInfo.batchSize * constInfo.kHeadNum * constInfo.outRowWidth;
        uint64_t singleCoreSize =
            IndexerCoarseScreenCommon::Align((totalOutputSize + aivCoreNum - 1) / aivCoreNum, GM_ALIGN_BYTES / sizeof(OUT_T));
        uint64_t baseSize = tmpBlockIdx * singleCoreSize;
        if (baseSize < totalOutputSize) {
            uint64_t dealSize =
                (baseSize + singleCoreSize <= totalOutputSize) ? singleCoreSize : totalOutputSize - baseSize;
            GlobalTensor<OUT_T> output = indiceOutGm[baseSize];
            AscendC::InitGlobalMemory(output, dealSize, constInfo.INVALID_IDX);
        }
    }
}

template <typename LIT>
__aicore__ inline void IndexerCoarseScreenKernel<LIT>::ProcessMain()
{
    if (aiCoreIdx >= usedCoreNum) {
        // 无任务核直接返回(同 lightning_indexer:任务检查最先,无任务核不参与任何同步)
        return;
    }

    if ASCEND_IS_AIV {
        // coarse 两段式编排:
        // Stage-1 组均值代理:本核对配对 AIC 的整个 b 区间 [bN2Start,bN2End] 逐请求
        // 池化出 q_bar/w_bar 写 workspace,完成后 PipeBarrier<PIPE_ALL>。
        // 再预置 syncV1C1×2 种子 flag(MTE2),既保证配对 AIC 首两轮 matmul 等到
        // 代理数据就绪,又供 AIC 尾部两轮 CrossCoreWaitFlag 收束(与生产 lightning_indexer 一致)。
        vectorService.PreprocessMean(splitCoreInfo.bN2Start, splitCoreInfo.bN2End);
        vectorService.AllocEventID();
        CrossCoreSetFlag<IndexerCoarseScreenCommon::ConstInfo::FIA_SYNC_MODE2, PIPE_MTE2>(constInfo.syncV1C1);
        CrossCoreSetFlag<IndexerCoarseScreenCommon::ConstInfo::FIA_SYNC_MODE2, PIPE_MTE2>(constInfo.syncV1C1);
    } else {
        matmulService.AllocEventID();
    }

    IndexerCoarseScreenCommon::RunInfo runInfo;
    uint32_t gloop = 0;
    for (uint32_t bN2LoopIdx = splitCoreInfo.bN2Start; bN2LoopIdx <= splitCoreInfo.bN2End; bN2LoopIdx++) {
        CalcGS1LoopParams(bN2LoopIdx);
        if (tempLoopInfo.curActSeqLenIsZero) {
            DealActSeqLenIsZero(tempLoopInfo.bIdx, tempLoopInfo.n2Idx, 0U);
            continue;
        }
        for (uint32_t gS1LoopIdx = splitCoreInfo.gS1Start; gS1LoopIdx <= tempLoopInfo.gS1LoopEnd; gS1LoopIdx++) {
            CalcS2LoopParams(bN2LoopIdx, gS1LoopIdx);
            for (int s2LoopIdx = splitCoreInfo.s2Start; s2LoopIdx <= tempLoopInfo.s2LoopEnd; s2LoopIdx++) {
                ProcessBaseBlock(gloop, s2LoopIdx, runInfo);
                ++gloop;
            }
            splitCoreInfo.s2Start = 0;
        }
        if (tempLoopInfo.needDealActS1LessThanS1) {
            DealActSeqLenIsZero(tempLoopInfo.bIdx, tempLoopInfo.n2Idx, tempLoopInfo.actS1Size);
        }
        splitCoreInfo.gS1Start = 0;
    }

    if ASCEND_IS_AIV {
        vectorService.FreeEventID();
    } else {
        matmulService.FreeEventID();
        CrossCoreWaitFlag(constInfo.syncV1C1);
        CrossCoreWaitFlag(constInfo.syncV1C1);
    }
}

template <typename LIT>
__aicore__ inline void IndexerCoarseScreenKernel<LIT>::ProcessBaseBlock(uint32_t loop,
                                                             uint64_t s2LoopIdx, IndexerCoarseScreenCommon::RunInfo &runInfo)
{
    CalcRunInfo(loop, s2LoopIdx, runInfo);
    if ASCEND_IS_AIC {
        CrossCoreWaitFlag(constInfo.syncV1C1);
        matmulService.ComputeMm1(runInfo);
        CrossCoreSetFlag<IndexerCoarseScreenCommon::ConstInfo::FIA_SYNC_MODE2, PIPE_FIX>(constInfo.syncC1V1);
    } else {
        CrossCoreWaitFlag(constInfo.syncC1V1);
        vectorService.ProcessVec(runInfo);
        CrossCoreSetFlag<IndexerCoarseScreenCommon::ConstInfo::FIA_SYNC_MODE2, PIPE_MTE2>(constInfo.syncV1C1);
    }
}

template <typename LIT>
__aicore__ inline void IndexerCoarseScreenKernel<LIT>::ProcessDecode()
{
    if ASCEND_IS_AIV {
        vectorService.InitLDBuffers(pipe);
        ICachePreLoad(LD_PREFETCH_LEN);
        SyncAll();
        if (splitCoreInfo.isLD) {
            vectorService.ProcessLD();
        }
    }
}
} // namespace LIKernel
#endif // INDEXER_COARSE_SCREEN_KERNEL_H
