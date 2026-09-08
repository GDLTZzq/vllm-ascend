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
 * \file indexer_coarse_screen_service_vector.h
 * \brief
 */
#ifndef INDEXER_COARSE_SCREEN_SERVICE_VECTOR_H
#define INDEXER_COARSE_SCREEN_SERVICE_VECTOR_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "../indexer_coarse_screen_common.h"
#include "indexer_coarse_screen_vector.h"

namespace LIKernel {
using namespace IndexerCoarseScreenCommon;
using namespace IndexerCoarseScreenServiceVec;
constexpr uint32_t BASE_TOPK = 2048;
constexpr uint32_t SPARSE_COUNT_4K = 4096;
constexpr uint32_t LD_PARAM_NUM = 16;
constexpr uint32_t EVENTID_V_TO_MTE2_PING = 0;
constexpr uint32_t EVENTID_V_TO_MTE2_PONG = 1;
constexpr uint32_t EVENTID_V_TO_MTE2_TMPUB = 2;

// 主模板：Q_T必选，W_T可选（默认void），无论W_T传什么，默认weightsType=Q_T
template<typename Q_T, typename W_T = void>
struct IndexerCoarseScreenTypeTraits {
    using weightsType = Q_T;   // 默认：weightsType绑定Q_T
};

// 偏特化1：固定第二个参数W_T=float，Q_T保留泛型
template<typename Q_T>
struct IndexerCoarseScreenTypeTraits<Q_T, float> {
    using weightsType = float;  // W_T=float时，强制weightsType为float
};

template <typename LIT>
class IndexerCoarseScreenServiceVector {
public:
    // =================================类型定义区=================================
    // 中间计算数据类型为float，高精度模式
    static constexpr bool DT_W_FLAG = LIT::weightsTypeFlag;
    using Q_T = typename LIT::queryType;
    static constexpr LI_LAYOUT LAYOUT_T = LIT::layout;
    using W_T = typename IndexerCoarseScreenTypeTraits<Q_T,
                                         typename std::conditional<DT_W_FLAG, float, void>::type>::weightsType;

    // MM输出数据类型, 当前只支持float
    using MM1_OUT_T = float;

    __aicore__ inline IndexerCoarseScreenServiceVector(){};
    __aicore__ inline void ProcessVec(const IndexerCoarseScreenCommon::RunInfo &info);
    __aicore__ inline void ProcessLD();
    __aicore__ inline void InitBuffers(TPipe *pipe);
    __aicore__ inline void InitParams(const struct IndexerCoarseScreenCommon::ConstInfo &constInfo,
                                      const IndexerCoarseScreenTilingData *__restrict tilingData);
    __aicore__ inline void InitVec1GlobalTensor(GlobalTensor<MM1_OUT_T> mm1ResGm, GlobalTensor<float> vec1ResGm,
                                                GlobalTensor<int64_t> vec1ParamGm, GlobalTensor<W_T> weightsGm,
                                                GlobalTensor<int32_t> indiceOutGm);
    // Stage-1 组均值代理池化:本核对 b 区间 [bStart,bEnd] 逐请求计算 q_bar/w_bar 写 workspace
    __aicore__ inline void InitPreprocessTensor(GlobalTensor<Q_T> queryGm, GlobalTensor<W_T> inputWeightsGm,
                                                GlobalTensor<W_T> rowWeightsGm, GlobalTensor<Q_T> qBarGm,
                                                GlobalTensor<W_T> wBarGm, GlobalTensor<uint32_t> actualSeqLengthsGmQ);
    __aicore__ inline void PreprocessMean(uint32_t bStart, uint32_t bEnd);
    __aicore__ inline void CleanInvalidOutput(int64_t invalidS1offset);
    // 输出行融合写(over2k needCopyOutGm,每请求恰一次,偶数 AIV):
    //   粗筛 top-min(L,4096) 前段(Extract idx 原序)++ 本地窗 union(尾项去重,自有恒新增,升序)++ -1 pad → 单次全行 CopyOut。
    //   topkSrc = globalTopkUb_[innerS1Idx*virTopK*2]((value,index) 交错,有效位前连续、尾 -1)。
    __aicore__ inline void FuseWindowRowOut(uint32_t bIdx, uint64_t gmRowOffset, LocalTensor<float> topkSrc, uint32_t L);
    // L==0(seq_lens==g_r,域空)请求:自有窗行 [0,g_r) 升序 + -1 pad 至 outRowWidth。
    __aicore__ inline void WriteZeroRow(uint32_t bIdx, uint64_t gmRowOffset);
    // 读 aslq cum 差分得本请求组宽 g_r(与 PreprocessMean 同取法,恒宽场景 = gMax)
    __aicore__ inline uint32_t GetGroupWidthFromCum(uint32_t bIdx);
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();
    __aicore__ inline void InitLDBuffers(TPipe *pipe);

protected:
    GlobalTensor<MM1_OUT_T> mm1ResGm;
    GlobalTensor<float> vec1ResGm;
    GlobalTensor<int64_t> vec1ParamGm;
    GlobalTensor<W_T> weightsGm;
    GlobalTensor<int32_t> indiceOutGm;
    // Stage-1 池化输入(query 原值 TND、逐行 weights、row_weights)与代理输出 workspace
    GlobalTensor<Q_T> queryGm;           // 输入 query [N,H,D] TND(Stage-1 池化读)
    GlobalTensor<W_T> inputWeightsGm;    // 输入 weights [N,H](Stage-1 池化读)
    GlobalTensor<W_T> rowWeightsGm;      // 输入 row_weights [R,g](Stage-1 池化读)
    GlobalTensor<Q_T> qBarGm;            // workspace [K,H,D] bf16 池化代理 query(Stage-2 cube 读)
    GlobalTensor<W_T> wBarGm;            // workspace [K,H] bf16 池化代理 weights(DoScale 读)
    GlobalTensor<uint32_t> actualSeqLengthsGmQ; // aslq_q cum(池化取组宽 g_r)
    // =================================常量区=================================

private:
    // ================================Local Buffer区====================================
    // queue
    TQue<QuePosition::VECOUT, 1> outQueue_;

    // tmp buff for vector
    TBuf<TPosition::VECCALC> sortOutBuf_;
    TBuf<TPosition::VECCALC> tmpBuf_;
    TBuf<TPosition::VECCALC> indexBuf_;
    TBuf<TPosition::VECCALC> reduceOutBuf_;
    TBuf<TPosition::VECCALC> brcBuf_;
    TBuf<TPosition::VECCALC> paramBuf_;

    // tmp buff for LD
    TBuf<> ldToBeMrgBuf_;
    TBuf<> ldTmpBuf_;
    TBuf<> ldOutValueBuf_;
    TBuf<> ldOutIdxBuf_;

    LocalTensor<float> tmpUb_;
    LocalTensor<int32_t> globalTopkIndice_;
    LocalTensor<float> globalTopkUb_;
    LocalTensor<float> SortedBasicBlock_;

    int32_t blockId_ = -1;
    // para for vector
    int32_t groupInner_ = 0;
    int32_t globalTopkNum_ = 0;
    int64_t blockS2StartIdx_ = 0;
    int32_t gSize_ = 0;
    int32_t kHeadNum_ = 0;
    int32_t s1BaseSize_ = 0;
    int32_t s2BaseSize_ = 0;

    // para for LD
    uint32_t mrgListNum_ = 4;
    uint32_t paramNum_ = 16;
    int32_t virTopK = 0;

    constexpr static uint32_t REDUCE_BANK_CONFLICT_OFFSETS = 256;
    constexpr static uint32_t REDUCE_BANK_CONFLICT_NUM = REDUCE_BANK_CONFLICT_OFFSETS / sizeof(float);
    // Stage-1 池化块参数
    constexpr static uint32_t PREPROCESS_ROWS_CHUNK = 16; // 每块加载的 query 行数(UB 上界)
    constexpr static uint32_t PREPROCESS_MAX_GROUP = 512; // 每请求组宽 g_r 上界(元组宽度)

    struct IndexerCoarseScreenCommon::ConstInfo constInfo_;
};

template <typename LIT>
__aicore__ inline void IndexerCoarseScreenServiceVector<LIT>::InitBuffers(TPipe *pipe)
{
    // CopyOut 段需求(单 outValueUb,Extract 形态): 值[0,offset) + 索引[offset,2offset) = 2*offset floats。
    //   non-Over2K: offset=virTopK=2048, copyLen<=2048 → 4096;Over2K: offset=copyOff, copyNum=2 → 4096。
    //   reduceCacheBuf(groupInner_*s2BaseSize_+offsets) 更大,outQueue_ 按其取 max,富余充足。
    // coarse:sparseCount=4096(over2k 用例) — offset=virTopK=4096, 2*offset=8192 floats=32KB。
    uint32_t outNeedBufSize = (BASE_TOPK * 2) * 2 * sizeof(float);
    if (constInfo_.isSparseCountOver2K) {
        int64_t copyOff = (constInfo_.sparseCount <= SPARSE_COUNT_4K)
                              ? constInfo_.sparseCount
                              : constInfo_.sparseCount / 2;
        outNeedBufSize = 2 * copyOff * sizeof(float);
    }
    uint32_t reduceCacheSize = REDUCE_BANK_CONFLICT_OFFSETS + groupInner_ * s2BaseSize_ * sizeof(float);
    outNeedBufSize = reduceCacheSize > outNeedBufSize ? reduceCacheSize : outNeedBufSize;
    virTopK = constInfo_.isSparseCountOver2K ? constInfo_.sparseCount : BASE_TOPK;

    pipe->InitBuffer(outQueue_, 1, outNeedBufSize);                                            // 32KB  extract
    // 68KB 在搬运cube核计算得到的结果和weight时，分成两块34KB，用于db；在mrgsort时，用作临时UB
    pipe->InitBuffer(tmpBuf_, (groupInner_ * s2BaseSize_ + s2BaseSize_) * 2 * sizeof(float));
    pipe->InitBuffer(sortOutBuf_, CeilDiv(s1BaseSize_, 2) * virTopK * 2 * sizeof(float));    // 64KB
    pipe->InitBuffer(indexBuf_, s2BaseSize_ * sizeof(int32_t));                                // 2KB
    // coarse:无 candidates 掩码链 → reduceOutBuf_ 收缩回生产形态 2×s2BaseSize_
    //   [0,V) 分数 + [V,2V) 索引(cols),与生产 lightning_indexer 逐位一致。
    pipe->InitBuffer(reduceOutBuf_, s2BaseSize_ * 2 * sizeof(float));                          // 4KB
    pipe->InitBuffer(brcBuf_, groupInner_ * 8 * sizeof(float));
    pipe->InitBuffer(paramBuf_, LD_PARAM_NUM * sizeof(int64_t));

    tmpUb_ = tmpBuf_.Get<float>();
    globalTopkIndice_ = indexBuf_.Get<int32_t>();
    globalTopkUb_ = sortOutBuf_.Get<float>();
    SortedBasicBlock_ = globalTopkUb_[virTopK * 2 * 2];
    globalTopkNum_ = 0;

    // 基本块执行前初始化UB和GM
    // step1. 初始化一个有序索引 0 - s2BaseSize_
    ArithProgression<int32_t>(globalTopkIndice_, 0, 1, s2BaseSize_);
    // step2. globalTopkUb_ [CeilDiv(s1BaseSize_, 2), BASE_TOPK, 2]   -inf,-1
    InitSortOutBuf(globalTopkUb_, CeilDiv(s1BaseSize_, 2) * virTopK * 2);

    // step3. 初始化vec1ParamGm，是否进行LD的标志位设为-1(needFd=-1)
    // vec1ResIn32Gm = [aic, 2, s1BaseSize_, 16] int32
    // ws清零 [needFd, s2AcSeq, s2Start, s2End, isS2End, bn2idx, s1Idx, ......]
    LocalTensor<float> tmpBuff = outQueue_.AllocTensor<float>();
    Duplicate(tmpBuff.template ReinterpretCast<int32_t>(), -1, 2 * (s1BaseSize_ / 2) * paramNum_ * 2);
    outQueue_.EnQue<float>(tmpBuff);
    tmpBuff = outQueue_.DeQue<float>();
    int64_t wsInfoOffset = (blockId_ / 2) * s1BaseSize_ * 2 * paramNum_ +      // 2个AIV共同地址偏移
                           (blockId_ % 2) * (s1BaseSize_ / 2) * 2 * paramNum_; // 每个AIV的地址偏移，S1方向
    DataCopyPad(vec1ParamGm[wsInfoOffset], tmpBuff.template ReinterpretCast<int64_t>(),
                {1, static_cast<uint16_t>((s1BaseSize_ / 2) * 2 * paramNum_ * sizeof(int64_t)), 0, 0});
    outQueue_.FreeTensor(tmpBuff);
}

template <typename LIT>
__aicore__ inline void IndexerCoarseScreenServiceVector<LIT>::InitLDBuffers(TPipe *pipe)
{
    pipe->Reset();
    pipe->InitBuffer(ldToBeMrgBuf_, 2 * BASE_TOPK * mrgListNum_ * sizeof(float)); // 2：value + index
    pipe->InitBuffer(ldTmpBuf_, 2 * BASE_TOPK * mrgListNum_ * sizeof(float));     // 2：value + index
    pipe->InitBuffer(ldOutValueBuf_, BASE_TOPK * sizeof(float));
    pipe->InitBuffer(ldOutIdxBuf_, BASE_TOPK * sizeof(int32_t));
}

template <typename LIT>
__aicore__ inline void IndexerCoarseScreenServiceVector<LIT>::InitParams(const struct IndexerCoarseScreenCommon::ConstInfo &constInfo,
                                                 const IndexerCoarseScreenTilingData *__restrict tilingData)
{
    this->constInfo_ = constInfo;
    blockS2StartIdx_ = 0;
    gSize_ = constInfo.gSize;
    // define N2 para
    kHeadNum_ = constInfo.kHeadNum;
    // define MMBase para
    s1BaseSize_ = constInfo.s1BaseSize;
    s2BaseSize_ = constInfo.s2BaseSize;

    // group ub 切分因子当前按照UB空间强制为16
    groupInner_ = 16;

    blockId_ = GetBlockIdx();
}

template <typename LIT>
__aicore__ inline void
IndexerCoarseScreenServiceVector<LIT>::InitVec1GlobalTensor(GlobalTensor<MM1_OUT_T> mm1ResGm,
                                    GlobalTensor<float> vec1ResGm,
                                    GlobalTensor<int64_t> vec1ParamGm, GlobalTensor<W_T> weightsGm,
                                    GlobalTensor<int32_t> indiceOutGm)
{
    this->mm1ResGm = mm1ResGm;
    this->vec1ResGm = vec1ResGm;
    this->vec1ParamGm = vec1ParamGm;
    this->weightsGm = weightsGm;
    this->indiceOutGm = indiceOutGm;
}

template <typename LIT>
__aicore__ inline void
IndexerCoarseScreenServiceVector<LIT>::InitPreprocessTensor(GlobalTensor<Q_T> queryGm,
                                    GlobalTensor<W_T> inputWeightsGm,
                                    GlobalTensor<W_T> rowWeightsGm, GlobalTensor<Q_T> qBarGm,
                                    GlobalTensor<W_T> wBarGm, GlobalTensor<uint32_t> actualSeqLengthsGmQ)
{
    this->queryGm = queryGm;
    this->inputWeightsGm = inputWeightsGm;
    this->rowWeightsGm = rowWeightsGm;
    this->qBarGm = qBarGm;
    this->wBarGm = wBarGm;
    this->actualSeqLengthsGmQ = actualSeqLengthsGmQ;
}

template <typename LIT>
__aicore__ inline void IndexerCoarseScreenServiceVector<LIT>::AllocEventID()
{
    SetFlag<HardEvent::V_MTE2>(EVENTID_V_TO_MTE2_PING);
    SetFlag<HardEvent::V_MTE2>(EVENTID_V_TO_MTE2_PONG);
    SetFlag<HardEvent::V_MTE2>(EVENTID_V_TO_MTE2_TMPUB);
}

template <typename LIT>
__aicore__ inline void IndexerCoarseScreenServiceVector<LIT>::FreeEventID()
{
    WaitFlag<HardEvent::V_MTE2>(EVENTID_V_TO_MTE2_PING);
    WaitFlag<HardEvent::V_MTE2>(EVENTID_V_TO_MTE2_PONG);
    WaitFlag<HardEvent::V_MTE2>(EVENTID_V_TO_MTE2_TMPUB);
}

template <typename LIT>
__aicore__ inline void IndexerCoarseScreenServiceVector<LIT>::CleanInvalidOutput(int64_t invalidS1offset)
{
    // init -1 and copy to output(整行 outRowWidth,coarse 行宽)
    LocalTensor<float> valueULocal = outQueue_.AllocTensor<float>();
    LocalTensor<int32_t> idxULocal1 = valueULocal.template ReinterpretCast<int32_t>();
    Duplicate(idxULocal1, constInfo_.INVALID_IDX, constInfo_.outRowWidth);
    outQueue_.EnQue<float>(valueULocal);
    valueULocal = outQueue_.DeQue<float>();
    IndexerCoarseScreenServiceVec::CopyOut(indiceOutGm[invalidS1offset], idxULocal1, constInfo_.outRowWidth);
    outQueue_.FreeTensor(valueULocal);
}

template <typename LIT>
__aicore__ inline uint32_t IndexerCoarseScreenServiceVector<LIT>::GetGroupWidthFromCum(uint32_t bIdx)
{
    // 与 PreprocessMean 同取法:组宽 = aslq cum 差分;防御型取 cumDiff(1,gMax] 内值。
    // 恒宽 decode 场景 cum 差分恒 = row_weights.dim1 = gMax。
    uint32_t gR = static_cast<uint32_t>(constInfo_.gMax);
    if (constInfo_.actualLenQDims != 0) {
        uint32_t cumPrev = (bIdx == 0) ? 0 : actualSeqLengthsGmQ.GetValue(bIdx - 1);
        uint32_t cumDiff = actualSeqLengthsGmQ.GetValue(bIdx) - cumPrev;
        if (cumDiff > 0 && cumDiff < gR) {
            gR = cumDiff;
        }
    }
    return gR;
}

// 行缓冲在 tmpUb_ 内(int 别名):粗筛 idx 半区起点 = virTopK(Extract 的 idx 落点),故行 int 基址 = virTopK。
//   [0, virTopK)        Extract value 半区(后复用为 coarse int→float 区)
//   [virTopK, 2*virTopK)  Extract idx 半区 = 粗筛 top-min(L,4096) 实值 + 尾部 -1
//   行区 int [virTopK, virTopK + outRowWidth)(≤ ~4128 ints),与上面余量不冲突
//   [3*virTopK, 4*virTopK)  presence 差分/sqrt 暂存(float)
//   [4*virTopK, ...)        ReduceMax 结果(小块)
template <typename LIT>
__aicore__ inline void IndexerCoarseScreenServiceVector<LIT>::FuseWindowRowOut(uint32_t bIdx, uint64_t gmRowOffset,
                                                                LocalTensor<float> topkSrc, uint32_t L)
{
    const uint32_t span = static_cast<uint32_t>(virTopK);      // Extract 半宽 = 4096
    const uint32_t scratchF = 3U * span;                        // float 暂存区
    const uint32_t reduceF = 4U * span;                         // reduce 结果区
    const uint32_t gR = GetGroupWidthFromCum(bIdx);
    // 域 [0,L) 空:仅自有窗 [0,gR)(与 WriteZeroRow 等价,防御;正常 L==0 走 DealActSeqLenIsZero)
    if (L == 0) {
        WriteZeroRow(bIdx, gmRowOffset);
        return;
    }
    const uint32_t c = (L > span) ? span : L;                   // 粗筛实值数 = min(L, 4096)

    LocalTensor<float> ubF = tmpUb_;
    LocalTensor<int32_t> rowI = ubF.template ReinterpretCast<int32_t>();
    // Extract:value→ubF[0,span),idx→int [span, 2span),与生产 CopyOut 路径同构
    LocalTensor<uint32_t> dstIdx = ubF[span].template ReinterpretCast<uint32_t>();
    Extract(ubF, dstIdx, topkSrc, span / 32);

    uint32_t st = span + c;                                     // 行窗口追加起点(int 下标)
    if (L > span) {
        // 尾项 [L-gR+1, L)(≤ gR-1)逐个判粗筛成员:粗筛 idx(int)→float,平方差 ReduceMax==0 判在集
        LocalTensor<float> coarseF = ubF;                       // [0,span),复用 value-half
        LocalTensor<int32_t> coarseIdxInt = rowI[span];         // aarch64 严格模式:提命名变量再作 Cast 源
        Cast(coarseF, coarseIdxInt, RoundMode::CAST_NONE, span);
        LocalTensor<float> diffF = ubF[scratchF];
        LocalTensor<float> redUb = ubF[reduceF];
        PipeBarrier<PIPE_V>();
        for (uint32_t p = L - gR + 1; p < L; p++) {
            Adds(diffF, coarseF, -1.0f * static_cast<float>(p), span);
            PipeBarrier<PIPE_V>();
            Mul(diffF, diffF, diffF, span);
            PipeBarrier<PIPE_V>();
            ReduceMax(redUb, diffF, diffF, span);
            PipeBarrier<PIPE_V>();
            if (redUb.GetValue(0) != 0.0f) {
                rowI.SetValue(st, static_cast<int32_t>(p));
                st++;
            }
        }
    }
    // 自有 token [L, L+gR) 不入域、恒新增,升序追加
    for (uint32_t j = 0; j < gR; j++) {
        rowI.SetValue(st, static_cast<int32_t>(L + j));
        st++;
    }
    // -1 pad 至行尾 outRowWidth(8 对齐整段 Duplicate + ≤7 尾标量)
    const uint32_t rowEnd = span + constInfo_.outRowWidth;
    uint32_t padStart = (st + 7U) & ~7U;
    for (uint32_t i = st; i < padStart && i < rowEnd; i++) {
        rowI.SetValue(i, -1);
    }
    if (padStart < rowEnd) {
        LocalTensor<int32_t> padUb = rowI[padStart];
        Duplicate(padUb, -1, rowEnd - padStart);
    }
    PipeBarrier<PIPE_V>();
    LocalTensor<int32_t> rowSrc = rowI[span];
    IndexerCoarseScreenServiceVec::CopyOut(indiceOutGm[gmRowOffset], rowSrc, constInfo_.outRowWidth);
    // 2026-09-08 修复:全行 DataCopyPad(MTE3) 读本缓冲后,tmpUb_ 下一请求/子片复用(V/MTE)须等其完成
    AscendC::PipeBarrier<PIPE_ALL>();
}

// L==0 请求行:自有窗 [0,gR)(= 请求全部 token,域空),后 -1 pad 至 outRowWidth。
// 双 AIV 冗余写同内容(值一致),不引入事件;行缓冲同上在 tmpUb_ 内。
template <typename LIT>
__aicore__ inline void IndexerCoarseScreenServiceVector<LIT>::WriteZeroRow(uint32_t bIdx, uint64_t gmRowOffset)
{
    const uint32_t span = static_cast<uint32_t>(virTopK);
    const uint32_t gR = GetGroupWidthFromCum(bIdx);
    LocalTensor<float> ubF = tmpUb_;
    LocalTensor<int32_t> rowI = ubF.template ReinterpretCast<int32_t>();
    uint32_t st = span;
    for (uint32_t j = 0; j < gR; j++) {
        rowI.SetValue(st, static_cast<int32_t>(j));
        st++;
    }
    const uint32_t rowEnd = span + constInfo_.outRowWidth;
    uint32_t padStart = (st + 7U) & ~7U;
    for (uint32_t i = st; i < padStart && i < rowEnd; i++) {
        rowI.SetValue(i, -1);
    }
    if (padStart < rowEnd) {
        LocalTensor<int32_t> padUb = rowI[padStart];
        Duplicate(padUb, -1, rowEnd - padStart);
    }
    PipeBarrier<PIPE_V>();
    LocalTensor<int32_t> rowSrc = rowI[span];
    IndexerCoarseScreenServiceVec::CopyOut(indiceOutGm[gmRowOffset], rowSrc, constInfo_.outRowWidth);
    AscendC::PipeBarrier<PIPE_ALL>();
}

// Stage-1 组均值代理池化:对本核对 b 区间 [bStart,bEnd] 逐请求计算
//   q_bar[r,h] = Σ_i rw[r,i]·query[cum[r-1]+i,h,:] / Σ_i rw[r,i]  → qBarGm [K,H,D] bf16
//   w_bar[r,h] = Σ_i rw[r,i]·weights[cum[r-1]+i,h]     / Σ_i rw[r,i]  → wBarGm [K,H] bf16
// 精度:fp32 累加、按 rw 逐行加权、乘以 1/Σrw、bf16 CAST_ROUND —— 匹配 torch .mean(dim=1)。
// 约定:row_weights 行宽 = g_r(cum 差分),PIVOT decode 组宽恒定,正确;组宽>PREPROCESS_MAX_GROUP 越界。
// 幂等:同核对两个 AIV 各自全量池化 [bStart,bEnd](冗余写,值一致)。
template <typename LIT>
__aicore__ inline void IndexerCoarseScreenServiceVector<LIT>::PreprocessMean(uint32_t bStart, uint32_t bEnd)
{
    const int32_t headDim = static_cast<int32_t>(constInfo_.headDim);
    const int32_t gSize = static_cast<int32_t>(constInfo_.gSize);

    // tmpUb_ 划段(池化先于主流水,68KB 富余;仅写 tmpUb_,不动 globalTopkIndice_/globalTopkUb_)
    uint32_t f = 0;
    LocalTensor<float> qAccUb = tmpUb_[f];
    f += static_cast<uint32_t>(headDim);
    LocalTensor<float> invUb = tmpUb_[f];
    f += 8;
    LocalTensor<float> wF32Ub = tmpUb_[f];
    f += PREPROCESS_MAX_GROUP;
    LocalTensor<float> wBarUb = tmpUb_[f];
    f += 512; // H 上界
    LocalTensor<float> rowUbF32 = tmpUb_[f];
    f += PREPROCESS_ROWS_CHUNK * static_cast<uint32_t>(headDim);
    // Q_T(2B)缓冲接在 float 区之后(字节偏移 f*4)
    int64_t qOff = static_cast<int64_t>(f) * static_cast<int64_t>(sizeof(float));
    LocalTensor<Q_T> rowUbBf16 = tmpUb_.template ReinterpretCast<Q_T>()[qOff / static_cast<int64_t>(sizeof(Q_T))];
    LocalTensor<Q_T> wRowBf16 = rowUbBf16[PREPROCESS_ROWS_CHUNK * static_cast<uint32_t>(headDim)];
    LocalTensor<Q_T> qBarBf16 = wRowBf16[PREPROCESS_MAX_GROUP];
    LocalTensor<Q_T> wBarOutBf16 = qBarBf16[static_cast<uint32_t>(headDim)];

    AscendC::DataCopyPadExtParams<Q_T> pad{false, 0, 0, 0};
    for (uint32_t r = bStart; r <= bEnd; r++) {
        uint32_t cumPrev = (r == 0) ? 0 : actualSeqLengthsGmQ.GetValue(r - 1);
        uint32_t cumR = actualSeqLengthsGmQ.GetValue(r);
        int32_t gR = static_cast<int32_t>(cumR - cumPrev);
        if (gR <= 0) {
            // 组宽为 0:代理写零,避免 cube 读到垃圾(请求仍可能被主流水处理)
            Duplicate(qBarBf16.template ReinterpretCast<int32_t>(), 0, headDim * static_cast<int32_t>(sizeof(Q_T) / 2));
            Duplicate(wBarOutBf16.template ReinterpretCast<int32_t>(), 0, gSize * static_cast<int32_t>(sizeof(Q_T) / 2));
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::DataCopyPad(qBarGm[static_cast<uint64_t>(r) * gSize * headDim], qBarBf16,
                                 {1, static_cast<uint16_t>(headDim * gSize * sizeof(Q_T)), 0, 0}, pad);
            AscendC::DataCopyPad(wBarGm[static_cast<uint64_t>(r) * gSize], wBarOutBf16,
                                 {1, static_cast<uint16_t>(gSize * sizeof(W_T)), 0, 0}, pad);
            AscendC::PipeBarrier<PIPE_ALL>();
            continue;
        }
        if (gR > static_cast<int32_t>(PREPROCESS_MAX_GROUP)) {
            gR = static_cast<int32_t>(PREPROCESS_MAX_GROUP); // 防御:超界截断(数据错误优于越界)
        }

        // 载入本请求 row_weights 行(行宽按 g_r,行首 r*g_r)
        AscendC::DataCopy(wRowBf16, rowWeightsGm[static_cast<uint64_t>(r) * gR], gR);
        AscendC::PipeBarrier<PIPE_MTE2>();
        AscendC::Cast(wF32Ub, wRowBf16, RoundMode::CAST_NONE, gR);
        AscendC::PipeBarrier<PIPE_V>();

        // totalW = Σ rw[r,i](fp32 标量),inv = 1/totalW(向量 Div,避免标量除法位差)
        float totalW = 0.0f;
        for (int32_t i = 0; i < gR; i++) {
            totalW += wF32Ub.GetValue(i);
        }
        if (totalW == 0.0f) {
            totalW = 1.0f;
        }
        AscendC::Duplicate(invUb, 1.0f, 1);
        AscendC::PipeBarrier<PIPE_V>();
        LocalTensor<float> totalWUb = invUb[1];
        totalWUb.SetValue(0, totalW);
        AscendC::Div(invUb, invUb, totalWUb, 1);
        AscendC::PipeBarrier<PIPE_V>();
        float invW = invUb.GetValue(0);

        for (int32_t h = 0; h < gSize; h++) {
            // q_bar:逐组行加权平均(D fp32 累加)
            AscendC::Duplicate(qAccUb, 0.0f, headDim);
            AscendC::PipeBarrier<PIPE_V>();
            for (int32_t chunk = 0; chunk < gR; chunk += PREPROCESS_ROWS_CHUNK) {
                int32_t rows = (chunk + PREPROCESS_ROWS_CHUNK > gR) ? gR - chunk : PREPROCESS_ROWS_CHUNK;
                // query[cumPrev+i, h, :],i∈[chunk,chunk+rows):行距 H*D,行宽 D
                uint64_t srcOff = (static_cast<uint64_t>(cumPrev + chunk) * gSize + h) * headDim;
                uint32_t srcStride =
                    (static_cast<uint32_t>(gSize) * headDim - headDim) * sizeof(Q_T) / 32;
                AscendC::DataCopyPad(rowUbBf16, queryGm[srcOff],
                                     {static_cast<uint16_t>(rows), static_cast<uint16_t>(headDim * sizeof(Q_T)),
                                      static_cast<uint16_t>(srcStride), 0, 0},
                                     pad);
                AscendC::PipeBarrier<PIPE_MTE2>();
                AscendC::Cast(rowUbF32, rowUbBf16, RoundMode::CAST_NONE, rows * headDim);
                AscendC::PipeBarrier<PIPE_V>();
                for (int32_t j = 0; j < rows; j++) {
                    LocalTensor<float> rowSrc = rowUbF32[j * headDim];
                    AscendC::Muls(rowSrc, rowSrc, wF32Ub.GetValue(chunk + j), headDim);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Add(qAccUb, qAccUb, rowSrc, headDim);
                    AscendC::PipeBarrier<PIPE_V>();
                }
            }
            AscendC::Muls(qAccUb, qAccUb, invW, headDim);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(qBarBf16, qAccUb, RoundMode::CAST_ROUND, headDim);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::DataCopyPad(qBarGm[static_cast<uint64_t>(r) * gSize * headDim + static_cast<uint64_t>(h) * headDim],
                                 qBarBf16, {1, static_cast<uint16_t>(headDim * sizeof(Q_T)), 0, 0}, pad);
            AscendC::PipeBarrier<PIPE_MTE3>();

            // w_bar:逐头标量加权平均(g_r*H 次 scalar 读,组宽小,开销可忽略)
            float wAcc = 0.0f;
            for (int32_t i = 0; i < gR; i++) {
                wAcc += wF32Ub.GetValue(i) *
                        static_cast<float>(inputWeightsGm.GetValue(
                            static_cast<uint64_t>(cumPrev + i) * gSize + h));
            }
            wBarUb[h].SetValue(0, wAcc * invW);
            AscendC::PipeBarrier<PIPE_V>();
        }
        AscendC::Cast(wBarOutBf16, wBarUb, RoundMode::CAST_ROUND, gSize);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::DataCopyPad(wBarGm[static_cast<uint64_t>(r) * gSize], wBarOutBf16,
                             {1, static_cast<uint16_t>(gSize * sizeof(W_T)), 0, 0}, pad);
        AscendC::PipeBarrier<PIPE_ALL>();
    }
}

template <typename LIT>
__aicore__ inline void IndexerCoarseScreenServiceVector<LIT>::ProcessVec(const IndexerCoarseScreenCommon::RunInfo &info)
{
    int32_t cuBaseS1Idx = info.gS1Idx * s1BaseSize_;
    int32_t cuBaseS2Idx = info.s2Idx * s2BaseSize_;

    // 计算基本块基地址偏移 偶数循环 -> 0 + aic_offset  奇数循环 -> 512*512 + aic_offset
    int64_t mmGmOffset = (info.loop % 2) * (constInfo_.mBaseSizeAlign * s2BaseSize_);
    // (B,S1,N1,1);(T,N1,1) -> (B,S1,N2,G,1) 当前只切分到S1轴
    int64_t weightGmOffset = info.tensorWeightsOffset + cuBaseS1Idx * kHeadNum_ * gSize_;

    PipeBarrier<PIPE_V>();
    // cuS1BeginIdxPerAiv: 每个AIV的S1起始偏移
    int32_t cuS1BeginIdxPerAiv = cuBaseS1Idx;
    int32_t cuS1ProcNum =
        cuS1BeginIdxPerAiv + s1BaseSize_ > info.actS1Size ? info.actS1Size % s1BaseSize_ : s1BaseSize_;
    // cuS1ProcNumPerAiv: 每个AIv的S1计算量
    int32_t cuS1ProcNumPerAiv = blockId_ % 2 == 0 ? CeilDiv(cuS1ProcNum, 2) : (cuS1ProcNum / 2);
    cuS1BeginIdxPerAiv += (blockId_ % 2) * CeilDiv(cuS1ProcNum, 2);

    // 基本块基地址偏移奇数核加一个S1地址偏移
    weightGmOffset += (blockId_ % 2) * CeilDiv(cuS1ProcNum, 2) * kHeadNum_ * gSize_;
    mmGmOffset += (blockId_ % 2) * CeilDiv(cuS1ProcNum, 2) * gSize_ * info.actualSingleProcessSInnerSizeAlign;

    // cut G
    int32_t outerG = CeilDiv(gSize_, groupInner_);

    // 非首个基本块, M(S1)轴发生切换需要初始化
    if (info.loop != 0 && info.s2Idx == 0) {
        // globalTopkUb_ value,index=-inf,-1
        InitSortOutBuf(globalTopkUb_, CeilDiv(s1BaseSize_, 2) * virTopK * 2);
        blockS2StartIdx_ = 0;
    } else if (info.loop == 0) {
        blockS2StartIdx_ = info.s2Idx;
    }
    // cuRealAcSeq: 当前基本块S1对应的AcSeq
    int32_t cuRealAcSeq = info.actS2Size;
    if (constInfo_.attenMaskFlag) {
        // attenMask true场景
        cuRealAcSeq = info.actS2Size - (info.actS1Size - cuS1BeginIdxPerAiv);
    }
    LocalTensor<float> reduceOutBuff = reduceOutBuf_.Get<float>();
    LocalTensor<float> brcBuf = brcBuf_.Get<float>();
    // LD输出S1方向偏移，保证2个Vector输出的内容连续
    uint32_t ldS1Offset = (blockId_ % 2 == 0) ? s1BaseSize_ / 2 - cuS1ProcNumPerAiv : 0;
    for (int innerS1Idx = 0; innerS1Idx < cuS1ProcNumPerAiv; innerS1Idx++) {
        if (constInfo_.attenMaskFlag) {
            cuRealAcSeq += 1;
        }
        int32_t cuS2Len = cuBaseS2Idx + s2BaseSize_ >= cuRealAcSeq ? cuRealAcSeq - cuBaseS2Idx : s2BaseSize_;
        int32_t cuS1Idx = cuS1BeginIdxPerAiv + innerS1Idx;
        if (cuRealAcSeq > 0 && cuS2Len > 0) {
            int32_t cuS2LenVecAlign = CeilDiv(cuS2Len, s2BaseSize_) * s2BaseSize_;
            int32_t mmUbStride = (cuS2LenVecAlign - info.actualSingleProcessSInnerSizeAlign) / B32_BLOCK_ALIGN_NUM;
            LocalTensor<float> reduceOutInner = reduceOutBuff[s2BaseSize_];
            PipeBarrier<PIPE_V>();
            LocalTensor<float> reduceCacheBuf = outQueue_.AllocTensor<float>();
            if (constInfo_.isSparseCountOver2K) {
                WaitFlag<HardEvent::V_MTE2>(EVENTID_V_TO_MTE2_TMPUB);
            }
            for (int outerGidx = 0; outerGidx < outerG; outerGidx++) {
                int32_t procGnum = outerGidx != outerG - 1 ? groupInner_ : gSize_ - outerGidx * groupInner_;

                int32_t pingpong = outerGidx % 2;
                LocalTensor<float> dbTmpUb = tmpUb_[pingpong * (groupInner_ * s2BaseSize_ + s2BaseSize_)];
                LocalTensor<float> weightsInUb = dbTmpUb[procGnum * s2BaseSize_];
                WaitFlag<HardEvent::V_MTE2>(pingpong);
                LocalTensor<W_T> weightsInTUb = weightsInUb.template ReinterpretCast<W_T>();
                if constexpr (!IsSameType<W_T, float>::value) {
                    weightsInTUb = weightsInTUb[groupInner_];
                }
                int64_t mmGmAllOffet = mmGmOffset + innerS1Idx * gSize_ * info.actualSingleProcessSInnerSizeAlign +
                                       outerGidx * groupInner_ * info.actualSingleProcessSInnerSizeAlign;
                int64_t weightGmAllOffset = weightGmOffset + innerS1Idx * gSize_ + outerGidx * groupInner_;

                IndexerCoarseScreenServiceVec::CopyIn(dbTmpUb, weightsInTUb, mm1ResGm, weightsGm, mmGmAllOffet, weightGmAllOffset,
                                     procGnum, info.actualSingleProcessSInnerSizeAlign, mmUbStride);

                SetFlag<HardEvent::MTE2_V>(pingpong);
                WaitFlag<HardEvent::MTE2_V>(pingpong);
                IndexerCoarseScreenServiceVec::DoScale(reduceCacheBuf[REDUCE_BANK_CONFLICT_NUM], dbTmpUb, weightsInUb, weightsInTUb,
                                      brcBuf, procGnum, s2BaseSize_, outerGidx);
                // confused reduceOp in DoScale
                // neednot use IndexerCoarseScreenServiceVec::doReduce(mmInUb, reduceOutInner, procGnum, (s2BaseSize_+8));
                SetFlag<HardEvent::V_MTE2>(pingpong);
            }

            int32_t gRedCnt = groupInner_ > gSize_ ? gSize_ : groupInner_;
            bool isS2End = cuBaseS2Idx + s2BaseSize_ >= cuRealAcSeq;
            IndexerCoarseScreenServiceVec::DoReduce(reduceCacheBuf[REDUCE_BANK_CONFLICT_NUM], reduceOutInner, gRedCnt, s2BaseSize_);
            outQueue_.FreeTensor(reduceCacheBuf);

            // 无 candidates 掩码链 → 生产形态掩码:全宽预填 NEG_INF,[0,V) 分数只写一次、
            // [V,2V) 索引只写一次(cols + 尾对齐 -1),与生产 lightning_indexer 逐位一致。
            LocalTensor<float> sortScoreUb = reduceOutBuff;
            LocalTensor<float> sortIndiceUb = reduceOutBuff[cuS2LenVecAlign];
            LocalTensor<int32_t> scoreI32 = sortScoreUb.template ReinterpretCast<int32_t>();
            LocalTensor<int32_t> sortIndiceUbInt = sortIndiceUb.template ReinterpretCast<int32_t>();
            Duplicate(scoreI32, IndexerCoarseScreenServiceVec::NEG_INF, cuS2LenVecAlign);
            PipeBarrier<PIPE_V>();
            Adds(sortScoreUb, reduceOutInner, 0.0f, cuS2Len);
            PipeBarrier<PIPE_V>();
            if (cuS2LenVecAlign != cuS2Len) {
                Duplicate(sortIndiceUbInt, -1, cuS2LenVecAlign);
            }
            PipeBarrier<PIPE_V>();
            Adds(sortIndiceUbInt, globalTopkIndice_, static_cast<int32_t>(cuBaseS2Idx), cuS2Len);
            // 进 sort 前统一同步:reduceOutBuff 写(V/MTE)全部落定后才被排序读取。
            AscendC::PipeBarrier<PIPE_ALL>();

            LocalTensor<float> tmpSortBuf = outQueue_.AllocTensor<float>();
            if (info.actS1Size > 4 || constInfo_.isSparseCountOver2K || cuS2Len == s2BaseSize_) {
                // info.actS1Size > 4 则单个vector核内处理的 s1>2，缓存方案无法处理
                if (constInfo_.isSparseCountOver2K) {
                    // 2026-08-31 v11 根因修复: over2k 归并只用 2-list。双累积
                    //   acc_U(排名1-2048)+acc_L(排名2049-4096)。每 chunk SortAll(512) 后两次
                    //   2-list 归并(mrgDstNum=virTopK/2=2048≤3072, 永不进 3-segment):
                    //     MergeSort(acc_U, 2048, chunk, len, tmpUb_): 被丢弃的 len 个最小对留在
                    //       tmpUb_[virTopK, virTopK+2*len)(MrgSort 全量输出, DataCopy 只拷回前 2048)
                    //     MergeSort(acc_L, 2048, tmpUb_[virTopK], len, tmpUb_[virTopK+2*len])
                    //   输出 = acc_U+acc_L 拼接 == top-4096, 与旧单次 4096 归并逐位一致。
                    SortAll(reduceOutBuff, tmpSortBuf, cuS2LenVecAlign); // 整块 512 排序(probe prod 同款, 实证可靠)
                    PipeBarrier<PIPE_V>();
                    IndexerCoarseScreenServiceVec::MergeSort(globalTopkUb_[innerS1Idx * virTopK * 2], virTopK / 2,
                                            reduceOutBuff, cuS2LenVecAlign, tmpUb_);
                    // 2026-09-01 aarch64 原生工具链(严格模式)拒收 LocalTensor::operator[] 临时量作
                    //   非 const 左值引用形参(mrgSrc/tmpTensor): 先提命名变量。
                    LocalTensor<float> ubTail = tmpUb_[virTopK];
                    LocalTensor<float> ubScratch = tmpUb_[virTopK + 2 * cuS2LenVecAlign];
                    IndexerCoarseScreenServiceVec::MergeSort(globalTopkUb_[innerS1Idx * virTopK * 2 + virTopK], virTopK / 2,
                                            ubTail, cuS2LenVecAlign, ubScratch);
                } else if (cuS2LenVecAlign == s2BaseSize_) {
                    IndexerCoarseScreenServiceVec::SortAll(reduceOutBuff, tmpSortBuf, cuS2LenVecAlign);
                    PipeBarrier<PIPE_V>();
                    IndexerCoarseScreenServiceVec::MergeSort(globalTopkUb_[innerS1Idx * virTopK * 2], virTopK, reduceOutBuff,
                                            cuS2LenVecAlign, tmpSortBuf);
                } else {
                    IndexerCoarseScreenServiceVec::SortAll(reduceOutBuff, tmpSortBuf,
                                          cuS2LenVecAlign); //  cuS2LenVecAlign <= s2BaseSize_, fill -inf
                    PipeBarrier<PIPE_V>();
                    LocalTensor<float> UbTmpSort = constInfo_.isSparseCountOver2K ? tmpUb_ : tmpSortBuf;
                    IndexerCoarseScreenServiceVec::MergeSort(globalTopkUb_[innerS1Idx * virTopK * 2], virTopK, reduceOutBuff,
                                            cuS2LenVecAlign, UbTmpSort);
                }
            } else {
                int64_t globalTopkUbCacheIdx = (info.s2Idx - blockS2StartIdx_) % 4;
                Sort<float, true>(
                    SortedBasicBlock_[innerS1Idx * BASE_TOPK * 2 + globalTopkUbCacheIdx * s2BaseSize_ * 2],
                    reduceOutBuff, sortIndiceUbInt.template ReinterpretCast<uint32_t>(), tmpSortBuf,
                    cuS2LenVecAlign / 32);
                AscendC::PipeBarrier<PIPE_V>();
                // 缓存4块512或者S2结束, 需要进行精排
                if (globalTopkUbCacheIdx == 3 || isS2End || info.isAllLoopEnd) {
                    LocalTensor<float> tt = SortedBasicBlock_[innerS1Idx * BASE_TOPK * 2];
                    // 前4块直接精排覆盖到globalTopkUb_
                    if (info.s2Idx - blockS2StartIdx_ < 4) {
                        MrgBasicBlock(globalTopkUb_[innerS1Idx * BASE_TOPK * 2], tt,
                                      static_cast<int64_t>(globalTopkUbCacheIdx + 1), s2BaseSize_);
                    } else { // 后面缓存在 SortedBasicBlock_, 先精排, 再merge到globalTopkUb_
                        if (globalTopkUbCacheIdx > 0) {
                            MrgBasicBlock(tmpSortBuf, tt, static_cast<int64_t>(globalTopkUbCacheIdx + 1), s2BaseSize_);
                            PipeBarrier<PIPE_V>();
                            DataCopy(SortedBasicBlock_[innerS1Idx * BASE_TOPK * 2], tmpSortBuf,
                                     (globalTopkUbCacheIdx + 1) * s2BaseSize_ * 2);
                        }
                        PipeBarrier<PIPE_V>();
                        SparseTopK(globalTopkUb_[innerS1Idx * BASE_TOPK * 2],
                                   SortedBasicBlock_[innerS1Idx * BASE_TOPK * 2], tmpSortBuf, BASE_TOPK,
                                   s2BaseSize_ * (globalTopkUbCacheIdx + 1));
                    }
                }
            }
            if (constInfo_.isSparseCountOver2K) {
                SetFlag<HardEvent::V_MTE2>(EVENTID_V_TO_MTE2_TMPUB);
            }

            PipeBarrier<PIPE_V>();
            outQueue_.FreeTensor(tmpSortBuf);

            bool needCopyOutGm = blockS2StartIdx_ == 0 && isS2End;

            // 中间结果保存:over2k 下每请求整 S2 单核处理,始终 needCopyOutGm 优先,ws/LD 路径死代码。
            bool needCopyWsGm = info.isLastS2InnerLoop;

            if (needCopyOutGm) {
                if (constInfo_.isSparseCountOver2K) {
                    // 融合窗行输出(每请求恰一次、由偶 AIV 的 innerS1Idx=0 执行):
                    //   globalTopkUb_ 已持本请求全前缀 top-4096((value,index) 交错)。
                    //   FuseWindowRowOut 在 tmpUb_ 组行: 粗筛 idx[0,c)(c=min(L,4096)) 前连续
                    //   ++ 窗新增(去重后) ++ -1 pad, 整行单次 CopyOut 到 [bIdx,n2Idx,outRowWidth]。
                    LocalTensor<float> fuseSrc = globalTopkUb_[innerS1Idx * virTopK * 2];
                    FuseWindowRowOut(info.bIdx, info.indiceOutOffset + (uint64_t)cuS1Idx * constInfo_.outRowWidth,
                                     fuseSrc, info.actS2Size);
                } else {
                    // 生产形态 CopyOut: Extract 分离 globalTopkUb_ 的 (value,index) 交错对,
                    // 经 outQueue_ 缓冲后拷贝 idx 到输出。coarse 无 value 输出,只拷贝索引。
                    // (host 强制 sparse_count==4096 → 恒 over2k,此分支为死代码保留作模板参照)
                    int64_t offset = (constInfo_.sparseCount <= SPARSE_COUNT_4K) ? virTopK : constInfo_.sparseCount / 2;
                    int64_t copyLen = (constInfo_.sparseCount <= SPARSE_COUNT_4K)
                                    ? constInfo_.sparseCount
                                    : constInfo_.sparseCount / 2;
                    int64_t copyNum = (constInfo_.sparseCount <= SPARSE_COUNT_4K) ? 1 : 2;
                    for (int64_t i = 0; i < copyNum; i++) {
                        LocalTensor<float> outValueUb = outQueue_.AllocTensor<float>();
                        LocalTensor<uint32_t> outIdxUb = outValueUb[offset].template ReinterpretCast<uint32_t>();
                        Extract(outValueUb, outIdxUb,
                                globalTopkUb_[innerS1Idx * virTopK * 2 + 2 * i * offset], offset / 32);
                        LocalTensor<int32_t> idxULocal1 = outValueUb[offset].template ReinterpretCast<int32_t>();
                        outQueue_.EnQue<float>(outValueUb);
                        outValueUb = outQueue_.DeQue<float>();
                        IndexerCoarseScreenServiceVec::CopyOut(indiceOutGm[info.indiceOutOffset + cuS1Idx *
                                                                     constInfo_.outRowWidth + i * offset],
                                            idxULocal1, copyLen);
                        outQueue_.FreeTensor(outValueUb);
                    }
                }
            } else if (needCopyWsGm) {
                // vec1Res Gm = [aic, s1BaseSize_, 2, 2, topkOut_] float32
                // vec1Param Gm = [aic, s1BaseSize_, 2, 16] int64
                //     16 = [needFd, s2AcSeq, s2Start, s2End, isS2End, bn2idx, s1Idx, S1ProcNum, ......]

                int64_t wsOffset = (blockId_ / 2) * s1BaseSize_ * 2 * 2 * BASE_TOPK +       // 2个AIV共同地址偏移
                                   (blockId_ % 2) * (s1BaseSize_ / 2) * 2 * 2 * BASE_TOPK + // 每个AIV的地址偏移，S1方向
                                   (ldS1Offset + innerS1Idx) * 2 * 2 * BASE_TOPK;
                int64_t wsInfoOffset = (blockId_ / 2) * s1BaseSize_ * 2 * paramNum_ +       // 2个AIV共同地址偏移
                                       (blockId_ % 2) * (s1BaseSize_ / 2) * 2 * paramNum_ + // 每个AIV的地址偏移，S1方向
                                       (ldS1Offset + innerS1Idx) * 2 * paramNum_;

                LocalTensor<int64_t> tmpiBuff = paramBuf_.Get<int64_t>();
                SetWaitFlag<HardEvent::MTE3_S>(HardEvent::MTE3_S);
                tmpiBuff.SetValue(0, static_cast<int64_t>(1));
                tmpiBuff.SetValue(1, static_cast<int64_t>(cuRealAcSeq));
                tmpiBuff.SetValue(2, static_cast<int64_t>(blockS2StartIdx_));
                tmpiBuff.SetValue(3, static_cast<int64_t>(cuBaseS2Idx + cuS2Len));
                tmpiBuff.SetValue(4, static_cast<int64_t>(isS2End));
                tmpiBuff.SetValue(5, static_cast<int64_t>(info.bN2Idx));
                tmpiBuff.SetValue(6, static_cast<int64_t>(cuS1Idx));
                tmpiBuff.SetValue(7, static_cast<int64_t>(cuS1ProcNum));
                tmpiBuff.SetValue(8, static_cast<int64_t>(info.indiceOutOffset + cuS1Idx * constInfo_.outRowWidth));
                // 写入头尾判断
                // [head, tail]
                // head: 与前面规约，与前后规约
                // tail: 与后面规约
                bool isTailReduce = blockS2StartIdx_ == 0; // 一定是isLastTile
                // WS偏移规则 blockS2StartIdx_ != 0
                // 跟前面块做规约 写到0偏移 不用做计算 blockS2StartIdx_ == 0 and !isS2End
                // 跟后面块做规约 写到1偏移  需要 + s1BaseSize_, BASE_TOPK*2
                if (isTailReduce) { // S2不是最后结束的数据就需要往后做规约，放入第二块ws
                    wsInfoOffset += paramNum_;
                    wsOffset += 2 * BASE_TOPK;
                }
                SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
                IndexerCoarseScreenServiceVec::CopyOut(vec1ParamGm[wsInfoOffset], tmpiBuff, 16);
                SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
                IndexerCoarseScreenServiceVec::CopyOut(vec1ResGm[wsOffset], globalTopkUb_[innerS1Idx * BASE_TOPK * 2], 2 * BASE_TOPK);
                SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
            }
        } else if (cuRealAcSeq <= 0) {
            CleanInvalidOutput(info.indiceOutOffset + cuS1Idx * constInfo_.outRowWidth);
        }
    }

    // BNSD场景无效S1 输出-1
    if (LAYOUT_T == LI_LAYOUT::BSND) {
        // 最后一个S1的基本块, 需要 >= info.actS1Size
        bool isS1LoopEnd = (cuBaseS1Idx + s1BaseSize_) >= info.actS1Size;
        int32_t invalidS1Num = constInfo_.qSeqSize - info.actS1Size;
        // blockS2StartIdx_ == 0 控制S2从开始的核去做冗余清理
        if (invalidS1Num > 0 && isS1LoopEnd && blockS2StartIdx_ == 0) {
            int32_t s1NumPerAiv = blockId_ % 2 == 0 ? CeilDiv(invalidS1Num, 2) : (invalidS1Num / 2);
            int32_t s1OffsetPerAiv = info.actS1Size + (blockId_ % 2) * CeilDiv(invalidS1Num, 2);
            for (int innerS1Idx = 0; innerS1Idx < s1NumPerAiv; innerS1Idx++) {
                CleanInvalidOutput(info.indiceOutOffset + (s1OffsetPerAiv + innerS1Idx) * constInfo_.outRowWidth);
            }
        }

        int32_t invalidS1Num2 = info.actS1Size - info.actS2Size;
        if (invalidS1Num2 > 0 && isS1LoopEnd && blockS2StartIdx_ == 0 && constInfo_.attenMaskFlag) {
            int32_t s1NumPerAiv = blockId_ % 2 == 0 ? CeilDiv(invalidS1Num2, 2) : (invalidS1Num2 / 2);
            int32_t s1OffsetPerAiv = (blockId_ % 2) * CeilDiv(invalidS1Num2, 2);
            for (int innerS1Idx = 0; innerS1Idx < s1NumPerAiv; innerS1Idx++) {
                CleanInvalidOutput((info.bN2Idx * constInfo_.qSeqSize + s1OffsetPerAiv + innerS1Idx) *
                                   constInfo_.outRowWidth);
            }
        }
    }

    if (info.isLastS2InnerLoop) {
        // S2最后一个Loop后, 下一个基本块初始从0开始
        blockS2StartIdx_ = 0;
    }
}

template <typename LIT>
__aicore__ inline void IndexerCoarseScreenServiceVector<LIT>::ProcessLD()
{
    int32_t curCubeId = blockId_ / 2;
    int32_t tmpCubeId = curCubeId;

    int64_t s2ActSeq;
    int64_t s2Start;
    int64_t s2End;
    int64_t isS2End;
    int64_t s1Idx;
    uint32_t acc_list_num = 0;
    int64_t bIdx = 0;
    int64_t needFd;
    int64_t wsOffset;
    int64_t wsInfoOffset = 0;
    int64_t nextneedFd;
    int64_t valueOffset = 0;
    int64_t outOffset = 0;

    LocalTensor<float> curValueIdxUb = ldToBeMrgBuf_.Get<float>();
    LocalTensor<float> tmpUb = ldTmpBuf_.Get<float>();

    // S2开头信息
    // 开始必然没有头规约，因此从尾规约开始处理，while循环读取下一个核的头规约
    // 存满4个list或者遇到S2结尾，则做merge，直到做完S2
    // 每个核都忽略自己的头规约，因为必然由前面的核做完
    uint32_t s1LdStartIdx = 0;
    uint32_t s1ProcNum = 0;
    uint64_t paramGmCoreOffset = tmpCubeId * s1BaseSize_ * 2 * paramNum_;
    for (uint32_t innerS1Idx = 0; innerS1Idx < s1BaseSize_; innerS1Idx++) {
        needFd = vec1ParamGm.GetValue(paramGmCoreOffset + innerS1Idx * 2 * paramNum_ + paramNum_);
        if (needFd == 1) {
            s1LdStartIdx = (s1ProcNum == 0) ? innerS1Idx : s1LdStartIdx;
            s1ProcNum++;
        }
    }

    if (s1ProcNum == 0) {
        return;
    }

    // S1逐行计算
    uint32_t s1VecNum = CeilDiv(s1ProcNum, 2);
    if (blockId_ % 2 == 1) {
        s1LdStartIdx = s1LdStartIdx + s1VecNum;
        s1VecNum = s1ProcNum - s1VecNum;
    }
    for (uint32_t innerS1Idx = s1LdStartIdx; innerS1Idx < s1LdStartIdx + s1VecNum; innerS1Idx++) {
        // 重置偏移
        tmpCubeId = curCubeId;
        acc_list_num = 0;
        valueOffset = 0;

        // 搬入数据
        wsOffset = tmpCubeId * s1BaseSize_ * 2 * 2 * BASE_TOPK + // 2个AIV共同地址偏移
                   innerS1Idx * 2 * 2 * BASE_TOPK + 2 * BASE_TOPK;
        SetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2);
        SetWaitFlag<HardEvent::S_MTE2>(HardEvent::S_MTE2);
        DataCopyPad(curValueIdxUb, vec1ResGm[wsOffset],
                    {1, static_cast<uint16_t>(2 * BASE_TOPK * sizeof(int32_t)), 0, 0}, {true, 0, 0, 0});
        acc_list_num++;
        valueOffset += 2 * BASE_TOPK;

        // 获取下一个核规约信息
        tmpCubeId++;
        wsInfoOffset = tmpCubeId * s1BaseSize_ * 2 * paramNum_ + innerS1Idx * 2 * paramNum_;
        needFd = vec1ParamGm.GetValue(wsInfoOffset);
        isS2End = vec1ParamGm.GetValue(wsInfoOffset + 4);
        s1Idx = vec1ParamGm.GetValue(wsInfoOffset + 6);
        outOffset = vec1ParamGm.GetValue(wsInfoOffset + 8);

        while (needFd == 1) {
            // 搬入头规约数据
            wsOffset = tmpCubeId * s1BaseSize_ * 2 * 2 * BASE_TOPK + // 2个AIV共同地址偏移
                       innerS1Idx * 2 * 2 * BASE_TOPK;
            SetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2);
            SetWaitFlag<HardEvent::S_MTE2>(HardEvent::S_MTE2);
            DataCopyPad(curValueIdxUb[valueOffset], vec1ResGm[wsOffset],
                        {1, static_cast<uint16_t>(2 * BASE_TOPK * sizeof(int32_t)), 0, 0}, {true, 0, 0, 0});
            valueOffset += 2 * BASE_TOPK;
            acc_list_num++;

            // 每满4个list，聚合  前2K为mrg结果
            if (acc_list_num == mrgListNum_) {
                // MrgSort 四条2048的队列，Mrg成一条
                AscendC::MrgSort4Info params;
                params.elementLengths[0] = BASE_TOPK;
                params.elementLengths[1] = BASE_TOPK;
                params.elementLengths[2] = BASE_TOPK;
                params.elementLengths[3] = BASE_TOPK;
                params.ifExhaustedSuspension = true;
                params.validBit = 0b1111;
                params.repeatTimes = 1;

                AscendC::MrgSortSrcList<float> srcList;
                srcList.src1 = curValueIdxUb[0];
                srcList.src2 = curValueIdxUb[2 * BASE_TOPK];
                srcList.src3 = curValueIdxUb[4 * BASE_TOPK];
                srcList.src4 = curValueIdxUb[6 * BASE_TOPK];
                SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
                MrgSort(tmpUb, srcList, params);
                PipeBarrier<PIPE_V>();
                DataCopy(curValueIdxUb, tmpUb, 2 * BASE_TOPK);
                PipeBarrier<PIPE_V>();
                acc_list_num = 1;
                valueOffset = 2 * BASE_TOPK;
            }

            // reduce到S2末尾，则跳出
            if (isS2End == 1) {
                break;
            }

            tmpCubeId++;
            wsInfoOffset = tmpCubeId * s1BaseSize_ * 2 * paramNum_ + innerS1Idx * 2 * paramNum_;
            needFd = vec1ParamGm.GetValue(wsInfoOffset);
            isS2End = vec1ParamGm.GetValue(wsInfoOffset + 4);
        }

        // mrg不足4个list的数据
        if (acc_list_num != 1) {
            AscendC::MrgSort4Info params;
            params.elementLengths[0] = BASE_TOPK;
            params.elementLengths[1] = BASE_TOPK;
            params.elementLengths[2] = BASE_TOPK;
            params.elementLengths[3] = BASE_TOPK;
            params.ifExhaustedSuspension = true;
            if (acc_list_num == 2) {
                params.validBit = 0b0011;
            } else if (acc_list_num == 3) {
                params.validBit = 0b0111;
            }
            params.repeatTimes = 1;

            AscendC::MrgSortSrcList<float> srcList;
            srcList.src1 = curValueIdxUb[0];
            srcList.src2 = curValueIdxUb[2 * BASE_TOPK];
            srcList.src3 = curValueIdxUb[4 * BASE_TOPK];
            srcList.src4 = curValueIdxUb[6 * BASE_TOPK];
            SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
            MrgSort(tmpUb, srcList, params);
            PipeBarrier<PIPE_V>();
            DataCopy(curValueIdxUb, tmpUb, 2 * BASE_TOPK);
            PipeBarrier<PIPE_V>();
        }

        // 搬出(生产 lightning_indexer ProcessLD returnValue=false 同款):
        //   Extract 分离 (value,index) → 直拷 idx(列号)。coarse 无 candidates 掩码,
        //   位置 ≥ actS2Size 由 Duplicate(NEG_INF) 预填沉底, 尾部空槽 = InitSortOutBuf 的 -1。
        LocalTensor<float> outValueUb = ldOutValueBuf_.Get<float>();
        LocalTensor<uint32_t> outIdxUb = ldOutIdxBuf_.Get<uint32_t>();
        Extract(outValueUb, outIdxUb, curValueIdxUb, (BASE_TOPK / 32));
        LocalTensor<int32_t> idxULocal1 = outIdxUb.template ReinterpretCast<int32_t>();
        SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
        SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
        DataCopyPad(indiceOutGm[outOffset], idxULocal1,
                    {1, static_cast<uint16_t>(constInfo_.sparseCount * sizeof(int32_t)), 0, 0});
        SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
    }
}
} // namespace LIKernel
#endif
