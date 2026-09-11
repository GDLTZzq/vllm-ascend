/** Stage 2-4: ordered union, victim selection, cache update and slot repair. */

#ifndef A5_FUSED_LI_MANAGE_MTP_C8_UNION_H
#define A5_FUSED_LI_MANAGE_MTP_C8_UNION_H

#include "kernel_operator.h"
#include "a5_fused_li_manage_mtp_c8_victim_vf.h"
#include "a5_fused_li_manage_mtp_c8_workspace.h"

namespace a5_fused_li_manage_mtp_c8_impl {
using namespace AscendC;

constexpr uint32_t UNION_ROUTES = 4U;
constexpr uint32_t UNION_TOPK = 2048U;
constexpr uint32_t UNION_CAPACITY = UNION_ROUTES * UNION_TOPK;
constexpr uint32_t UNION_PAIR_WORDS = UNION_TOPK * 2U;
constexpr uint32_t UNION_SOURCE_MASK = (1U << 18U) - 1U;
constexpr uint32_t UNION_KEY_BASE_BITS = 0x40000000U;
constexpr int32_t UNION_KEY_DECODE_BASE =
    static_cast<int32_t>(UNION_KEY_BASE_BITS + UNION_SOURCE_MASK);
constexpr uint32_t B32_VECTOR_ELEMENTS = 64U;
constexpr uint32_t B32_VECTOR_REPEAT_STRIDE = 8U;
constexpr uint32_t VICTIM_SCAN_CHUNK = 2048U;
constexpr uint32_t VICTIM_SLOT_SHIFT = 18U;
constexpr uint32_t VICTIM_SLOT_MASK = (1U << 14U) - 1U;

template <HardEvent event>
__aicore__ inline void UnionSync(HardEvent value)
{
    event_t id = static_cast<event_t>(GetTPipePtr()->FetchEventID(value));
    SetFlag<event>(id);
    WaitFlag<event>(id);
}

__aicore__ inline void ExtractPairKeys(
    const LocalTensor<uint32_t> &keys,
    const LocalTensor<uint32_t> &pairs,
    uint32_t count)
{
    GatherMaskParams params;
    params.repeatTimes =
        (count * 2U * sizeof(uint32_t) + 255U) / 256U;
    params.src0BlockStride = 1;
    params.src0RepeatStride = B32_VECTOR_REPEAT_STRIDE;
    params.src1RepeatStride = 0;
    uint64_t reserved = 0U;
    GatherMask(keys, pairs, static_cast<uint8_t>(1), false,
               static_cast<uint32_t>(0), params, reserved);
    PipeBarrier<PIPE_V>();
}

class OrderedMissUnion {
public:
    __aicore__ inline void Init(
        GM_ADDR routePairs, GM_ADDR routeThresholds, GM_ADDR routeCounts,
        GM_ADDR scoreWorkspace, GM_ADDR candidateLens,
        GM_ADDR reqPoolEntries, GM_ADDR cacheSlots,
        GM_ADDR unionSources, GM_ADDR unionDestinations,
        GM_ADDR unionCounts, GM_ADDR topkSources, GM_ADDR topkSlots,
        GM_ADDR cacheTokens,
        uint32_t sourceCapacity, uint32_t batchSize,
        uint32_t unionRowStride, TPipe *pipe)
    {
        routePairsGm_.SetGlobalBuffer((__gm__ int32_t *)routePairs);
        routeThresholdsGm_.SetGlobalBuffer(
            (__gm__ uint16_t *)routeThresholds);
        routeCountsGm_.SetGlobalBuffer((__gm__ int32_t *)routeCounts);
        scoreWorkspaceGm_.SetGlobalBuffer((__gm__ uint16_t *)scoreWorkspace);
        candidateLensGm_.SetGlobalBuffer((__gm__ int32_t *)candidateLens);
        reqPoolEntriesGm_.SetGlobalBuffer((__gm__ int32_t *)reqPoolEntries);
        cacheSlotsGm_.SetGlobalBuffer((__gm__ int32_t *)cacheSlots);
        unionSourcesGm_.SetGlobalBuffer((__gm__ int32_t *)unionSources);
        unionDestinationsGm_.SetGlobalBuffer(
            (__gm__ int32_t *)unionDestinations);
        unionCountsGm_.SetGlobalBuffer((__gm__ int32_t *)unionCounts);
        topkSourcesGm_.SetGlobalBuffer((__gm__ int32_t *)topkSources);
        topkSlotsGm_.SetGlobalBuffer((__gm__ int32_t *)topkSlots);
        cacheTokensGm_.SetGlobalBuffer((__gm__ int32_t *)cacheTokens);
        sourceCapacity_ = sourceCapacity;
        batchSize_ = batchSize;
        unionRowStride_ = unionRowStride;
        pipe->InitBuffer(pairInputBuf_,
                         UNION_CAPACITY * 2U * sizeof(float));
        pipe->InitBuffer(pairOutputBuf_,
                         UNION_CAPACITY * 2U * sizeof(float));
        pipe->InitBuffer(sourceBuf_,
                         UNION_CAPACITY * sizeof(int32_t));
        pipe->InitBuffer(countBuf_, 32U);
        pipe->InitBuffer(
            thresholdBuf_,
            UNION_ROUTES *
                a5_fused_li_manage_mtp_c8_workspace::THRESHOLD_STRIDE *
                sizeof(uint16_t));
    }

    __aicore__ inline void Process(uint32_t first, uint32_t stride)
    {
        for (uint32_t batch = first; batch < batchSize_; batch += stride) {
            ProcessRequest(batch);
        }
    }

private:
    __aicore__ inline uint32_t ScalarDeduplicate(
        LocalTensor<uint32_t> keys, uint32_t total,
        LocalTensor<int32_t> output)
    {
        uint32_t count = 0U;
        int32_t last = -1;
        for (uint32_t index = 0U; index < total; ++index) {
            const uint32_t key = keys.GetValue(index);
            const int32_t source = static_cast<int32_t>(
                UNION_SOURCE_MASK - (key - UNION_KEY_BASE_BITS));
            if (source != last) {
                output.SetValue(count++, source);
                last = source;
            }
        }
        return count;
    }

    __aicore__ inline uint32_t Deduplicate(
        LocalTensor<float> merged, LocalTensor<float> scratch,
        uint32_t total, LocalTensor<int32_t> output)
    {
        LocalTensor<uint32_t> keys = scratch.ReinterpretCast<uint32_t>();
        ExtractPairKeys(keys, merged.ReinterpretCast<uint32_t>(), total);
        if (total == 1U) {
            UnionSync<HardEvent::V_S>(HardEvent::V_S);
            output.SetValue(0U, static_cast<int32_t>(
                UNION_SOURCE_MASK -
                (keys.GetValue(0U) - UNION_KEY_BASE_BITS)));
            return 1U;
        }

        const uint32_t aligned =
            (total + B32_VECTOR_ELEMENTS - 1U) /
            B32_VECTOR_ELEMENTS * B32_VECTOR_ELEMENTS;
        if (aligned > total) {
            // A5 vector instructions require an aligned UB base.  `total`
            // is an arbitrary sum of four miss counts, so keys[total] is
            // not necessarily aligned and cannot be the destination of a
            // Duplicate.  The tail is at most 63 words; fill it from the
            // scalar pipe and synchronize before the next vector consumer.
            UnionSync<HardEvent::V_S>(HardEvent::V_S);
            for (uint32_t index = total; index < aligned; ++index) {
                keys.SetValue(index, 0U);
            }
            UnionSync<HardEvent::S_V>(HardEvent::S_V);
        }

        LocalTensor<int32_t> predecessorOffsets =
            scratch.ReinterpretCast<int32_t>()[UNION_CAPACITY];
        ArithProgression(predecessorOffsets, 0, 1, total);
        PipeBarrier<PIPE_V>();
        Adds(predecessorOffsets, predecessorOffsets, -1, total);
        PipeBarrier<PIPE_V>();
        Relu(predecessorOffsets, predecessorOffsets, total);
        PipeBarrier<PIPE_V>();
        Muls(predecessorOffsets, predecessorOffsets,
             static_cast<int32_t>(sizeof(uint32_t)), total);
        PipeBarrier<PIPE_V>();

        LocalTensor<uint32_t> predecessors =
            merged.ReinterpretCast<uint32_t>();
        Gather(predecessors.ReinterpretCast<float>(),
               keys.ReinterpretCast<float>(),
               predecessorOffsets.ReinterpretCast<uint32_t>(), 0U, total);
        PipeBarrier<PIPE_V>();
        UnionSync<HardEvent::V_S>(HardEvent::V_S);
        predecessors.SetValue(0U, predecessors.GetValue(0U) + 1U);
        if (aligned > total) {
            // Keep the same scalar-tail rule for the predecessor row.  It
            // also starts at the arbitrary `total` offset.
            for (uint32_t index = total; index < aligned; ++index) {
                predecessors.SetValue(index, 0U);
            }
        }
        UnionSync<HardEvent::S_V>(HardEvent::S_V);

        LocalTensor<uint8_t> uniqueMask =
            merged[UNION_CAPACITY].ReinterpretCast<uint8_t>();
        Compare(uniqueMask, keys.ReinterpretCast<float>(),
                predecessors.ReinterpretCast<float>(), CMPMODE::NE,
                aligned);
        PipeBarrier<PIPE_V>();

        GatherMaskParams compact;
        compact.repeatTimes = 1;
        compact.src0BlockStride = 1;
        compact.src0RepeatStride = B32_VECTOR_REPEAT_STRIDE;
        compact.src1RepeatStride = B32_VECTOR_REPEAT_STRIDE;
        uint64_t uniqueCount = 0U;
        GatherMask(output.ReinterpretCast<uint32_t>(), keys,
                   uniqueMask.ReinterpretCast<uint32_t>(), true, aligned,
                   compact, uniqueCount);
        // uniqueCount 是 gather 压缩的硬件计数，等价于 classify 的 AR 特存：
        // V_S 事件对特存的排序不足（S 管可能在任何 V 排空信号前读到陈旧值），
        // 必须以 PIPE_ALL 读绝对计数。曾用 PipeBarrier<PIPE_V>+UnionSync<V_S>
        // 导致 printf-free 下 dedup 计数偶尔偏小 → pool 卡基数丢 source。
        PipeBarrier<PIPE_ALL>();

        const uint32_t count = static_cast<uint32_t>(uniqueCount);
        if (count == 0U || count > total) {
            return ScalarDeduplicate(keys, total, output);
        }
        Muls(output, output, -1, count);
        PipeBarrier<PIPE_V>();
        Adds(output, output, UNION_KEY_DECODE_BASE, count);
        UnionSync<HardEvent::V_S>(HardEvent::V_S);
        return count;
    }

    __aicore__ inline void PublishCount(
        uint32_t batch, LocalTensor<int32_t> local, uint32_t count)
    {
        local.SetValue(0, static_cast<int32_t>(count));
        UnionSync<HardEvent::S_MTE3>(HardEvent::S_MTE3);
        DataCopyPad(unionCountsGm_[batch], local,
                    {1, static_cast<uint16_t>(sizeof(int32_t)), 0, 0});
        UnionSync<HardEvent::MTE3_S>(HardEvent::MTE3_S);
    }

    __aicore__ inline void PublishInactiveRequest(
        uint32_t batch, LocalTensor<int32_t> local,
        LocalTensor<int32_t> countLocal)
    {
        // Stage 1 is followed by a global barrier and TPipe::Reset().  Do not
        // issue inactive-row InitGlobalMemory operations from that phase: on
        // A5 their MTE3 lifetime can extend across the reset.  The Stage-2
        // request owner publishes the final no-op output from its own pipe.
        Duplicate(local, static_cast<int32_t>(-1), UNION_CAPACITY);
        Duplicate(countLocal, static_cast<int32_t>(0), UNION_ROUTES);
        PipeBarrier<PIPE_V>();
        UnionSync<HardEvent::V_MTE3>(HardEvent::V_MTE3);
        DataCopy(
            topkSourcesGm_[static_cast<uint64_t>(batch) * UNION_CAPACITY],
            local, UNION_CAPACITY);
        DataCopy(
            topkSlotsGm_[static_cast<uint64_t>(batch) * UNION_CAPACITY],
            local, UNION_CAPACITY);
        DataCopyPad(
            routeCountsGm_[static_cast<uint64_t>(batch) * UNION_ROUTES],
            countLocal,
            {1, static_cast<uint16_t>(UNION_ROUTES * sizeof(int32_t)),
             0, 0});
        UnionSync<HardEvent::MTE3_S>(HardEvent::MTE3_S);
        PublishCount(batch, countLocal, 0U);
    }

    __aicore__ inline uint32_t HashVictimScanSeed(
        uint32_t candidate, uint32_t poolRow)
    {
        uint32_t value = candidate ^ ((poolRow + 1U) * 0x9e3779b9U);
        value ^= value >> 16U;
        value *= 0x7feb352dU;
        value ^= value >> 15U;
        value *= 0x846ca68bU;
        value ^= value >> 16U;
        return value;
    }

    __aicore__ inline void LoadThresholds(
        uint32_t batch, LocalTensor<uint16_t> local,
        uint16_t values[UNION_ROUTES])
    {
        constexpr uint32_t STRIDE =
            a5_fused_li_manage_mtp_c8_workspace::THRESHOLD_STRIDE;
        const uint64_t routeBase =
            static_cast<uint64_t>(batch) * UNION_ROUTES;
        for (uint32_t route = 0U; route < UNION_ROUTES; ++route) {
            DataCopyPad(
                local[route * STRIDE],
                routeThresholdsGm_[(routeBase + route) * STRIDE],
                {1, STRIDE * static_cast<uint32_t>(sizeof(uint16_t)),
                 0, 0, 0},
                {false, 0, 0, 0});
        }
        UnionSync<HardEvent::MTE2_S>(HardEvent::MTE2_S);
        for (uint32_t route = 0U; route < UNION_ROUTES; ++route) {
            values[route] = local.GetValue(route * STRIDE);
        }
    }

    __aicore__ inline uint32_t CompactSafeVictims(
        uint32_t batch, uint32_t candidate, uint32_t poolRow,
        uint32_t budget, uint32_t required,
        const uint16_t thresholds[UNION_ROUTES],
        LocalTensor<int32_t> destinations,
        LocalTensor<int32_t> victimSources)
    {
        LocalTensor<uint8_t> scratch = pairInputBuf_.Get<uint8_t>();
        LocalTensor<uint16_t> score0 =
            scratch.ReinterpretCast<uint16_t>();
        LocalTensor<uint16_t> score1 = score0[VICTIM_SCAN_CHUNK];
        LocalTensor<uint16_t> score2 =
            score0[VICTIM_SCAN_CHUNK * 2U];
        LocalTensor<uint16_t> score3 =
            score0[VICTIM_SCAN_CHUNK * 3U];
        LocalTensor<int32_t> slots32 =
            scratch[VICTIM_SCAN_CHUNK * UNION_ROUTES *
                    sizeof(uint16_t)].ReinterpretCast<int32_t>();
        LocalTensor<int16_t> slots16 =
            scratch[VICTIM_SCAN_CHUNK *
                    (UNION_ROUTES * sizeof(uint16_t) + sizeof(int32_t))]
                .ReinterpretCast<int16_t>();
        LocalTensor<uint32_t> compact =
            scratch[VICTIM_SCAN_CHUNK *
                    (UNION_ROUTES * sizeof(uint16_t) + sizeof(int32_t) +
                     sizeof(int16_t))].ReinterpretCast<uint32_t>();

        const uint32_t chunks =
            (candidate + VICTIM_SCAN_CHUNK - 1U) / VICTIM_SCAN_CHUNK;
        const uint32_t firstChunk =
            HashVictimScanSeed(candidate, poolRow) % chunks;
        const uint64_t requestScoreBase =
            static_cast<uint64_t>(batch) * UNION_ROUTES * sourceCapacity_;
        const uint64_t cacheBase =
            static_cast<uint64_t>(poolRow) * sourceCapacity_;
        uint32_t written = 0U;
        for (uint32_t visit = 0U;
             visit < chunks && written < required; ++visit) {
            const uint32_t chunk = (firstChunk + visit) % chunks;
            const uint32_t chunkBase = chunk * VICTIM_SCAN_CHUNK;
            const uint32_t chunkLen =
                chunkBase + VICTIM_SCAN_CHUNK > candidate
                    ? candidate - chunkBase
                    : VICTIM_SCAN_CHUNK;
            const uint32_t alignedLen =
                (chunkLen + B32_VECTOR_ELEMENTS - 1U) /
                B32_VECTOR_ELEMENTS * B32_VECTOR_ELEMENTS;
            const DataCopyExtParams scoreCopy{
                1, chunkLen * static_cast<uint32_t>(sizeof(uint16_t)),
                0, 0, 0};
            const DataCopyPadExtParams<uint16_t> scorePad{
                true, 0,
                static_cast<uint8_t>(alignedLen - chunkLen), 0U};
            DataCopyPad(
                score0,
                scoreWorkspaceGm_[requestScoreBase + chunkBase],
                scoreCopy, scorePad);
            DataCopyPad(
                score1,
                scoreWorkspaceGm_[requestScoreBase + sourceCapacity_ +
                                  chunkBase],
                scoreCopy, scorePad);
            DataCopyPad(
                score2,
                scoreWorkspaceGm_[requestScoreBase +
                                  sourceCapacity_ * 2U + chunkBase],
                scoreCopy, scorePad);
            DataCopyPad(
                score3,
                scoreWorkspaceGm_[requestScoreBase +
                                  sourceCapacity_ * 3U + chunkBase],
                scoreCopy, scorePad);
            DataCopyPad(
                slots32, cacheSlotsGm_[cacheBase + chunkBase],
                {1, chunkLen * static_cast<uint32_t>(sizeof(int32_t)),
                 0, 0, 0},
                {false, 0, 0, 0});
            UnionSync<HardEvent::MTE2_V>(HardEvent::MTE2_V);

            Cast(slots16, slots32, RoundMode::CAST_NONE, chunkLen);
            if (alignedLen > chunkLen) {
                Duplicate(
                    slots16[chunkLen], static_cast<int16_t>(-1),
                    alignedLen - chunkLen);
            }
            PipeBarrier<PIPE_V>();
            A5MtpC8VictimVF::CompactEligiblePayloads(
                (__ubuf__ uint32_t *)compact.GetPhyAddr(),
                (__ubuf__ uint16_t *)score0.GetPhyAddr(),
                (__ubuf__ uint16_t *)score1.GetPhyAddr(),
                (__ubuf__ uint16_t *)score2.GetPhyAddr(),
                (__ubuf__ uint16_t *)score3.GetPhyAddr(),
                (__ubuf__ uint16_t *)slots16.GetPhyAddr(),
                thresholds[0], thresholds[1], thresholds[2],
                thresholds[3], static_cast<uint16_t>(budget),
                chunkBase, alignedLen / B32_VECTOR_ELEMENTS);
            // CompactEligiblePayloads 在 V 管 ClearSpr<AR> + POST_MODE_UPDATE 累加
            // compact 计数，此处 S 管 GetSpr 读取前必须保证 AR 已落。V_S 事件对
            // 该特存的排序在 stage1 classify 已被证实不足（service_vector.h 同型
            // 竞态：打印延时的核 mc 对、无打印的核错），统一用 PIPE_ALL 全流水
            // 屏障让 S 等到 V 排空后再读 AR。
            PipeBarrier<PIPE_ALL>();
            const uint32_t compactCount = static_cast<uint32_t>(
                GetSpr<SpecialPurposeReg::AR>() / sizeof(uint32_t));
            PipeBarrier<PIPE_V>();
            const uint32_t remaining = required - written;
            const uint32_t keep =
                compactCount < remaining ? compactCount : remaining;
            for (uint32_t index = 0U; index < keep; ++index) {
                const uint32_t payload = compact.GetValue(index);
                destinations.SetValue(
                    written + index,
                    static_cast<int32_t>(
                        (payload >> VICTIM_SLOT_SHIFT) & VICTIM_SLOT_MASK));
                victimSources.SetValue(
                    written + index,
                    static_cast<int32_t>(payload & UNION_SOURCE_MASK));
            }
            written += keep;
        }
        return written;
    }

    __aicore__ inline uint32_t AppendExactVictims(
        uint32_t batch, uint32_t candidate, uint32_t poolRow,
        uint32_t budget, uint32_t written, uint32_t required,
        LocalTensor<int32_t> destinations,
        LocalTensor<int32_t> victimSources)
    {
        if (written >= required) {
            return written;
        }
        // One int32 marker per physical cache slot fits in pairInputBuf_
        // because the public cache budget is capped at 16256.
        LocalTensor<int32_t> protectedSlots =
            pairInputBuf_.Get<int32_t>();
        Duplicate(protectedSlots, static_cast<int32_t>(0), budget);
        PipeBarrier<PIPE_V>();
        UnionSync<HardEvent::V_S>(HardEvent::V_S);
        for (uint32_t index = 0U; index < written; ++index) {
            const int32_t slot = destinations.GetValue(index);
            if (slot >= 0 && static_cast<uint32_t>(slot) < budget) {
                protectedSlots.SetValue(static_cast<uint32_t>(slot), 1);
            }
        }
        const uint64_t topkBase =
            static_cast<uint64_t>(batch) * UNION_CAPACITY;
        for (uint32_t index = 0U; index < UNION_CAPACITY; ++index) {
            const int32_t slot = topkSlotsGm_.GetValue(topkBase + index);
            if (slot >= 0 && static_cast<uint32_t>(slot) < budget) {
                protectedSlots.SetValue(static_cast<uint32_t>(slot), 1);
            }
        }

        const uint32_t chunks =
            (candidate + VICTIM_SCAN_CHUNK - 1U) / VICTIM_SCAN_CHUNK;
        const uint32_t firstChunk =
            HashVictimScanSeed(candidate, poolRow) % chunks;
        const uint64_t cacheBase =
            static_cast<uint64_t>(poolRow) * sourceCapacity_;
        for (uint32_t visit = 0U;
             visit < chunks && written < required; ++visit) {
            const uint32_t chunk = (firstChunk + visit) % chunks;
            const uint32_t begin = chunk * VICTIM_SCAN_CHUNK;
            const uint32_t end =
                begin + VICTIM_SCAN_CHUNK < candidate
                    ? begin + VICTIM_SCAN_CHUNK
                    : candidate;
            for (uint32_t source = begin;
                 source < end && written < required; ++source) {
                const int32_t slot =
                    cacheSlotsGm_.GetValue(cacheBase + source);
                if (slot < 0 || static_cast<uint32_t>(slot) >= budget ||
                    protectedSlots.GetValue(
                        static_cast<uint32_t>(slot)) != 0) {
                    continue;
                }
                destinations.SetValue(written, slot);
                victimSources.SetValue(
                    written, static_cast<int32_t>(source));
                protectedSlots.SetValue(static_cast<uint32_t>(slot), 1);
                ++written;
            }
        }
        return written;
    }

    __aicore__ inline uint32_t FindVictims(
        uint32_t batch, uint32_t count,
        LocalTensor<int32_t> destinations,
        LocalTensor<int32_t> victimSources)
    {
        if (count == 0U) {
            return 0U;
        }
        Duplicate(destinations, static_cast<int32_t>(-1), count);
        Duplicate(victimSources, static_cast<int32_t>(-1), count);
        PipeBarrier<PIPE_V>();

        LocalTensor<uint16_t> thresholdLocal = thresholdBuf_.Get<uint16_t>();
        uint16_t thresholds[UNION_ROUTES];
        LoadThresholds(batch, thresholdLocal, thresholds);
        const uint32_t candidate = static_cast<uint32_t>(
            candidateLensGm_.GetValue(batch));
        const uint32_t poolRow = static_cast<uint32_t>(
            reqPoolEntriesGm_.GetValue(batch));
        const uint32_t budget = static_cast<uint32_t>(
            cacheTokensGm_.GetValue(batch));
        uint32_t written = CompactSafeVictims(
            batch, candidate, poolRow, budget, count, thresholds,
            destinations, victimSources);
        written = AppendExactVictims(
            batch, candidate, poolRow, budget, written, count,
            destinations, victimSources);
        return written;
    }

    __aicore__ inline uint32_t ApplyCacheUpdates(
        uint32_t batch, uint32_t count,
        LocalTensor<int32_t> sources,
        LocalTensor<int32_t> destinations,
        LocalTensor<int32_t> victimSources)
    {
        const uint32_t candidate = static_cast<uint32_t>(
            candidateLensGm_.GetValue(batch));
        const uint32_t poolRow = static_cast<uint32_t>(
            reqPoolEntriesGm_.GetValue(batch));
        const uint32_t budget = static_cast<uint32_t>(
            cacheTokensGm_.GetValue(batch));
        const uint64_t cacheBase =
            static_cast<uint64_t>(poolRow) * sourceCapacity_;
        uint32_t updated = 0U;
        for (uint32_t index = 0U; index < count; ++index) {
            const int32_t source = sources.GetValue(index);
            const int32_t destination = destinations.GetValue(index);
            const int32_t victimSource = victimSources.GetValue(index);
            if (source < 0 || victimSource < 0 || destination < 0 ||
                static_cast<uint32_t>(source) >= candidate ||
                static_cast<uint32_t>(victimSource) >= candidate ||
                static_cast<uint32_t>(destination) >= budget) {
                break;
            }
            cacheSlotsGm_.SetValue(
                cacheBase + static_cast<uint32_t>(victimSource), -1);
            cacheSlotsGm_.SetValue(
                cacheBase + static_cast<uint32_t>(source), destination);
            ++updated;
        }
        if (updated != 0U) {
            PipeBarrier<PIPE_ALL>();
        }
        return updated;
    }

    __aicore__ inline void PrepareTopkMissPrefixes(
        uint32_t batch, const uint32_t lengths[UNION_ROUTES],
        uint32_t unionCount, LocalTensor<int32_t> unionSources,
        LocalTensor<int32_t> unionDestinations,
        LocalTensor<int32_t> allDestinations)
    {
        LocalTensor<int32_t> routePairs = pairInputBuf_.Get<int32_t>();
        // pairInputBuf_ has 16384 int32 words. One 4096-word interleaved
        // route-pair row plus four 2048-word destination prefixes use only
        // 12288 words and remain disjoint.
        const uint64_t requestPairBase =
            static_cast<uint64_t>(batch) * UNION_CAPACITY * 2U;
        for (uint32_t route = 0U; route < UNION_ROUTES; ++route) {
            const uint32_t length = lengths[route];
            if (length == 0U) {
                continue;
            }
            const uint32_t pairOffset = route * UNION_PAIR_WORDS;
            DataCopyPad(
                routePairs,
                routePairsGm_[requestPairBase + pairOffset],
                {1, length * 2U * static_cast<uint32_t>(sizeof(int32_t)),
                 0, 0, 0},
                {false, 0, 0, 0});
            UnionSync<HardEvent::MTE2_S>(HardEvent::MTE2_S);

            LocalTensor<int32_t> rowDestinations =
                allDestinations[route * UNION_TOPK];
            uint32_t unionCursor = 0U;
            for (uint32_t miss = 0U; miss < length; ++miss) {
                const int32_t source = routePairs.GetValue(miss * 2U + 1U);
                while (unionCursor < unionCount &&
                       unionSources.GetValue(unionCursor) < source) {
                    ++unionCursor;
                }
                const int32_t destination =
                    unionCursor < unionCount &&
                            unionSources.GetValue(unionCursor) == source
                        ? unionDestinations.GetValue(unionCursor)
                        : -1;
                rowDestinations.SetValue(miss, destination);
            }
        }
    }

    __aicore__ inline void PublishFinalOutputs(
        uint32_t batch, const uint32_t lengths[UNION_ROUTES],
        uint32_t count, LocalTensor<int32_t> countLocal,
        LocalTensor<int32_t> unionSources,
        LocalTensor<int32_t> unionDestinations,
        LocalTensor<int32_t> topkMissDestinations)
    {
        countLocal.SetValue(0U, static_cast<int32_t>(count));
        UnionSync<HardEvent::S_MTE3>(HardEvent::S_MTE3);
        if (count != 0U) {
            // Row stride is the host's miss-output width, not this engine's
            // internal UNION_CAPACITY: the A5 op declares [B, 4*2048] while the
            // shared A3 schema declares [B, 32768].
            const uint64_t unionOffset =
                static_cast<uint64_t>(batch) * unionRowStride_;
            const uint16_t unionBytes = static_cast<uint16_t>(
                count * static_cast<uint32_t>(sizeof(int32_t)));
            DataCopyPad(
                unionSourcesGm_[unionOffset], unionSources,
                {1, unionBytes, 0, 0});
            DataCopyPad(
                unionDestinationsGm_[unionOffset], unionDestinations,
                {1, unionBytes, 0, 0});
            for (uint32_t route = 0U; route < UNION_ROUTES; ++route) {
                if (lengths[route] == 0U) {
                    continue;
                }
                const uint64_t rowOffset =
                    (static_cast<uint64_t>(batch) * UNION_ROUTES + route) *
                    UNION_TOPK;
                DataCopyPad(
                    topkSlotsGm_[rowOffset],
                    topkMissDestinations[route * UNION_TOPK],
                    {1, static_cast<uint16_t>(
                            lengths[route] * sizeof(int32_t)),
                     0, 0});
            }
        }
        DataCopyPad(
            unionCountsGm_[batch], countLocal,
            {1, static_cast<uint16_t>(sizeof(int32_t)), 0, 0});
        UnionSync<HardEvent::MTE3_S>(HardEvent::MTE3_S);
    }

    __aicore__ inline void ProcessRequest(uint32_t batch)
    {
        LocalTensor<float> pairs = pairInputBuf_.Get<float>();
        LocalTensor<float> merged = pairOutputBuf_.Get<float>();
        LocalTensor<int32_t> sources = sourceBuf_.Get<int32_t>();
        LocalTensor<int32_t> countLocal = countBuf_.Get<int32_t>();
#if C8_MTP_BISECT_PRINT
        AscendC::printf("[ST2] blk=%lld U0 batch=%lld enter\n",
                        (int64_t)GetBlockIdx(), (int64_t)batch);
#endif
        if (cacheTokensGm_.GetValue(batch) == 0) {
            PublishInactiveRequest(batch, sources, countLocal);
            return;
        }
#if C8_MTP_BISECT_PRINT
        {
            const int32_t candDiag = candidateLensGm_.GetValue(batch);
            const int32_t budDiag = cacheTokensGm_.GetValue(batch);
            AscendC::printf(
                "[ST2] blk=%lld U1 batch=%lld cand=%lld bud=%lld\n",
                (int64_t)GetBlockIdx(), (int64_t)batch, (int64_t)candDiag,
                (int64_t)budDiag);
        }
#endif
        const uint64_t routeBase =
            static_cast<uint64_t>(batch) * UNION_CAPACITY * 2U;
        DataCopyPad(countLocal, routeCountsGm_[batch * UNION_ROUTES],
                    {1, static_cast<uint32_t>(UNION_ROUTES * sizeof(int32_t)),
                     0, 0, 0},
                    {false, 0, 0, 0});
        UnionSync<HardEvent::MTE2_S>(HardEvent::MTE2_S);

        uint32_t lengths[UNION_ROUTES] = {0U, 0U, 0U, 0U};
        uint32_t total = 0U;
        LocalTensor<int32_t> pairWords = pairs.ReinterpretCast<int32_t>();
        for (uint32_t route = 0U; route < UNION_ROUTES; ++route) {
            int32_t rawLength = countLocal.GetValue(route);
            uint32_t length = rawLength < 0
                ? 0U : static_cast<uint32_t>(rawLength);
            if (length > UNION_TOPK) {
                length = UNION_TOPK;
            }
            lengths[route] = length;
            total += length;
            if (length > 0U) {
                const uint32_t pairOffset = route * UNION_PAIR_WORDS;
                DataCopyPad(
                    pairWords[pairOffset],
                    routePairsGm_[routeBase + pairOffset],
                    {1, static_cast<uint32_t>(
                            length * 2U * sizeof(int32_t)),
                     0, 0, 0},
                    {false, 0, 0, 0});
            }
        }
        if (total == 0U) {
            PublishCount(batch, countLocal, 0U);
            return;
        }

        UnionSync<HardEvent::MTE2_V>(HardEvent::MTE2_V);
        MrgSort4Info params;
        params.elementLengths[0] = lengths[0];
        params.elementLengths[1] = lengths[1];
        params.elementLengths[2] = lengths[2];
        params.elementLengths[3] = lengths[3];
        params.ifExhaustedSuspension = false;
        params.validBit =
            (lengths[0] > 0U ? 0b0001 : 0U) |
            (lengths[1] > 0U ? 0b0010 : 0U) |
            (lengths[2] > 0U ? 0b0100 : 0U) |
            (lengths[3] > 0U ? 0b1000 : 0U);
        params.repeatTimes = 1;
        MrgSortSrcList<float> inputs;
        inputs.src1 = pairs;
        inputs.src2 = pairs[UNION_PAIR_WORDS];
        inputs.src3 = pairs[UNION_PAIR_WORDS * 2U];
        inputs.src4 = pairs[UNION_PAIR_WORDS * 3U];
        MrgSort<float>(merged, inputs, params);
        // MrgSort 的 merged 输出被 Deduplicate 的 ExtractPairKeys（向量读）消费。
        // PIPE_V 只保证 V 管先前指令完成；排序引擎完成/特存结算可能不被它覆盖，
        // printf-free 下偶现 merged 读陈旧 → 归并前缀丢/乱。全管道屏障最稳妥。
        PipeBarrier<PIPE_ALL>();
#if C8_MTP_BISECT_PRINT
        AscendC::printf("[ST2] blk=%lld U2 batch=%lld total=%lld\n",
                        (int64_t)GetBlockIdx(), (int64_t)batch,
                        (int64_t)total);
#endif
        const uint32_t count = Deduplicate(merged, pairs, total, sources);
#if C8_MTP_BISECT_PRINT
        AscendC::printf("[ST2] blk=%lld U3 batch=%lld count=%lld\n",
                        (int64_t)GetBlockIdx(), (int64_t)batch,
                        (int64_t)count);
#endif
        LocalTensor<int32_t> victimStorage =
            pairOutputBuf_.Get<int32_t>();
        LocalTensor<int32_t> destinations = victimStorage;
        LocalTensor<int32_t> victimSources =
            victimStorage[UNION_CAPACITY];
        const uint32_t found = FindVictims(
            batch, count, destinations, victimSources);
#if C8_MTP_BISECT_PRINT
        AscendC::printf("[ST2] blk=%lld U4 batch=%lld found=%lld\n",
                        (int64_t)GetBlockIdx(), (int64_t)batch,
                        (int64_t)found);
#endif
        const uint32_t updated = found == count
            ? ApplyCacheUpdates(
                  batch, count, sources, destinations, victimSources)
            : 0U;
#if C8_MTP_BISECT_PRINT
        AscendC::printf("[ST2] blk=%lld U5 batch=%lld updated=%lld\n",
                        (int64_t)GetBlockIdx(), (int64_t)batch,
                        (int64_t)updated);
#endif
#if C8_MTP_DIAG_MC
        if (GetBlockIdx() == 0U) {
            AscendC::printf("[UDIAG] b=%lld l0=%lld l1=%lld l2=%lld l3=%lld\n",
                            (int64_t)batch, (int64_t)lengths[0],
                            (int64_t)lengths[1], (int64_t)lengths[2],
                            (int64_t)lengths[3]);
            AscendC::printf("[UDIAG] tot=%lld dedup=%lld found=%lld upd=%lld\n",
                            (int64_t)total, (int64_t)count, (int64_t)found,
                            (int64_t)updated);
            // dupLeak = sources[0:count] 内实际重复源数（O(count^2) 判重）。若大 →
            // MrgSort/Deduplicate 没把跨 route 同源并邻，漏去重 → apply 每次重复源
            // 多逐出 1 个旧 token → pool 基数按 dupLeak 掉。
            int64_t dupLeak = 0;
            for (uint32_t di = 0U; di < count; ++di) {
                const int32_t dsrc = sources.GetValue(di);
                for (uint32_t dj = 0U; dj < di; ++dj) {
                    if (dsrc == sources.GetValue(dj)) {
                        ++dupLeak;
                        break;
                    }
                }
            }
            // recount = apply 后池行 resident 数（对账测试 12194/12288）。
            const uint32_t candDiag = static_cast<uint32_t>(
                candidateLensGm_.GetValue(batch));
            const uint32_t poolRowDiag = static_cast<uint32_t>(
                reqPoolEntriesGm_.GetValue(batch));
            const uint32_t budgetDiag = static_cast<uint32_t>(
                cacheTokensGm_.GetValue(batch));
            const uint64_t cacheBaseDiag =
                static_cast<uint64_t>(poolRowDiag) * sourceCapacity_;
            int64_t resident = 0;
            for (uint32_t dsrc = 0U; dsrc < candDiag; ++dsrc) {
                if (cacheSlotsGm_.GetValue(cacheBaseDiag + dsrc) >= 0) {
                    ++resident;
                }
            }
            AscendC::printf("[UDIAG2] dupLeak=%lld resident=%lld budget=%lld\n",
                            dupLeak, resident, (int64_t)budgetDiag);
        }
#endif
        LocalTensor<int32_t> topkScratch = pairInputBuf_.Get<int32_t>();
        LocalTensor<int32_t> topkMissDestinations =
            topkScratch[UNION_PAIR_WORDS];
        PrepareTopkMissPrefixes(
            batch, lengths, updated, sources, destinations,
            topkMissDestinations);
        PublishFinalOutputs(
            batch, lengths, updated, countLocal, sources, destinations,
            topkMissDestinations);
#if C8_MTP_BISECT_PRINT
        AscendC::printf("[ST2] blk=%lld U6 batch=%lld publish done\n",
                        (int64_t)GetBlockIdx(), (int64_t)batch);
#endif
    }

    GlobalTensor<int32_t> routePairsGm_;
    GlobalTensor<uint16_t> routeThresholdsGm_;
    GlobalTensor<int32_t> routeCountsGm_;
    GlobalTensor<uint16_t> scoreWorkspaceGm_;
    GlobalTensor<int32_t> candidateLensGm_;
    GlobalTensor<int32_t> reqPoolEntriesGm_;
    GlobalTensor<int32_t> cacheSlotsGm_;
    GlobalTensor<int32_t> unionSourcesGm_;
    GlobalTensor<int32_t> unionDestinationsGm_;
    GlobalTensor<int32_t> unionCountsGm_;
    GlobalTensor<int32_t> topkSourcesGm_;
    GlobalTensor<int32_t> topkSlotsGm_;
    GlobalTensor<int32_t> cacheTokensGm_;
    TBuf<TPosition::VECCALC> pairInputBuf_;
    TBuf<TPosition::VECCALC> pairOutputBuf_;
    TBuf<TPosition::VECCALC> sourceBuf_;
    TBuf<TPosition::VECCALC> countBuf_;
    TBuf<TPosition::VECCALC> thresholdBuf_;
    uint32_t sourceCapacity_ = 0U;
    uint32_t batchSize_ = 0U;
    uint32_t unionRowStride_ = 0U;
};
} // namespace a5_fused_li_manage_mtp_c8_impl

#endif
