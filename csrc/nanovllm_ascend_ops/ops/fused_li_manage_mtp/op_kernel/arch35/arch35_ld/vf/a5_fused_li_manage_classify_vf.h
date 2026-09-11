/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */

#ifndef LIGHTNING_INDEXER_DECODE_UPDATE_A5_CLASSIFY_VF_H
#define LIGHTNING_INDEXER_DECODE_UPDATE_A5_CLASSIFY_VF_H

#include "kernel_operator.h"

namespace TopkIndexerClassifyVF {
using namespace AscendC;

constexpr uint32_t CHUNK_SIZE = 64;
constexpr uint32_t INDEXER_TOKEN_BITS = 18;
constexpr uint32_t INDEXER_TOKEN_MASK = (1U << INDEXER_TOKEN_BITS) - 1U;
constexpr uint32_t INDEXER_MISS_SLOT = (1U << (32U - INDEXER_TOKEN_BITS)) - 1U;

__simd_vf__ void SqueezeIndexerMissTokenIds(__ubuf__ uint32_t *dst,
                                             __ubuf__ uint32_t *payload,
                                             uint32_t vecLoopNum)
{
    // 本 VF 必须以 ClearSpr<AR> 开头：StoreUnAlign POST_MODE_UPDATE 以 AR 作本 store
    // 链的累计写偏移/字节数，进链前 AR 必须归零，写入才落在 dst+0 且 AR 终值 = 本 VF
    // 存储字节数。曾为「基线差分」删除 ClearSpr（计数改由宿主差分），结果计数(差分)
    // 虽对，但写入被进链残留 AR 位移到 dst 之后 → classifiedIndex[0:mc] 全是宿值垃圾。
    // 宿主已改 PIPE_ALL 后读绝对值，本 VF 恢复清零（与 hit VF / union CompactSafeVictims
    // 同型，数据通道在 references 中被证实可靠）。见 service_vector.h PublishRoute。
    MicroAPI::MaskReg all = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::RegTensor<uint32_t> tokenMask;
    MicroAPI::RegTensor<uint32_t> missSlot;
    MicroAPI::Duplicate(tokenMask, INDEXER_TOKEN_MASK);
    MicroAPI::Duplicate(missSlot, INDEXER_MISS_SLOT);
    MicroAPI::ClearSpr<AscendC::SpecialPurposeReg::AR>();
    MicroAPI::UnalignRegForStore alignOut;

    for (uint32_t i = 0; i < vecLoopNum; ++i) {
        MicroAPI::RegTensor<uint32_t> packed;
        MicroAPI::RegTensor<uint32_t> slot;
        MicroAPI::RegTensor<uint32_t> token;
        MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_NORM>(
            packed, payload + i * CHUNK_SIZE);
        MicroAPI::ShiftRights(slot, packed, static_cast<int16_t>(INDEXER_TOKEN_BITS), all);
        MicroAPI::And(token, packed, tokenMask, all);
        MicroAPI::MaskReg isMiss;
        MicroAPI::Compare<uint32_t, CMPMODE::EQ>(isMiss, slot, missSlot, all);
        MicroAPI::RegTensor<uint32_t> compact;
        MicroAPI::Squeeze<uint32_t, MicroAPI::GatherMaskMode::STORE_REG>(compact, token, isMiss);
        MicroAPI::StoreUnAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(dst, compact, alignOut);
    }
    MicroAPI::StoreUnAlignPost(dst, alignOut);
    MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE,
                          AscendC::MicroAPI::MemType::VEC_LOAD>();
}

__simd_vf__ void SqueezeIndexerHitTokenIds(__ubuf__ uint32_t *dst,
                                            __ubuf__ uint32_t *payload,
                                            uint32_t vecLoopNum)
{
    MicroAPI::MaskReg all = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::RegTensor<uint32_t> tokenMask;
    MicroAPI::RegTensor<uint32_t> missSlot;
    MicroAPI::Duplicate(tokenMask, INDEXER_TOKEN_MASK);
    MicroAPI::Duplicate(missSlot, INDEXER_MISS_SLOT);
    MicroAPI::ClearSpr<AscendC::SpecialPurposeReg::AR>();
    MicroAPI::UnalignRegForStore alignOut;

    for (uint32_t i = 0; i < vecLoopNum; ++i) {
        MicroAPI::RegTensor<uint32_t> packed;
        MicroAPI::RegTensor<uint32_t> slot;
        MicroAPI::RegTensor<uint32_t> token;
        MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_NORM>(
            packed, payload + i * CHUNK_SIZE);
        MicroAPI::ShiftRights(slot, packed, static_cast<int16_t>(INDEXER_TOKEN_BITS), all);
        MicroAPI::And(token, packed, tokenMask, all);
        MicroAPI::MaskReg isMiss;
        MicroAPI::MaskReg isHit;
        MicroAPI::Compare<uint32_t, CMPMODE::EQ>(isMiss, slot, missSlot, all);
        MicroAPI::Xor(isHit, isMiss, all, all);
        MicroAPI::RegTensor<uint32_t> compact;
        MicroAPI::Squeeze<uint32_t, MicroAPI::GatherMaskMode::STORE_REG>(compact, token, isHit);
        MicroAPI::StoreUnAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(dst, compact, alignOut);
    }
    MicroAPI::StoreUnAlignPost(dst, alignOut);
    MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE,
                          AscendC::MicroAPI::MemType::VEC_LOAD>();
}

__simd_vf__ void SqueezeIndexerHitSlots(__ubuf__ uint32_t *dst,
                                         __ubuf__ uint32_t *payload,
                                         uint32_t vecLoopNum)
{
    MicroAPI::MaskReg all = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::RegTensor<uint32_t> missSlot;
    MicroAPI::Duplicate(missSlot, INDEXER_MISS_SLOT);
    MicroAPI::ClearSpr<AscendC::SpecialPurposeReg::AR>();
    MicroAPI::UnalignRegForStore alignOut;

    for (uint32_t i = 0; i < vecLoopNum; ++i) {
        MicroAPI::RegTensor<uint32_t> packed;
        MicroAPI::RegTensor<uint32_t> slot;
        MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_NORM>(
            packed, payload + i * CHUNK_SIZE);
        MicroAPI::ShiftRights(slot, packed, static_cast<int16_t>(INDEXER_TOKEN_BITS), all);
        MicroAPI::MaskReg isMiss;
        MicroAPI::MaskReg isHit;
        MicroAPI::Compare<uint32_t, CMPMODE::EQ>(isMiss, slot, missSlot, all);
        MicroAPI::Xor(isHit, isMiss, all, all);
        MicroAPI::RegTensor<uint32_t> compact;
        MicroAPI::Squeeze<uint32_t, MicroAPI::GatherMaskMode::STORE_REG>(compact, slot, isHit);
        MicroAPI::StoreUnAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(dst, compact, alignOut);
    }
    MicroAPI::StoreUnAlignPost(dst, alignOut);
    MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE,
                          AscendC::MicroAPI::MemType::VEC_LOAD>();
}

} // namespace TopkIndexerClassifyVF

#endif
