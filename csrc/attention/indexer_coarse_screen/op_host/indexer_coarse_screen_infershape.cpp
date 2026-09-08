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
 * \file indexer_coarse_screen_infershape.cpp
 * \brief
 */
#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "err/ops_err.h"


using namespace ge;

namespace ops {
constexpr uint32_t QUERY_INDEX = 0;
constexpr uint32_t KEY_INDEX = 1;
constexpr uint32_t ROW_WEIGHTS_INDEX = 3;
constexpr uint32_t ACTUAL_SEQ_Q_INDEX = 4;
constexpr uint32_t ATTR_QUERY_LAYOUT_INDEX = 0;
constexpr uint32_t ATTR_KEY_LAYOUT_INDEX = 1;
constexpr uint32_t ATTR_SPARSE_COUNT_INDEX = 2;
constexpr int64_t MAX_GROUP_INFERSHAPE = 16; // 本地窗 group 宽上界(镜像 python _MAX_GROUP)
constexpr int64_t DIM_ONE = 1;

static ge::graphStatus InferShapeIndexerCoarseScreen(gert::InferShapeContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("IndexerCoarseScreen", "InferShapeContext is nullptr!"),
               return ge::GRAPH_FAILED);
    const gert::Shape *queryShape = context->GetInputShape(QUERY_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, queryShape);
    const gert::Shape *keyShape = context->GetInputShape(KEY_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, keyShape);
    const gert::Shape *actualSeqQShape = context->GetInputShape(ACTUAL_SEQ_Q_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, actualSeqQShape);
    const gert::Shape *rowWeightsShape = context->GetInputShape(ROW_WEIGHTS_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, rowWeightsShape);

    gert::Shape *sparseIndicesShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, sparseIndicesShape);

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const char *inputLayoutQueryPtr = attrs->GetAttrPointer<char>(ATTR_QUERY_LAYOUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputLayoutQueryPtr);
    const char *inputLayoutKeyPtr = attrs->GetAttrPointer<char>(ATTR_KEY_LAYOUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputLayoutKeyPtr);
    const int64_t *seleced_count = attrs->GetInt(ATTR_SPARSE_COUNT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, seleced_count);
    std::string inputLayoutQueryPtrStr = std::string(inputLayoutQueryPtr);
    std::string inputLayoutKeyPtrStr = std::string(inputLayoutKeyPtr);
    // coarse_screen 固定 query=TND, key=PA_BSND
    OP_CHECK_IF(
        inputLayoutQueryPtrStr != "TND",
        OP_LOGE(context, "The attr layout_query should be TND, but got %s.", inputLayoutQueryPtrStr.c_str()),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        inputLayoutKeyPtrStr != "PA_BSND",
        OP_LOGE(context, "The attr layout_key should be PA_BSND, but got %s.", inputLayoutKeyPtrStr.c_str()),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        queryShape->GetDimNum() != 3,
        OP_LOGE(context, "Layout TND, queryDims (%zu) must be 3!", queryShape->GetDimNum()),
        return ge::GRAPH_FAILED);

    // 本地窗 group 宽 g = row_weights dim1(∈[1,16]),驱动输出行宽 W8
    int64_t g = rowWeightsShape->GetDim(DIM_ONE);
    OP_CHECK_IF(
        ((rowWeightsShape->GetDimNum() != 2) || (g <= 0) || (g > MAX_GROUP_INFERSHAPE)),
        OP_LOGE(context, "row_weights must be rank-2 and its last dim g must be in (0, %ld], but got dim_num=%zu g=%ld.",
            MAX_GROUP_INFERSHAPE, rowWeightsShape->GetDimNum(), g),
        return ge::GRAPH_FAILED);
    // 输出单行宽 = 粗筛 topk 宽 + 本地窗最大新增(自有 g + 尾项去重保留 ≤ g-1) 后 8 对齐
    int64_t outRowWidth = (*seleced_count + 2 * g - 1 + 7) / 8 * 8;
    // 输出 TND 布局 [R, N2(恒 1), outRowWidth],R 为 batch(请求数)
    sparseIndicesShape->SetDimNum(3);
    sparseIndicesShape->SetDim(0, actualSeqQShape->GetDim(0)); // 0:Dim R(actual_seq_lengths_query 长度)
    sparseIndicesShape->SetDim(1, keyShape->GetDim(2));         // 1:Dim N(PA_BSND key [BlockNum,BlockSize,N,D])
    sparseIndicesShape->SetDim(2, outRowWidth);                 // 2:Dim K(outRowWidth,见上)
    OP_LOGI(context->GetNodeName(), "IndexerCoarseScreen InferShape end.");

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeIndexerCoarseScreen(gert::InferDataTypeContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("IndexerCoarseScreen", "InferDataTypeContext is nullptr!"),
               return ge::GRAPH_FAILED);
    OP_LOGI(context->GetNodeName(), "Enter IndexerCoarseScreen InferDataType impl.");
    // 输出 topk_indices 恒为 int32
    context->SetOutputDataType(0, ge::DT_INT32);
    OP_LOGI(context->GetNodeName(), "IndexerCoarseScreen InferDataType end.");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(IndexerCoarseScreen)
    .InferShape(InferShapeIndexerCoarseScreen)
    .InferDataType(InferDataTypeIndexerCoarseScreen);
} // namespace ops
