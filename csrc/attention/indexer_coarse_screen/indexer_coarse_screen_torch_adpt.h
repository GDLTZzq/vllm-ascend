/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef INDEXER_COARSE_SCREEN_TORCH_ADPT_H
#define INDEXER_COARSE_SCREEN_TORCH_ADPT_H

namespace vllm_ascend {

at::Tensor construct_indexer_coarse_screen_output_tensor(
    const at::Tensor& query, const at::Tensor& key, const at::Tensor& row_weights,
    const c10::optional<at::Tensor>& actual_seq_lengths_query, int64_t sparse_count,
    const std::string& query_layout_str, const std::string& key_layout_str)
{
    constexpr int64_t SIZE = 8;
    constexpr int64_t ALIGN_8 = 8;    // 输出行宽 8 元素对齐(镜像 host AlignUpTo8)
    constexpr int64_t MAX_GROUP = 16; // group 宽 g 上界(镜像 python _MAX_GROUP)
    constexpr int64_t DIM_0 = 0;
    constexpr int64_t DIM_1 = 1;
    constexpr int64_t DIM_2 = 2;

    at::SmallVector<int64_t, SIZE> output_size;
    for (size_t i = 0; i < query.sizes().size(); i++) {
        TORCH_CHECK(query.size(i) > 0,
                    "All values within query's shape should be greater "
                    "than 0, but shape[",
                    i, "] is ", query.size(i));
    }
    for (size_t i = 0; i < key.sizes().size(); i++) {
        TORCH_CHECK(key.size(i) > 0,
                    "All values within key's shape should be greater "
                    "than 0, but shape[",
                    i, "] is ", key.size(i));
    }
    TORCH_CHECK(sparse_count > 0,
                "sparse count should be greater than 0, but now is ",
                sparse_count);
    TORCH_CHECK(row_weights.dim() == 2 && row_weights.size(DIM_1) > 0 &&
                    row_weights.size(DIM_1) <= MAX_GROUP,
                "row_weights must be rank-2 and its last dim g must be in (0, ",
                MAX_GROUP, "], but got dim=", row_weights.dim(),
                " g=", row_weights.dim() > 1 ? row_weights.size(DIM_1) : -1);
    // coarse_screen 固定 TND query + PA_BSND key: out [R, key.shape[2](恒1), outRowWidth]
    TORCH_CHECK(query_layout_str == "TND",
                "layout_query only supported TND for coarse_screen, but got ",
                query_layout_str);
    TORCH_CHECK(key_layout_str == "PA_BSND",
                "layout_key only supported PA_BSND for coarse_screen, but got ",
                key_layout_str);
    TORCH_CHECK(actual_seq_lengths_query.has_value(),
                "actual_seq_lengths_query must be provided for TND coarse_screen.");
    int64_t batchSize = actual_seq_lengths_query->size(DIM_0);
    // 输出单行宽 = Align8(sparse_count + 2*g − 1):行 = 排名 top-min(lo,4096) + 行尾
    // 「组窗口∪自有」段(宽 ≤ 2*g − 1)+ -1 pad
    int64_t g = row_weights.size(DIM_1);
    int64_t outRowWidth = (sparse_count + 2 * g - 1 + ALIGN_8 - 1) / ALIGN_8 * ALIGN_8;
    output_size = {batchSize, key.size(DIM_2), outRowWidth};

    return at::empty(output_size, query.options().dtype(at::kInt));
}

at::Tensor npu_indexer_coarse_screen(
    const at::Tensor& query, const at::Tensor& key, const at::Tensor& weights,
    const at::Tensor& row_weights,
    const c10::optional<at::Tensor>& actual_seq_lengths_query,
    const c10::optional<at::Tensor>& actual_seq_lengths_key,
    const c10::optional<at::Tensor>& block_table, c10::string_view layout_query,
    c10::string_view layout_key, int64_t sparse_count)
{
    TORCH_CHECK(query.numel() > 0, "Tensor query is empty.");
    TORCH_CHECK(key.numel() > 0, "Tensor key is empty.");
    TORCH_CHECK(weights.numel() > 0, "Tensor weights is empty.");
    TORCH_CHECK(row_weights.numel() > 0, "Tensor row_weights is empty.");

    std::string query_layout_str = std::string(layout_query);
    std::string key_layout_str = std::string(layout_key);

    at::Tensor sparse_indices_out = construct_indexer_coarse_screen_output_tensor(
        query, key, row_weights, actual_seq_lengths_query, sparse_count, query_layout_str, key_layout_str);

    char* query_layout_ptr = const_cast<char*>(query_layout_str.c_str());
    char* key_layout_ptr = const_cast<char*>(key_layout_str.c_str());

    EXEC_NPU_CMD(aclnnIndexerCoarseScreen, query, key, weights, row_weights,
                 actual_seq_lengths_query, actual_seq_lengths_key, block_table,
                 query_layout_ptr, key_layout_ptr, sparse_count,
                 sparse_indices_out);

    return sparse_indices_out;
}
}  // namespace vllm_ascend

#endif  // INDEXER_COARSE_SCREEN_TORCH_ADPT_H
