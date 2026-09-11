#ifndef NANOVLLM_FUSED_LI_MANAGE_MTP_TORCH_ADPT_H_
#define NANOVLLM_FUSED_LI_MANAGE_MTP_TORCH_ADPT_H_

namespace vllm_ascend {

inline void npu_fused_li_manage_mtp(
    const at::Tensor& index_weights, const at::Tensor& query_dequant_scale,
    const at::Tensor& query, const at::Tensor& index_key_dequant_scale,
    const at::Tensor& index_key_cache, const at::Tensor& index_block_table,
    const at::Tensor& actual_seq_lengths_query,
    const at::Tensor& actual_seq_lengths_key,
    const at::Tensor& offload_seq_lengths_key,
    const at::Tensor& num_cache_tokens, const at::Tensor& request_state,
    const at::Tensor& req_pool_entries, at::Tensor cache_slots_pool,
    at::Tensor topk_src_ids, at::Tensor topk_dst_slots,
    at::Tensor topk_miss_counts, at::Tensor miss_src_ids,
    at::Tensor miss_dst_slots, at::Tensor miss_counts) {
  constexpr int64_t kTopK = 2048;
  constexpr int64_t kMissCapacity = 32768;
  constexpr int64_t kBlockSize = 128;
  TORCH_CHECK(query.dim() == 3 &&
                  (query.size(1) == 32 || query.size(1) == 64) &&
                  query.size(2) == 128 && query.size(0) > 0,
              "LIM-MTP query must be [T, H, 128], H=32 or 64 and T>0.");
  TORCH_CHECK(query.device().is_privateuseone(), "LIM-MTP tensors must be on NPU.");
  const int64_t total_queries = query.size(0);
  TORCH_CHECK(index_weights.dim() == 2 && index_weights.size(0) == total_queries &&
                  index_weights.size(1) == query.size(1),
              "index_weights must be [T, H].");
  TORCH_CHECK(query_dequant_scale.sizes() == index_weights.sizes(),
              "query_dequant_scale must be [T, H].");
  TORCH_CHECK(index_key_cache.dim() == 4 && index_key_cache.size(0) > 0 &&
                  index_key_cache.size(1) == kBlockSize &&
                  index_key_cache.size(2) == 1 && index_key_cache.size(3) == 128,
              "index_key_cache must be [blocks, 128, 1, 128].");
  TORCH_CHECK(index_key_dequant_scale.dim() == 3 &&
                  index_key_dequant_scale.size(0) == index_key_cache.size(0) &&
                  index_key_dequant_scale.size(1) == kBlockSize &&
                  index_key_dequant_scale.size(2) == 1,
              "index_key_dequant_scale must be [blocks, 128, 1].");
  TORCH_CHECK(index_block_table.dim() == 2 && index_block_table.size(0) > 0 &&
                  index_block_table.size(1) > 0 && index_block_table.size(1) <= (1 << 14),
              "index_block_table must be non-empty [B, max_blocks], max_blocks<=16384.");
  const int64_t batch_size = index_block_table.size(0);
  auto check_batch_vector = [batch_size](const at::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.dim() == 1 && tensor.size(0) == batch_size, name, " must be [B].");
  };
  check_batch_vector(actual_seq_lengths_query, "actual_seq_lengths_query");
  check_batch_vector(actual_seq_lengths_key, "actual_seq_lengths_key");
  check_batch_vector(offload_seq_lengths_key, "offload_seq_lengths_key");
  check_batch_vector(num_cache_tokens, "num_cache_tokens");
  check_batch_vector(request_state, "request_state");
  check_batch_vector(req_pool_entries, "req_pool_entries");
  TORCH_CHECK(cache_slots_pool.dim() == 2 && cache_slots_pool.size(0) > 0 &&
                  cache_slots_pool.size(1) == index_block_table.size(1) * kBlockSize,
              "cache_slots_pool width must equal max_blocks*128.");
  TORCH_CHECK(topk_src_ids.dim() == 3 && topk_src_ids.size(0) == total_queries &&
                  topk_src_ids.size(1) == 1 && topk_src_ids.size(2) == kTopK &&
                  topk_dst_slots.sizes() == topk_src_ids.sizes(),
              "topk outputs must be [T, 1, 2048].");
  TORCH_CHECK(topk_miss_counts.dim() == 1 && topk_miss_counts.size(0) == total_queries,
              "topk_miss_counts must be [T].");
  TORCH_CHECK(miss_src_ids.dim() == 2 && miss_src_ids.size(0) == batch_size &&
                  miss_src_ids.size(1) == kMissCapacity &&
                  miss_dst_slots.sizes() == miss_src_ids.sizes(),
              "miss outputs must be [B, 32768].");
  check_batch_vector(miss_counts, "miss_counts");

  // fp8 (float8_e4m3fn) query/index_key_cache carries a REQUIRED fp32 dequant
  // scale that the arch35 engine consumes; weights stay unquantized.  The A3
  // (arch22) path never sees fp8, so this widening only admits the A5 shape.
  const auto data_dtype = query.scalar_type();
  TORCH_CHECK(data_dtype == index_key_cache.scalar_type(),
              "query and index_key_cache must share dtype.");
  TORCH_CHECK(data_dtype == at::kHalf || data_dtype == at::kBFloat16 ||
                  data_dtype == at::kFloat8_e4m3fn,
              "query/index_key_cache must be fp16, bf16 or float8_e4m3fn.");
  TORCH_CHECK(index_weights.scalar_type() == at::kHalf ||
                  index_weights.scalar_type() == at::kBFloat16,
              "index_weights must be fp16 or bf16.");
  TORCH_CHECK(query_dequant_scale.scalar_type() == at::kFloat &&
                  index_key_dequant_scale.scalar_type() == at::kFloat,
              "dequant scales must be fp32.");
  const at::Tensor int_tensors[] = {
      index_block_table, actual_seq_lengths_query, actual_seq_lengths_key,
      offload_seq_lengths_key, num_cache_tokens, request_state, req_pool_entries,
      cache_slots_pool, topk_src_ids, topk_dst_slots, topk_miss_counts,
      miss_src_ids, miss_dst_slots, miss_counts};
  for (const auto& tensor : int_tensors) {
    TORCH_CHECK(tensor.scalar_type() == at::kInt, "metadata and outputs must be int32.");
  }
  const at::Tensor all_tensors[] = {
      index_weights, query_dequant_scale, query, index_key_dequant_scale,
      index_key_cache, index_block_table, actual_seq_lengths_query,
      actual_seq_lengths_key, offload_seq_lengths_key, num_cache_tokens,
      request_state, req_pool_entries, cache_slots_pool, topk_src_ids,
      topk_dst_slots, topk_miss_counts, miss_src_ids, miss_dst_slots, miss_counts};
  const auto device = query.device();
  for (const auto& tensor : all_tensors) {
    TORCH_CHECK(tensor.device() == device, "all LIM-MTP tensors must be on the same NPU.");
    TORCH_CHECK(tensor.is_contiguous(), "all LIM-MTP tensors must be contiguous.");
  }

  // fp8 E4M3 is carried as uint8 (see the op def comment): opbuild_gen_inner
  // segfaults on ge::DT_FLOAT8_E4M3FN, so the def/proto only advertises uint8,
  // and the aclnn dtype gate would otherwise reject the fp8 tensor.  The byte
  // pattern is identical, and the arch35 kernel reads fp8 regardless of dtype.
  const bool is_fp8 = data_dtype == at::kFloat8_e4m3fn;
  const at::Tensor query_arg = is_fp8 ? query.view(at::kByte) : query;
  const at::Tensor key_arg =
      is_fp8 ? index_key_cache.view(at::kByte) : index_key_cache;

  auto keepalive = std::make_tuple(
      index_weights, query_dequant_scale, query, query_arg,
      index_key_dequant_scale, index_key_cache, key_arg, index_block_table,
      actual_seq_lengths_query, actual_seq_lengths_key,
      offload_seq_lengths_key, num_cache_tokens, request_state,
      req_pool_entries, cache_slots_pool, topk_src_ids, topk_dst_slots,
      topk_miss_counts, miss_src_ids, miss_dst_slots, miss_counts);
  EXEC_NPU_CMD_ORDERED(
      aclnnNanovllmFusedLiManageMtp, keepalive,
      index_weights, query_dequant_scale, query_arg, index_key_dequant_scale,
      key_arg, index_block_table, actual_seq_lengths_query,
      actual_seq_lengths_key, offload_seq_lengths_key, num_cache_tokens,
      request_state, req_pool_entries, cache_slots_pool, topk_src_ids,
      topk_dst_slots, topk_miss_counts, miss_src_ids, miss_dst_slots,
      miss_counts, cache_slots_pool);
}

}  // namespace vllm_ascend
#endif
