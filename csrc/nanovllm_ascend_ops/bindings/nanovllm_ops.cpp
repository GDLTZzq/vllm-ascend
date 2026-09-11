#include <acl/acl.h>
#include <torch/extension.h>
#include <torch/library.h>

#include "common/torch_adapter/op_api_common.h"

thread_local char g_hashBuf[kHashBufSize];
thread_local int g_hashOffset = 0;

#include "ops/fused_li_manage_mtp/fused_li_manage_mtp_torch_adpt.h"

namespace {

void fused_li_manage_mtp_torch_op(
    const at::Tensor& index_weights,
    const at::Tensor& query_dequant_scale,
    const at::Tensor& query,
    const at::Tensor& index_key_dequant_scale,
    const at::Tensor& index_key_cache,
    const at::Tensor& index_block_table,
    const at::Tensor& actual_seq_lengths_query,
    const at::Tensor& actual_seq_lengths_key,
    const at::Tensor& offload_seq_lengths_key,
    const at::Tensor& num_cache_tokens,
    const at::Tensor& request_state,
    const at::Tensor& req_pool_entries,
    at::Tensor cache_slots_pool,
    at::Tensor topk_src_ids,
    at::Tensor topk_dst_slots,
    at::Tensor topk_miss_counts,
    at::Tensor miss_src_ids,
    at::Tensor miss_dst_slots,
    at::Tensor miss_counts) {
  vllm_ascend::npu_fused_li_manage_mtp(
      index_weights, query_dequant_scale, query, index_key_dequant_scale,
      index_key_cache, index_block_table, actual_seq_lengths_query,
      actual_seq_lengths_key, offload_seq_lengths_key, num_cache_tokens,
      request_state, req_pool_entries, cache_slots_pool, topk_src_ids,
      topk_dst_slots, topk_miss_counts, miss_src_ids, miss_dst_slots,
      miss_counts);
}

void fused_li_manage_mtp_meta(
    const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
    const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
    const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
    at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
    at::Tensor) {}

}  // namespace

TORCH_LIBRARY(nanovllm_dsa, ops) {
  ops.def(
      "fused_li_manage_mtp(Tensor index_weights,"
      " Tensor query_dequant_scale, Tensor query,"
      " Tensor index_key_dequant_scale, Tensor index_key_cache,"
      " Tensor index_block_table, Tensor actual_seq_lengths_query,"
      " Tensor actual_seq_lengths_key, Tensor offload_seq_lengths_key,"
      " Tensor num_cache_tokens, Tensor request_state,"
      " Tensor req_pool_entries, Tensor(a!) cache_slots_pool,"
      " Tensor(b!) topk_src_ids, Tensor(c!) topk_dst_slots,"
      " Tensor(d!) topk_miss_counts, Tensor(e!) miss_src_ids,"
      " Tensor(f!) miss_dst_slots, Tensor(g!) miss_counts) -> ()");
}

TORCH_LIBRARY_IMPL(nanovllm_dsa, PrivateUse1, ops) {
  ops.impl("fused_li_manage_mtp", &fused_li_manage_mtp_torch_op);
}

TORCH_LIBRARY_IMPL(nanovllm_dsa, Meta, ops) {
  ops.impl("fused_li_manage_mtp", &fused_li_manage_mtp_meta);
}

PYBIND11_MODULE(_C, m) {
  m.doc() = "standardized GLM fused_li_manage_mtp registration";
}
