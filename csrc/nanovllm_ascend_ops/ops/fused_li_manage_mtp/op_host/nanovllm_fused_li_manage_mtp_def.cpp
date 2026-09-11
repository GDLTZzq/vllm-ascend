/** Copyright (c) 2026 Huawei Technologies Co., Ltd. */
#include "register/op_def_registry.h"

namespace ops {
class NanovllmFusedLiManageMtp : public OpDef {
public:
    explicit NanovllmFusedLiManageMtp(const char *name) : OpDef(name)
    {
        // Keep every declaration explicit.  CANN 8.5 opbuild extracts the
        // schema from these calls and does not reliably recognize declarations
        // hidden behind helper lambdas or loops.
        this->Input("index_weights").ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("query_dequant_scale").ParamType(REQUIRED)
            .DataTypeList({ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND}).AutoContiguous();
        // fp8 E4M3 is carried as DT_UINT8 (a byte-identical one-byte
        // reinterpretation), mirroring the a5 ops.json practice of standing in
        // uint8 for fp8.  ge::DT_FLOAT8_E4M3FN must NOT be declared here (the
        // CANN 9.1 aclnnInner autogen segfaults on it), and a bf16/fp16-only
        // list makes the aclnn dtype gate reject an fp8 graph before the kernel
        // is reached.  The torch adapter reinterprets an fp8 tensor to uint8
        // ahead of the aclnn call, and the arch35/Ascend950 host fp8 branch in
        // fused_li_manage_mtp_tiling.cpp keys off DT_UINT8 at tiling time.
        //
        // query/index_key_cache use DataTypeList (independent allowed dtypes),
        // NOT DataType (parallel combination rows): the uint8 carrier must pair
        // with a bf16 index_weights, which cannot be expressed as a positional
        // row.  opbuild zips equal-length DataType lists across inputs and
        // segfaults (Error 139) when they differ, so mixing a 3-entry list here
        // with the 2-entry index_weights list is fatal.  DataTypeList entries
        // (already used by every scale/metadata input) carry no such coupling.
        this->Input("query").ParamType(REQUIRED)
            .DataTypeList({ge::DT_BF16, ge::DT_FLOAT16, ge::DT_UINT8})
            .FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("index_key_dequant_scale").ParamType(REQUIRED)
            .DataTypeList({ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("index_key_cache").ParamType(REQUIRED)
            .DataTypeList({ge::DT_BF16, ge::DT_FLOAT16, ge::DT_UINT8})
            .FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("index_block_table").ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("actual_seq_lengths_query").ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("actual_seq_lengths_key").ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("offload_seq_lengths_key").ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("num_cache_tokens").ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("request_state").ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("req_pool_entries").ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("cache_slots_pool").ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND}).AutoContiguous();

        this->Output("topk_src_ids").ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND});
        this->Output("topk_dst_slots").ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND});
        this->Output("topk_miss_counts").ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND});
        this->Output("miss_src_ids").ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND});
        this->Output("miss_dst_slots").ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND});
        this->Output("miss_counts").ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND});
        this->Output("cache_slots_pool_out").ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND});

        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true).DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true).DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false).PrecisionReduceFlag(true)
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn")
            .ExtendCfgInfo("jitCompile.flag", "static_false,dynamic_false");
        this->AICore().AddConfig("ascend910b", config);
        this->AICore().AddConfig("ascend910_93", config);
        this->AICore().AddConfig("ascend950", config);
    }
};
OP_ADD(NanovllmFusedLiManageMtp);
} // namespace ops
