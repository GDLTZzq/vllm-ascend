# fused_li_manage_mtp

这是标准化 `fused_li_manage_mtp` 的独立开发与调优工程。算子支持每请求
1–14 路 query（MTP0–MTP13），并在一次 NPU 调用中融合 Lightning Indexer
TopK、首次 resident 建表、卸载稳态 hit/miss、淘汰、搬运列表生成和持久化
pool 更新。

- 完整接口、状态机和约束：
  [`FUSED_LI_MANAGE_MTP_INTERFACE.md`](FUSED_LI_MANAGE_MTP_INTERFACE.md)
- 历史与当前性能结果：
  [`FUSED_LI_MANAGE_MTP_PERFORMANCE.md`](FUSED_LI_MANAGE_MTP_PERFORMANCE.md)

## 关键规格

- Q 范围：`1..14`，对应 MTP0–MTP13。
- 状态：`-3` 非卸载、`-2` 首次卸载、`-1` 卸载稳态；同一 batch 可混合 Q
  和状态。
- cache budget：`C <= 32640`（32767 以下最大的 128 倍数），并受逐请求
  `Q`、`L` 动态约束。
- source capacity：最大 `2^21 = 2,097,152`。
- 内部 payload：`[slot15 | source_low17]`；长序列用 FP32 排序 key 低 4 bit
  携带 `source_high4`。
- `miss_src_ids/miss_dst_slots`：`int32[B,32768]`，仅前
  `miss_counts[b]` 项有效。

## 算子接口

```python
torch.ops.nanovllm_dsa.fused_li_manage_mtp.default(
    index_weights,                 # bf16/fp16 [T, N]
    query_dequant_scale,           # fp32 [T, N]
    query,                         # bf16/fp16 [T, N, 128]
    index_key_dequant_scale,       # fp32 [INDEX_BLOCKS, 128, 1]
    index_key_cache,               # bf16/fp16 [INDEX_BLOCKS, 128, 1, 128]
    index_block_table,             # int32 [B, INDEX_MAX_BLOCKS]
    actual_seq_lengths_query,      # int32 [B]，累计 query 结束位置
    actual_seq_lengths_key,        # int32 [B]
    offload_seq_lengths_key,       # int32 [B]，稳定 source prefix L
    num_cache_tokens,              # int32 [B]，cache budget C
    request_state,                 # int32 [B]，仅允许 -3/-2/-1
    req_pool_entries,              # int32 [B]
    cache_slots_pool,              # int32 [POOL_SIZE, SOURCE_CAPACITY]，原地更新
    topk_src_ids,                  # int32 [T, 1, 2048]，输出
    topk_dst_slots,                # int32 [T, 1, 2048]，输出
    topk_miss_counts,              # int32 [T]，输出
    miss_src_ids,                  # int32 [B, 32768]，输出
    miss_dst_slots,                # int32 [B, 32768]，输出
    miss_counts,                   # int32 [B]，输出
) -> None
```

其中 `T = sum(Q_i)`，每请求 `1 <= Q_i <= 14`，`N` 只能为 32 或 64；
`query/index_weights/index_key_cache` 必须具有相同的 BF16 或 FP16 dtype。算子没有
返回 tensor，pool 和全部输出均由调用方预分配并原地写入。`miss_src_ids` 与
`miss_dst_slots` 必须使用新的 `[B,32768]` ABI，旧 `[B,16384]` shape 会被明确拒绝。

状态语义：`-3` 执行非卸载 LI 并恢复 identity pool；`-2` 忽略旧映射并首次建立
包含 C 个 resident token 的 sparse pool；`-1` 在已有 sparse pool 上执行稳态
hit/miss、淘汰和搬运列表生成。完整逐 tensor 语义、因果长度和动态约束见
[`FUSED_LI_MANAGE_MTP_INTERFACE.md`](FUSED_LI_MANAGE_MTP_INTERFACE.md)。

## 环境与编译

```bash
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-8.5.1
export CANN_INSTALL_PATH=/usr/local/Ascend/cann-8.5.1
export PYTHONPATH=$PWD:$PYTHONPATH
export PYTHONUNBUFFERED=1
export SOC_VERSION=ascend910_9391
export NANOVLLM_CANN_BUILD_JOBS=64
export NANOVLLM_EXT_BUILD_JOBS=1

bash scripts/build_nanovllm_ops.sh
```

运行前设置：

```bash
unset NANOVLLM_CUST_OPAPI_LIB
unset ASCEND_CUSTOM_OPP_PATH

export ASCEND_HOME_PATH=/usr/local/Ascend/cann-8.5.1
export PYTHONUNBUFFERED=1
export PYTHONPATH=$PWD:$PYTHONPATH
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ASCEND_LAUNCH_BLOCKING=0
export ASCEND_RT_VISIBLE_DEVICES=4
```

## 运行 UT

### 短序列完整回归

`all` 会执行 payload codec、Q8/Q12/Q14 宽路、严格稳态语义、真实淘汰、首次
卸载、correctness、lifecycle、invalid 和基础 perf。BF16/FP16 均需执行：

```bash
for DTYPE in bf16 fp16; do
  python3 ut_ops/test_fused_li_manage_mtp.py \
    --mode all \
    --device npu:0 \
    --heads 32 \
    --dtype "$DTYPE" \
    --q-pattern 1,2,3,4,5,6,7 \
    --source-capacity 16384 \
    --offload-len 8192 \
    --cache-tokens 8192 \
    --warmup 10 \
    --iters 300 \
    --seed 7
done
```

`all` 中的固定 wide case 已覆盖 Q8/Q12/Q14、C32640、Q14 全不重叠
union=28672、Q15 拒绝和旧 `[B,16384]` ABI 拒绝。调用方 Q1–14 混合布局可再用
以下轻量 correctness 验证：

```bash
python3 ut_ops/test_fused_li_manage_mtp.py \
  --mode correctness \
  --device npu:0 \
  --heads 32 \
  --dtype bf16 \
  --q-pattern 1,2,3,4,5,6,7,8,9,10,11,12,13,14 \
  --source-capacity 32896 \
  --offload-len 32768 \
  --cache-tokens 32640 \
  --warmup 1 \
  --iters 1 \
  --seed 7
```

### 长序列完整回归

`long-regression` 单独覆盖 Q1/4/7/8/12/14 的 `-3/-2/-1`、真实 `L>C`
淘汰、首次卸载、三条 lifecycle、混合 Q/状态、replacement、first decode、
21-bit 边界、4-bit key-tag cutoff 和代表性 invalid。它必须满足
`source_capacity = offload_len + 128`，不应把这些大张量参数直接传给 `all`。

`131072` 是不使用 key tag 的直接编码上界；`131200` 是第一个 128 对齐的长路径
验收点。最大卸载 prefix 为 `2097024 = 2^21 - 128`。

```bash
# 128K 直接编码边界使用普通 correctness。
python3 ut_ops/test_fused_li_manage_mtp.py \
  --mode correctness \
  --device npu:0 \
  --heads 32 \
  --dtype bf16 \
  --q-pattern 1,4,7 \
  --source-capacity 131072 \
  --offload-len 130944 \
  --cache-tokens 14336 \
  --warmup 1 \
  --iters 1 \
  --seed 7

# 128K+128、256Ki、1Mi 和最大 2Mi-128 长路径。
for SOURCE_LEN in 131200 262144 1048576 2097024; do
  for DTYPE in bf16 fp16; do
    python3 ut_ops/test_fused_li_manage_mtp.py \
      --mode long-regression \
      --device npu:0 \
      --heads 32 \
      --dtype "$DTYPE" \
      --source-capacity $((SOURCE_LEN + 128)) \
      --offload-len "$SOURCE_LEN" \
      --warmup 1 \
      --iters 1 \
      --seed 7
  done
done
```

### `--mode` 支持

- `correctness`：按 `--q-pattern` 验证 `-3/-2/-1` 单状态和混合状态的 TopK、
  slot、union、miss 和 pool mapping。
- `lifecycle`：验证 `-2 -> -1 -> -1`、`-3 -> -1`、
  `-3 -> -2 -> -1`。
- `replacement-regression` / `steady-semantic-regression`：验证 `L>C` 真实淘汰、
  victim 合法唯一、slot 双射、随机 mapping/block table 和重复稳态。
- `first-decode-regression`：验证 `L>C` 首次 resident 初始化及 identity row 重建。
- `payload-codec-regression`：验证 15/17-bit payload、slot/source 边界和 invalid。
- `wide-route-regression`：验证每个 Q8–Q14 的 `-3/-2/-1`、C32640，以及 Q14
  每路 2048 miss、全不重叠 union=28672。
- `wide-compact-union-regression`：验证 Q8/Q12/Q14 普通宽路 union 与 route
  destination publication。
- `occurrence-regression`：验证 Q1–Q14、多档 miss、高 occurrence、Q4 的实际
  `4092/4096/4100` 快路边界、历史 `6908/6912/6916` 邻点，以及 Q4/Q7/Q14
  最大 disjoint source-join fallback。
- `key-tag-regression`：验证 4-bit tag、0–16 ULP 和跨 `2^17` cutoff；也由
  `long-regression` 自动执行。
- `long-regression`：长序列状态、生命周期、宽路、淘汰、首次卸载和边界集合。
- `invalid`：验证 Host 静态拒绝和 kernel 动态保护。
- `perf` / `standard-mtp-perf` / `first-decode-perf` / `mtp-perf`：基础、非卸载、
  首次卸载和卸载稳态计时。
- `mixed-mtp-perf` / `mixed-standard-mtp-perf`：同批混合 Q 的卸载/非卸载计时。
- `mixed-state-perf`：同一批次同时混合 Q 和 `-3/-2/-1` 状态计时。
- `all`：短序列完整集合；不代替 `long-regression`。

高 occurrence 边界可单独复测；Q1–Q14 均有代表用例，occurrence 覆盖低值、
2049、4096 邻域和 6908/6912/6916；Q4 额外精确覆盖 4092、4096、4100。
该专项把物理 source capacity 保持在 `2^17`，只验证 occurrence publication；
长序列 key-tag 扰动由 `key-tag-regression` 和 `long-regression` 单独验证：

```bash
python3 ut_ops/test_fused_li_manage_mtp.py \
  --mode occurrence-regression \
  --device npu:0 \
  --heads 32 \
  --dtype bf16 \
  --source-capacity 131072 \
  --offload-len 130944 \
  --warmup 1 \
  --iters 1 \
  --seed 7
```

测试失败时，union 断言会报告首个错误索引及实际/期望窗口；TopK destination
断言会额外报告 route、source、miss/hit 类型、最终 cache slot 和相邻 slot 窗口。
这两类诊断只在失败时产生，不改变算子接口或性能计时。

## 性能测试脚本

所有脚本使用 NPU event，报告 median/p95。建议同机连续跑三轮，比较
`overhead = fused - official` 的三轮中位数。长序列只测 BS=1/2/5；BS=16/24
的大张量显存与频率影响较大，不作为常规验收口径。

### 非卸载 `-3`：MTP0–10

```bash
for SOURCE_LEN in 65536 131072 131200 524288 1048576 2097024; do
  if [ "$SOURCE_LEN" -le 131200 ]; then
    BATCHES="1 2 5 16 24"
  else
    BATCHES="1 2 5"
  fi

  for MTP in $(seq 0 10); do
    Q=$((MTP + 1))
    for BS in $BATCHES; do
      python3 ut_ops/test_fused_li_manage_mtp.py \
        --mode standard-mtp-perf \
        --device npu:0 \
        --q-pattern "$Q" \
        --batch-size "$BS" \
        --heads 32 \
        --dtype bf16 \
        --source-len "$SOURCE_LEN" \
        --warmup 10 \
        --iters 300 \
        --seed 7
    done
  done
done
```

### 卸载稳态 `-1`：MTP0–10

Q1–7 沿用成熟口径；Q8–11 使用 `C=Q*2048`、每路 miss=200、union=600。

```bash
for SOURCE_LEN in 65536 131072 131200 524288 1048576 2097024; do
  if [ "$SOURCE_LEN" -le 131200 ]; then
    BATCHES="1 2 5 16 24"
  else
    BATCHES="1 2 5"
  fi

  for MTP in $(seq 0 10); do
    Q=$((MTP + 1))
    case "$MTP" in
      0)     CACHE_TOKENS=8192;  UNION_MISSES=200 ;;
      1|2|3) CACHE_TOKENS=12288; UNION_MISSES=300 ;;
      4|5)   CACHE_TOKENS=12288; UNION_MISSES=400 ;;
      6)     CACHE_TOKENS=14336; UNION_MISSES=400 ;;
      *)     CACHE_TOKENS=$((Q * 2048)); UNION_MISSES=600 ;;
    esac

    for BS in $BATCHES; do
      python3 ut_ops/test_fused_li_manage_mtp.py \
        --mode mtp-perf \
        --device npu:0 \
        --q-pattern "$Q" \
        --batch-size "$BS" \
        --heads 32 \
        --dtype bf16 \
        --source-len "$SOURCE_LEN" \
        --cache-tokens "$CACHE_TOKENS" \
        --query-miss-count 200 \
        --union-miss-count "$UNION_MISSES" \
        --query-noise 0.25 \
        --warmup 10 \
        --iters 300 \
        --seed 7
    done
  done
done
```

### 首次卸载 `-2`：MTP0–10

```bash
for SOURCE_LEN in 65536 131072 131200 524288 1048576 2097024; do
  if [ "$SOURCE_LEN" -le 131200 ]; then
    BATCHES="1 2 5 16 24"
  else
    BATCHES="1 2 5"
  fi

  for MTP in $(seq 0 10); do
    Q=$((MTP + 1))
    case "$MTP" in
      0)     CACHE_TOKENS=8192 ;;
      1|2|3|4|5) CACHE_TOKENS=12288 ;;
      6)     CACHE_TOKENS=14336 ;;
      *)     CACHE_TOKENS=$((Q * 2048)) ;;
    esac

    for BS in $BATCHES; do
      python3 ut_ops/test_fused_li_manage_mtp.py \
        --mode first-decode-perf \
        --device npu:0 \
        --q-pattern "$Q" \
        --batch-size "$BS" \
        --heads 32 \
        --dtype bf16 \
        --source-capacity $((SOURCE_LEN + 128)) \
        --offload-len "$SOURCE_LEN" \
        --cache-tokens "$CACHE_TOKENS" \
        --warmup 10 \
        --iters 300 \
        --seed 7
    done
  done
done
```

### 混合 MTP、混合 state

同一 invocation 同时包含不同 Q，并覆盖两状态组合 `-1/-2`、`-1/-3`、
`-2/-3` 以及完整的 `-3/-2/-1` 组合。`-1` 请求使用具有精确 union miss 数的
resident cache，`-2` 请求每轮从空 pool row 开始，`-3` 请求每轮从未初始化 row
开始，因此计时包含各状态的真实管理开销。

```bash
for SPEC in \
  "1,4|-1,-2" \
  "1,4|-1,-3" \
  "1,4|-2,-3" \
  "1,4,7|-3,-2,-1" \
  "1,4,7,8,11|-3,-1,-2,-3,-1" \
  "1,2,3,4,5,6,7,8,9,10,11|-3,-1,-1,-2,-1,-3,-1,-2,-1,-3,-1"; do
  PATTERN=${SPEC%%|*}
  STATES=${SPEC#*|}
  for BS in 1 2 5; do
    python3 ut_ops/test_fused_li_manage_mtp.py \
      --mode mixed-state-perf \
      --device npu:0 \
      --q-pattern "$PATTERN" \
      --state-pattern="$STATES" \
      --batch-size "$BS" \
      --heads 32 \
      --dtype bf16 \
      --source-capacity 65664 \
      --offload-len 65536 \
      --cache-tokens 32640 \
      --union-miss-count 600 \
      --warmup 10 \
      --iters 300 \
      --seed 7
  done
done
```

Q14、每路 miss=2048、全不重叠 union=28672 的精确边界已由
`wide-route-regression` 做正确性验收。常规性能脚本不把该极端构造混入随机
workload，避免把数据生成失败误判为 kernel 性能问题。
