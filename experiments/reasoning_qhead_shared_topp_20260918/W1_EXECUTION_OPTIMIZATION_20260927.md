# W1 代表头压缩：保持算法不变的执行优化

新方法：`qhead_shared_representative_compact_optimized_gpu`。原方法 `qhead_shared_representative_compact_recent_gpu`、完成的 MATH500 结果和正在运行的随机压缩多数据集实验均保留。

约束：原代表头私有随机流及抽取次数、当前W1 query、BF16 QK与缩放、FP32 softmax、稳定概率排序与并列规则、TopK960、原位置排序、完整prompt保护、recent64、reasoning1024、attention后每64步压缩、GQA、RoPE与生成采样设置不变。

## 已实现

1. 只在下一次attention会触发压缩时保存当前Q。普通decode不复制Q；压缩事件仍使用原方法的query保存函数，保持与原版相同的内存布局。
2. 使用两个完整容量 `P+1088` 的KV缓存。保护prompt只在首次压缩时复制到另一个缓存。
3. Triton一次gather K/V的reasoning部分到备用缓存，然后交换两个缓存；不再每次重拷贝prompt，也不再把压缩结果复制回原缓存。
4. batch过滤同时过滤两份缓存，避免只剩某个rollout时使用错误的prompt副本。原始位置跟踪和pair计数保持相同。

相对原来的 `P+1024` scratch，备用缓存多64个位置。Qwen3-4B/batch2/36层/BF16下，额外缓存约18MiB；实际峰值以测量为准。

未使用删除softmax、修改QK精度、改变代表头或排序规则等可能改变有限精度选择结果的简化。

## 验证与测量

探索目录：`results/w1_optimization_exploration_20260927`。

- CPU测试通过，覆盖长prompt、并列、recent=0、精确预算、多次压缩和batch过滤。
- GPU测试通过，包含真实Qwen3的32 Q heads / 8 KV heads / head_dim128 / BF16、1285步序列、长prompt和batch缩小。
- 逐步要求attention输出、KV、选中索引、概率分数、代表头随机状态、累计位置、pair计数位级一致；全局随机状态保持不变。
- 同卡交替微基准6轮，每轮512步：原版缓存路径中位0.088541ms/层/步，仅按需保存Q为0.060130ms，按需Q+双缓存为0.059363ms。完整优化相对原版减少约32.95%。这只测缓存路径，不含模型projection/MLP，不能当作整模型加速。

完整生成目录：`results/w1_optimized_pair_20260927`，作业 `1348063`。

先在分配的GPU运行所有相关回归测试，再执行原版A、优化版A、优化版B、原版B（ABBA）。每轮完全相同的MATH500 idx0/1/2、每题2条完整生成、seed42、temperature0.6、generation top-p0.95、总长度32768与两token不计时warmup。共6条唯一输出，每种方法重复两次；不能把重复计时当作更多独立准确率样本。

每轮输出必须与第一轮逐字相同；最终还核验input_tokens、input_ids_sha256、生成长度、EOS、selected/causal pairs、压缩事件和最终位置长度完全一致。否则失败并禁止加速结论。计时来自统一runner同步prefill/decode/total，排除加载、warmup、诊断文件序列化。

结果入口：`pipeline_status.json`、`correctness.log/json`、四个pass目录、`summary.json`、`REPORT.md`。本次不自动重跑500题；原版全量87.1%仍是原版结果，优化版只报告严格等价验证和三题受控计时。

## 完成结果（2026-09-27 14:38 EDT）

作业1348063完成，45项GPU correctness/回归全部通过。ABBA四轮完整生成与自动汇总完成；6条唯一输出跨4轮逐字一致，每轮6/6正确。input token哈希、生成长度、EOS、exact pair计数、压缩事件与最终缓存/累计位置全部一致。

| 指标 | 原W1 | 优化W1 |
|---|---:|---:|
| 平均decode秒/题（两条生成） | 107.790 | 104.493 |
| Decode tokens/s | 46.634 | 48.105 |
| 峰值allocated显存GiB | 9.090 | 8.814 |
| 两轮各自decode均值 | 108.587 / 106.993 | 105.054 / 103.933 |

本次受控三题计时的decode耗时降低3.06%。完整模型收益小于缓存路径微基准的32.95%；没有把微基准当作整模型加速，也未重跑500题。备用缓存理论多64个位置不等于整模型峰值必然增加；上表是整个生成过程的实测峰值，不单独归因于缓存容量。

最终报告：[REPORT.md](results/w1_optimized_pair_20260927/REPORT.md)。独立实现文件：`ra_harness/kvcompress/engine/representative_compact_optimized_cache.py`；统一runner使用 `--eviction_mode qhead_shared_representative_compact_optimized_gpu`，其他W1参数保持不变。
