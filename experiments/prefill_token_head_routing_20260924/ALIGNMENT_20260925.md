# AutoBlock 对齐记录（静态修改，尚未执行）

本轮按用户选择使用 **Llama-3.1-8B-Instruct + InfiniteBench**。用户要求实验运行前取得同意；本轮未执行测试、模型 forward、smoke、正式评测或作业提交。此前成功日志只证明此前冻结版本，不能验证本次 FA2 和计时改动。

## 对齐基准

本地 autoblock 源码基准为 `8b21a6a18056eadc00b6248bb319a4af3c6ccc4d`。复用 `experiments/run_shareprefill_ae3_infinitebench.py`，以 `experiments/run_llama_ae8_autoblock_formal.py` 的 Llama 设置为准。原文件未修改。

| 项目 | 对齐后的约束 |
|---|---|
| 模型/tokenizer | 已下载官方 Llama-3.1-8B-Instruct，revision `0e9e39f249a16976918f6564b8830bc894c89659`；运行时逐文件校验已有 provenance |
| 精度、原 attention | bf16、FlashAttention-2，不自动降级 SDPA/fp16 |
| 上下文、batch、seed | 131072（包含生成）、1、42 |
| 输入 | 原 task YAML、chat template、middle truncation、left padding、按任务保留生成预算 |
| 生成/评分 | 原 fixed_generate_until 和 lm_eval 流程；greedy，每题一次；原官方 task grader |
| 输入参照 | A/B/C 均需原 AutoBlock 的 online_metrics.jsonl；禁止用新 A 替代旧输入参照 |
| 主计时 | 原 GenerationMetricsRecorder：CUDA-event prefill、同步 wall total、decode=total-prefill |
| 额外记录 | 原样本日志 + token/停止/cache/pair 明细；明细文件写入在原主计时结束后 |
| 覆盖 | 候选十任务、每条件 3493 题；须先核验实际 filtered 数据，不能通过删题凑计数 |
| 资源 | 原 8 GPU 为任务级独立 worker；不视为新实验已获 8 GPU 授权 |

模型尺寸：32 层，hidden=4096，Q heads=32，KV heads=8，head_dim=128；Q/K/V/O 权重分别为 [4096,4096]/[1024,4096]/[1024,4096]/[4096,4096]，无投影 bias，无 Q/K norm。原 RoPE、scale、GQA 映射 h//4 保留。

## 唯一研究变量及内核实现

A 保留原 dense prefill。B/C 用原位置 i%32 路由，每 token 仅保留所选 Q head 输出，scatter 合并后显式重复拼接 32 次，再经过完整 o_proj。所有条件保留完整 QKV 投影、完整 prompt KV cache，并用原 dense FA2 decode。各层 cache 来自本方法实际 hidden states。

C 的每组 Q/K 同序且等长，直接使用 causal FA2，整个组一次调用，不能将 Q 单独切块后误用 FA2 的 causal 对齐。B 的 Q 是不连续原位置，K 为完整合法序列；普通 FA2 causal mask 不能表达，因此使用 SDPA 的显式原位置 mask。该内核区别属于算法实现，必须随性能结果报告。不得称 B/C 节省完整投影或 cache，也不得把 pair 缩减称为端到端加速。

`eligible_pairs` 为合法 causal pair 数；`matmul_pairs` 为包含被 mask 元素的矩形域大小，在 FA2 下是上界，不是实际 GPU 指令计数。原 recorder 中 block 指标不适用于 token routing（兼容输出可能为 0/NaN），报告使用 token-pair 字段。

保持原主计时公式。额外 decode-forward CUDA-event 时间仅为诊断，不能替换 canonical decode=total-prefill。明细 JSON 写入移出计时；路由检查、gather/scatter、pair 统计和内存记录仍有开销，计入运行。不能据此宣称与旧硬件/旧环境的绝对耗时可直接比较。原 recorder 仅识别长度>1的 prefill；正式适配器将拒绝单 token 输入，底层路由对单 token 的支持另行测试。

## 尚待核验与审批

- 原 filtered task YAML/数据与原 baseline 输入参照未完成本地核验。仅复用脚本或下载公开数据不足以证明输入一致。
- 原实验完整依赖版本未恢复。当前隔离环境 Transformers 5.7.0、Torch 2.11.0+cu126，不能声称与旧环境完全一致。
- FA2 尚未安装验证；历史 V100/fp16/SDPA smoke 不验证 bf16/FA2。需要经同意在兼容 GPU 上验证。
- `test_alignment.py` 已加入 mock FA2 的组内 mask/GQA/padding 契约测试，以及明细写入位于主计时之外的测试；**尚未运行**。mock 不能替代真实 FA2 数值/cache/生成测试。
- 下一次审批应先提供明确的小规模测试命令、GPU allocation、输入/生成上限、时间上限和新输出目录。取得同意后运行参考测试及真实模型接入、未来不变性、完整 cache 多步 decode 和生成 smoke。
- 正式方案仍为候选 A/B/C、十任务、单卡 worker；草案上限 1 H100-80GB、每条件 24h，三条件 72 GPU-h 是资源上限，不是实测成本。128k 显存和耗时尚未验证，正式运行范围/资源要另行审批。
- 输出位于本独立目录的新运行根下；condition/task 目录已存在即拒绝运行。提交脚本保留硬退出，提交前使用原子目录锁并保存 job ID；不自动重试不确定提交。

入口为 `run_infinitebench.py`，主计时桥接为 `aligned_recorder.py`，`recorder.py` 保留为此前开发实现，不用于本次正式对齐入口。`formal.sbatch.draft` 仍不可执行评测。没有模型效果或加速有效性结论。
