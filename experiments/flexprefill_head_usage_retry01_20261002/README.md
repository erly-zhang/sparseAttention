# FlexPrefill 各层 head 算法使用统计

用户 2026-10-02 请求统计本轮 FlexPrefill 每层 Vertical-Slash 和另一算法的使用头数。
原结果只记录 baseline 配置，未记录动态分支或保存中间 Q/K；因此对完全相同的 200 个输入重放 prefill。
复用全部 32 层、原 BF16 模型、RoPE/GQA、完整 KV、MLP、torch/FA2/Triton、gamma=.95、tau=.1、min_budget=1024、block_size=128。
按原 FlexPrefill 的 10 个长度平衡 shard 并行，各 1 张 H100-80、8 CPU、128 GB，调度上限 2 小时，push QoS。
本次采集官方 prefill 的实际分支，每条输入重放两次，测量分类稳定性。
独立诊断发现两次不带观测器的普通 prefill 也不能做到 logits 逐位一致，具体数值差异来源尚未定位。
因此统计明确属于**相同配置重放**，无法从历史结果精确恢复原运行的 head 决策；历史得分保持原样。
保存每次重放的 logits，记录与历史结果的逐位一致性、最大误差和第一 token 是否相同，供独立审计。

`observer.py` 包装官方 √JS 的返回入口和 Query-Aware 分支的 `score_cover_idx` 调用入口。
官方源码和算法原样执行；仅保留 32 个 √JS 值，attention 完成后执行相同的 tensor `< tau` 比较，
用实际 Query-Aware 分支调用次数独立校验分类，无额外 QK 计算、mask 修改或输出替换。
`block_sparse_mask=True` 为 Query-Aware Block-Sparse（√JS < .1），False 为 Vertical-Slash。
Query-Aware 分支仍保留最小预算的 Vertical-Slash scaffold；统计的是分支选择，不能解释为它完全不用 Vertical-Slash。
分类随样本、层和 head 变化，报告平均头数、最小/最大头数、每 head 的使用频次及四任务分项。

每完成一个样本立即原子保存 `outputs/worker-N/<key>.json`，含两次的 32 层实际 head IDs、√JS、原结果 hash 和 logits 差异。
最后完成的 worker 自动 CPU 校验全部 200 样本、每轮 204800 次 head 决策、两轮的一致率，输出：

- `outputs/independent-audit.json`：全部统计、各任务统计及逐 head 频次。
- `outputs/layer-head-counts.csv`：32 层总体平均值和范围。
- `outputs/task-layer-head-counts.csv`：四任务分别的 32 层统计。

用户此前已要求速度优先、不限制总体资源；此次统计重放的源码与来源 hashes 单独锁定。
原 1400 条生成结果、配置和审计保留原样。
