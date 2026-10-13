# FlexPrefill 各层 head 算法使用统计

用户 2026-10-02 请求统计本轮 FlexPrefill 每层 Vertical-Slash 和另一算法的使用头数。
原结果只记录 baseline 配置，未记录动态分支或保存中间 Q/K；因此对完全相同的 200 个输入重放 prefill。
复用全部 32 层、原 BF16 模型、RoPE/GQA、完整 KV、MLP、torch/FA2/Triton、gamma=.95、tau=.1、min_budget=1024、block_size=128。
按原 FlexPrefill 的 10 个长度平衡 shard 并行，各 1 张 H100-80、8 CPU、128 GB，调度上限 2 小时，push QoS。
本次仅采集原官方 prefill 的分支；每条样本的最终 prompt logits 必须与原结果 **逐位一致**。

`observer.py` 包装调用入口，并以 Python profile return event 读取官方 `get_active_blocks` 的局部变量。
官方源码和算法原样执行；读取 `block_sparse_mask` 和 `kl_div`，无额外 QK 计算、mask 修改、输出替换。
`block_sparse_mask=True` 为 Query-Aware Block-Sparse（√JS < .1），False 为 Vertical-Slash。
Query-Aware 分支仍保留最小预算的 Vertical-Slash scaffold；统计的是分支选择，不能解释为它完全不用 Vertical-Slash。
分类随样本、层和 head 变化，报告平均头数、最小/最大头数、每 head 的使用频次及四任务分项。

每完成一个样本立即原子保存 `outputs/worker-N/<key>.json`，含 32 层实际 head IDs、√JS、原结果 hash 和 logits 一致性。
最后完成的 worker 自动 CPU 校验全部 200 样本、204800 次 head 决策，输出：

- `outputs/independent-audit.json`：全部统计、各任务统计及逐 head 频次。
- `outputs/layer-head-counts.csv`：32 层总体平均值和范围。
- `outputs/task-layer-head-counts.csv`：四任务分别的 32 层统计。

用户此前已要求速度优先、不限制总体资源；此次统计重放的源码与来源 hashes 单独锁定。
原 1400 条生成结果、配置和审计保留原样。
