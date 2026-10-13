# 代表头共享 Token+Vertical / Token+Vertical+Slash：速度优先正式评测

根据用户 2026-10-01 更新，本轮代表头方案只运行 Token+Vertical、Token+Vertical+Slash；
不限制总体计算资源，以尽快完成为目标。用户此前要求全部 32 层、四任务各 50 条。
本次回复授权开始修订方案；通过 CPU 检查后记录该授权，执行小 tensor GPU correctness 和正式实验。
上一版 `../representative_patterns_fast_retry01_20261001` 的配置与 CPU 检查记录完整保留。

## 7 组实验，共 1400 条正式生成

| 选头方式 | Token+Vertical | Token+Vertical+Slash |
|---|---|---|
| 3 个独立校准样本选头 | coverage_token_vertical | coverage_token_vertical_slash |
| 每层固定随机头 | random_token_vertical | random_token_vertical_slash |

主对照 Dense；另运行 FlexPrefill、MInference。AutoBlock-only 和 Token-only 本轮不执行。

- 模型：本地 Llama-3.1-8B-Instruct，全部 32 层、32 query heads、8 KV heads、head_dim=128，BF16。
- KV Retrieval / En.MC / En.QA / Code.Debug 各 50 条完整输入，固定 200 条、无截断，长度 69713–130876。
- 输入逐字节复用 `shared_topp_mask_bench_20260930/inputs`；token IDs、positions、mask、prompt、raw row 均校验 SHA256。
- 官方 chat template 和 InfiniteBench scorer；greedy、seed=42；最大生成长度分别 50 / 40 / 40 / 5。
- 全部方法保留完整 KV；decode 使用原 Transformers Dense FA2 forward。无训练、无 KV 压缩、位置不变。
- 新 Dense 全部 32 层使用历史 FP32 full attention 实现。旧 Dense 有 16 层显式 attention，因此新 Dense 重跑。

## 算法与固定选头

校准 3 个样本总计，与全部 200 条评测样本的 source identity 和 IDs hash 均不重叠：
KV source 355、En.MC source 153、En.QA source 224；长度 124135、87950、69704。
三个样本各自并行 Dense prefill；在每层最后 128 个 query 上，provider head 构造 causal Top-p=.95 mask，
以它保留 target heads 的注意力质量计算方向覆盖 `S[provider,target]`。
每样本选平均 outgoing coverage 最大的 head，三样本多数投票；平票按获胜票分数之和，再按较小 ID。
每层最终固定一个代表头，在两个 mask 变体及全部评测样本中复用。

Random 用 `random.Random(20261001)` 按层 0→31 各均匀采样 `randrange(32)` 一次，同样跨变体和样本固定。
只使用一个随机种子，不作跨 seed 方差结论。

基础 mask 沿用仓库 `autoblock` 的 token selector：query tile=128，1/3、2/3、最后、平均 query 四个 probe，
权重 `[.2,.3,.4,.1]`，固定 causal TopK=8192（不足时保留合法 keys），直接保留 token，不投影整块。
校准阶段的 Top-p=.95 用于选头，和正式 mask 的 TopK=8192 是不同步骤。
原 selector 已修正 `sorted=False` 后截取 rank 的因果前缀错误为 `sorted=True`，CPU regression 使用原 selector 验证。

使用 [FlexPrefill Algorithm 3](https://arxiv.org/html/2502.20766v1) 的方向求和与累计质量索引选择：

1. 用当前层代表头最后 128 个 query 对全长 K 计算 causal FP32 softmax。
2. Vertical 按 key 列求和并归一化，取累计质量首次达到 gamma=.95 的最小排序前缀，无额外 cap。
3. Slash 按原始位置距离 `d=i-j >= 0` 求和归一化，独立用 gamma=.95 选最小前缀；query i 加入 `j=i-d`。
4. 基础 token mask 与新增 pattern 作并集去重，共享给同层全部 query heads，各自用自己的 Q/K/V 和 GQA KV head 重归一化。

新增 pattern 保留 token 粒度，不直接用官方 128×128 block pooling 扩展。
CSR 是 query tile 候选 key 并集，Triton kernel 再检查逐 query slash membership，避免将 slash 扩大成共享列。
分别记录有效 mask pairs 和候选 QK pairs；独立 gamma 不保证合并 mask 在所有 query/heads 上保留 .95 质量。
执行时间包含选索引与诊断，本轮不据此宣称加速。

## 官方 baseline 的同输入重跑

`baseline-reuse-audit.json`：FlexPrefill 和 MInference 各有 32 条完整旧 KV 记录，
与本次 200 条固定输入各只有 1 条 hash 重合，旧生成上限 128 与本次 KV 50 不同，缺少另三任务。
这些记录不能直接用于四任务各 50 条的正式比较，因此两个 baseline 重跑。

- FlexPrefill 官方 `flex_prefill_attention`：gamma=.95、tau=.1、min_budget=1024、max_budget=None、block_size=128。
- MInference 官方 `minference_prefill_kernel`：MODEL2PATH 对应的 Llama-3.1-8B-Instruct `best_pattern_v2.json`，ratio=1、starting_layer=0。
- 统一 Llama 4.44 adapter 的 RoPE、cache、位置和 Dense decode，只替换 prefill 的官方 kernel，全部 32 层。

## 环境与验证

复用 Python 3.12.13 / torch 2.6.0+cu124 / transformers 4.44.0 / triton 3.0.0 /
flash-attn 2.7.4.post1 / numpy 2.1.3，以及历史 overlay、modules、compiler、共享库设置，无安装替换。
`source-lock.json` 冻结配置、运行脚本、原实现和官方源码/pattern/extension；inputs 和 worker plan 单独绑定 hashes。
CPU 检查包括原 selector 一致性、独立逐 query mask oracle、slash 方向/最小前缀/去重/实际 pairs、
GQA、自有概率、固定 random heads、完整 KV/Dense decode、全部 1400 对覆盖、审批先于 GPU/提交。
GPU correctness 在 calibration job 首先执行：BF16 Triton 对精确 FP32 reference（513 token、部分末尾 tile），
以及官方 baseline 的不同常量 V 验证 GQA 映射（16384 token、层 0/15/31）。

## 速度优先调度

读取当前 Slurm 配额后使用账户允许的 `push` 高优先级 QoS，资源可用时尽快启动；不设总体 GPU-hours cap。

1. Calibration array 0–2，各 1 张 H100-80GB、8 CPU、128GB RAM、2h 调度上限，各处理一个校准样本。
2. 最后完成的校准 worker 释放模型显存后直接做 CPU aggregate，验证三份完成记录与 GPU correctness，汇总固定 heads。
3. Formal array 0–69，最多并发 70 个单 GPU worker；每 worker 只运行一个方法的 20 条输入（四任务各 5 条）。
   10 个 shard 按输入长度平方平衡；每方法总计 200 条，全部 7 方法=1400 条。
   每 worker 1 张 H100-80GB、8 CPU、128GB RAM、8h 调度上限。
   可用 H100 的实际并发由集群决定，70 个任务并非保证同时获得 70 张卡。
4. 最后完成的正式 worker 释放模型显存后自动做 CPU audit，校验 1400 条配对结果、逐条官方分数、文件 hashes、cache/heads/精确 mask。

原 100 GPU-hours cap 和 runtime projection 导致的提前终止已移除，projection 仅记录。
Slurm walltime 是单作业运行条件，不是总体资源预算；失败先诊断并修复，再用独立记录续跑，避免盲目重复。
`approval.json` 记录此次用户授权及 config/input/source 三个 hashes；不会沿用上一版 hash 的审批。
`submit.py` 验证两个 GPU 脚本后提交 calibration→formal 依赖链；汇总与审计在最后完成的 worker 内执行，用原子文件锁确保仅执行一次。
输出在 `outputs/`，正式评分和相对 Dense 的差值写入 `outputs/independent-audit.json`。

账户不能使用普通 RM-shared 分区；aggregate/audit 是纯 CPU 函数，在已完成生成并释放显存的 GPU worker 内运行，无额外 GPU 分配或分区排队。
