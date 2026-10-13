# 代表头共享 mask：Vertical / Vertical+Slash 正式评测

本版根据 2026-10-01 的讨论准备代码与环境。**所有方法均应用于全部 32 层**。
尚未获得 GPU 运行同意。`setup.py` 和单元测试只使用 CPU，不加载 checkpoint、不提交作业。
上一版 `../representative_vertical_20261001` 的配置和验证记录完整保留。

## 评测矩阵

Dense 是主对照。固定代表头在同层所有 32 个 query heads 间共享布尔 mask；每个 head 用自己的 Q/K/V（以及自己的 GQA KV head）重新归一化。

| 选头方式 | AutoBlock 共享 | Token 共享 | Token + Vertical | Token + Vertical + Slash |
|---|---|---|---|---|
| 3 个独立样本校准 | coverage_autoblock | coverage_token | coverage_token_vertical | coverage_token_vertical_slash |
| 每层随机指定，固定种子 | random_autoblock | random_token | random_token_vertical | random_token_vertical_slash |

另有 Dense、FlexPrefill、MInference，共 11 组，200 × 11 = 2200 条正式生成。
AutoBlock 和 token-only 保留为基础对照，因此同时比较选头和新增 pattern 的影响。
Random 只使用一个 seed，报告不能视为跨随机种子的方差估计。

## 输入、模型与生成

- 模型：本地 Llama-3.1-8B-Instruct，32 层、32 query heads、8 KV heads、head_dim=128，BF16。
- KV Retrieval、En.MC、En.QA、Code.Debug 各 50 条，同一冻结 manifest 的 200 条完整输入；输入长 69713–130876 tokens，无截断。
- 复用 `shared_topp_mask_bench_20260930/inputs` 的逐字节输入；记录 token IDs、positions、mask、prompt、raw row 的 SHA256。
- 使用历史固定的官方 chat template 和 InfiniteBench scorer，greedy decode，seed=42。
- 最大生成长度依次为 50 / 40 / 40 / 5。每个方法保留全部 KV cache，decode 恢复 Transformers 的原始 Dense FA2 forward。
- 不训练、不优化参数、不压缩 KV、不改变位置索引。
- 新 Dense 在全部 32 层使用历史 FP32 full attention 实现；旧 Dense 为 16 层显式 FP32 attention + 16 层 FA2。
  因层适配范围改变，新 Dense 重跑；对旧输出的一致性只作诊断记录，正式对照使用本次 Dense。

## 代表头选择

校准为 3 个样本总计，不是每任务 3 条，且与全部 200 条评测输入的 source identity 和 IDs hash 均不重叠：
KV source 355、En.MC source 153、En.QA source 224，长度分别 124135、87950、69704。

每个样本运行 Dense prefill，在每层最后 128 个 query 上计算各 head 的 causal softmax。
对 provider head 逐 query 构造 Top-p=0.95 mask，度量它保留其他 target heads 的注意力质量：
`S[provider,target] = mean_query sum_key(mask_provider * probability_target)`。
取平均 outgoing coverage 最大的 head；3 个样本投票，平票比较投票样本的获胜分数之和，再平票选较小 ID。
输出每层一个固定代表头，所有正式样本及四种 mask 变体均复用。

Random 用 Python `random.Random(20261001)`，按层 0→31 顺序均匀采样 `randrange(32)` 一次，
所有评测样本及四种 mask 变体复用，不随 query 或样本轮转。

## 基础 mask 与 Algorithm 3 扩展

1. **基础 token mask** 使用仓库 `autoblock` 的 `RepresentativeTokenFirstBlockSelector`。
   Query tile=128；使用 1/3、2/3、末尾、平均 query 四个 probe，权重 `[0.2,0.3,0.4,0.1]`；
   每 tile 选 causal key 的固定 TopK=8192（不足时保留合法 key）。此处固定 TopK 是原 AutoBlock 的 mask 构造；
   校准阶段的 Top-p=0.95 用于选头，二者是不同步骤。
2. **AutoBlock 变体** 将这些 targets 投影到 32/64/128 自适应 key blocks，tolerance=0.02，
   minimum block coverage=0.125，final token budget=8192，保留 sink/diagonal blocks。
   Token 变体直接保留选出的 token，不作整块扩展，也不另加保护块。
3. **Vertical** 用当前层代表头的最后 128 个 query 和全长 K 计算 causal FP32 softmax，沿 key 列求和并归一化。
   排序后取累计质量首次达到 gamma=0.95 的最小前缀，无额外 cap，与基础 token mask 作并集。
4. **Slash** 使用同一代表头的 attention 子矩阵，按原始位置距离 `d=i-j >= 0` 聚合并归一化，
   独立取 gamma=0.95 的最小前缀；每个 query i 加入 `j=i-d` 的合法位置，与 token+vertical 作并集并去重。
   本版沿用 [FlexPrefill Algorithm 3](https://arxiv.org/html/2502.20766v1) 的方向求和、归一化和累计质量选索引，
   在 token 粒度加入 pattern，不直接使用官方实现的 128×128 block pooling 扩展。
5. mask 共享给同层全部 heads，各 head 使用自身 Q/K/V 计算并重归一化；不共享概率或 attention output。

Slash 的 CSR 存储为 query tile 的候选 key 并集；新增 Triton kernel 再检查每个 query 的相对距离。
候选列本身不会把 slash 扩大成 vertical。分别记录有效 mask pairs 和候选 QK pairs；执行时间包括选索引及诊断，
本轮用途是质量与 mask 分析，不能据此宣称加速。
Vertical/slash 的独立 gamma 不意味着合并 mask 在所有 query、所有 member heads 上保留 95% 质量。

仓库原 TopK 存在 `sorted=False` 后按 rank 截取的因果边界问题。准备阶段已改为 `sorted=True`，
防止前缀不足 8192 个合法 key 时丢弃有效 key；CPU regression 使用原 selector 验证。

## 两个官方 baseline

`baseline-reuse-audit.json` 记录了旧结果比对：FlexPrefill 和 MInference 各 32 条完整 KV 记录，
与本次 200 条固定输入各仅 1 条 IDs hash 重合；旧 KV 生成上限 128、本次为 50，缺少另外三任务。
这些旧记录不足以直接复用为本次四任务各 50 条的正式结果，因此两个 baseline 同输入重跑。

- FlexPrefill：官方 `flex_prefill_attention` kernel；gamma=.95、tau=.1、min_budget=1024、max_budget=None、block_size=128。
- MInference：官方 `minference_prefill_kernel`，Llama-3.1-8B-Instruct 的 `best_pattern_v2.json`（与 runtime MODEL2PATH 一致），ratio=1、starting_layer=0。
- 使用同一 Llama 4.44 适配器处理 RoPE、完整 KV、位置和 Dense decode，只替换 prefill 的官方 kernel；
  不修改官方 kernel。全部 32 层，输入与评分完全相同。结果注明这个公共适配器运行方式。

## 环境与检查

复用已验证环境：Python 3.12.13、torch 2.6.0+cu124、transformers 4.44.0、triton 3.0.0、flash-attn 2.7.4.post1、numpy 2.1.3。
`env.sh` 复用之前的 modules、overlay、compiler 和共享库设置，启用 offline 模式，未安装或替换包。
`source-lock.json` 冻结配置、脚本、公共实现、官方 kernel 源码和 MInference pattern/extension。

CPU 检查包含：原 selector 一致性、因果边界、方向求和与最小 gamma 前缀、去重和精确 pairs、
独立逐 query dense-mask oracle、GQA 私有概率/输出、固定 random heads、完整 KV/Dense decode 生命周期、审批先于执行。
GPU BF16 kernel 对独立 FP32 reference 的 parity、两 baseline 的 GQA 检查会在获批后的 calibration job 首先执行；
CPU 检查不能证明 Triton 在 GPU 上通过。

## 获批后的运行

审批必须明确针对本配置和资源预算；`approval.json` 必须包含 `approved=true`、`profile=full_suite`，
以及 config、input manifest、source-lock 的三个 SHA256。未获用户同意不会写审批或调用 `submit.py`。

- Calibration：1 张 H100-80GB，8 CPU、128GB RAM、最长 4h；先小 tensor GPU correctness，再 3 样本选头。
- Formal：依赖 calibration 成功，array 0–3 最多并发 4；各 1 张 H100-80GB、8 CPU、128GB RAM、最长 24h。
  每 worker 分到 50 条冻结输入，在每条输入上跑全部 11 个方法。总正式规模为每方法四任务各 50 条。
- GPU 预算上限 100 GPU-hours；没有额外 2 样本 pilot、自动重试或自动扩容。
- 每条保存 token IDs、prediction、stop reason、官方 score、首个生成位置全词表 logits、mask 与 coverage 诊断。
  运行投影超过批准时限时保留现有结果并报错，不把部分结果当成 50 条的最终表。

获批后 `submit.py` 才会检查 Slurm test-only 并提交 calibration→afterok formal。
正式作业完毕后隐藏 GPU 运行 `audit.py`，逐条重算官方分数、核对文件 hashes/完整 KV/选头来源/每任务 50 条，
输出 `independent-audit.json`，主结果为 Dense 分数与各方法相对 Dense 的差值。
