# J-lens 对稀疏 prefill 的分析：运行前方案

状态：2026-10-06 用户已批准正式实验；首次 GPU smoke 的数值兼容性检查失败，完成修正与验证后重新提交。

## 研究对象

固定本地 Llama-3.1-8B-Instruct 权重和 tokenizer；32 层、4096 维残差、32 个 query heads、8 个 KV heads，BF16。复用 `representative_patterns_fast_retry03_20261001` 的七种方法：Dense、coverage/random 的 Token+Vertical / Token+Vertical+Slash、FlexPrefill、MInference。每层固定代表头、mask 参数、官方 baseline 配置和完整输入均不改动，不重新选头。

这轮问的是：稀疏 prefill 相对 Dense 的残差与 KV 数值变化，是否伴随可表达概念的读出变化；变化在哪些层与 token 位置出现。观察性比较不能证明因果，也不将 Llama 的读出自动等同于论文在 Claude 上识别的工作空间。

## 与原始 J-lens 是否相同

**拟合器和读出公式相同。** 直接调用本地原仓库 `jlens.fit`，不改 `fitting.py`、`hooks.py` 或 `lens.py`，原仓库 commit 为 `581d398613e5602a5af361e1c34d3a92ea82ba8e`。

对源层 block 输出残差 h_l，目标是 block 31 输出、最终 RMSNorm 之前的残差。对每个输出维度，在所有有效目标位置同时注入 one-hot cotangent；反向传播后，对有效源位置取平均，再对提示词取平均。因此是 **目标位置求和、源位置平均、提示词平均**，不是对所有 (源位置, 目标位置) 对等权取平均。序列长度 128，默认跳过前 16 位置和最后位置；有效位置 16–126，共 111 个。因果模型使较早目标位置相对较晚源位置的梯度为零。

每个源层得到 J_l，shape=[4096,4096]。0–30 层拟合共 31 个矩阵；第 31 层展示为 J=I。

读出使用模型自己的 `LM_head(RMSNorm(J_l h_l))`；分布比较再使用 softmax。无拟合损失、无优化器、无模型参数更新、无 affine bias、无 sparse-mask 梯度或 stop-gradient-QK 变体。`dim_batch=32` 仅把输出维度的反向传播分批并行，不改变估计器。

32 个 GPU 分片对同一固定列表的不同提示词拟合；CPU 按成功提示词数加权合并，数学上等价于原 `JacobianLens.merge` 的平均规则，FP32 累加顺序与串行拟合可能产生少量舍入差异。分片与合并都保存 FP32，减少额外序列化误差。

**不同点与扩展：**

- 模型换为我们的 Llama；拟合语料为公开 WikiText-103 train，不是论文的内部预训练语料，也不是 Qwen 的预拟合 lens。
- 为适配已有 Transformers 4.44，使用 `LensModel` protocol 兼容层，不调用要求新 Transformers 的 `from_hf`。模型 forward、block 输出位置、最终归一化和输出层保持原语义。
- 用在 Dense 上拟合的同一个固定 lens 读取全部稀疏方法，不为每种方法重新拟合。因此度量的是稀疏激活在 Dense 读出坐标下的变化，不是稀疏模型自身平均因果 Jacobian 的估计；其读出不能直接作为稀疏模型未来输出效应的证明。
- 原始短文本拟合流程用于 69k–131k 的完整输入；长上下文上的可解释性需要单独检查，不预设短文本平均 Jacobian 足够准确。
- 新增逐方法 JS/KL、排名、残差误差和抽样 K/V 误差；这些不是原论文的完整实验复现，也不是 gradient-pursuit 的 J-space 稀疏分解。
- 本轮无 steering、swap、clamping、ablation 或 KV 修复实验。

## 输入与输出

### 阶段 0：CPU 准备（无需真实模型推理）

输入：历史 source lock、输入 manifest、token IDs/positions/mask、原始 prompt、历史 logits/任务分数、本地 tokenizer、WikiText parquet，以及原仓库六组 lens 评测提示词。

处理：验证来源与 SHA256；固定 1000 条不同的训练集文本（至少 600 字符且至少 128 个 token）；tokenizer 加其原配置中的 special tokens，截取前 128 token。这个截取只作用于 lens 拟合文本，**不截取正式长上下文输入**。排除与正式评测 prompt 的完全相同文本。它并不构成对训练语料与公开 benchmark 语义重合的全面审计。

输出：`fit-prompts.json`、`input-manifest.json`、`source-lock.json`、CPU 测试日志和验证记录。模型与 tokenizer 文件固定 SHA256，并在各阶段检查文件大小/修改时间；正式输入和代码在运行前验证 SHA256。

WikiText 来源：`Salesforce/wikitext` 的 `wikitext-103-raw-v1` train 第一个 parquet 分片；revision `b08601e04326c79dfdd32d625aee71d232d685c3`，文件 SHA256 `74da360f23826045b3e6ac6375411fdb15f003030aa74f2596ed08b857cb9212`。

### 阶段 1：GPU 兼容性与正确性检查

输入：本地模型、第一条拟合文本、随机小型 Q/K/V。

处理：未安装历史 adapter 的 Dense 模型做 batch=2 反向传播可用性检查；短文本上比较原始 FA2 Dense 与历史显式 FP32 attention 的 logits，要求逐 logit atol=.2/rtol=.02、JS≤.001 nats、首选 token 相同，并保存完整诊断；稀疏 Triton 内核对独立逐 query mask/GQA oracle；baseline 做 constant-V 检查。

输出：`outputs/gpu-smoke.json`。不更新权重；这一阶段也必须先获得用户批准。

### 阶段 2：拟合并合并固定 Dense lens

输入：1000 条独立 WikiText 文本，每条 128 token；标准 Dense Llama，不安装禁止梯度的稀疏 adapter，也不安装只支持 inference 的 MLP 改写。

处理：32 个 H100 workers，每个 31 或 32 条文本；`jlens.fit`，source layers=0..30，target=31，dim_batch=32，skip_first=16。每 8 条保存原生可续跑检查点。CPU 流式按提示词数加权合并。

输出：每个分片的 `checkpoint.pt`、`lens.pt`、完成记录；合并的 `outputs/lens.pt`（31×4096×4096 FP32，约 1.94 GiB）及来源记录。

### 阶段 3：lens 读出质量检查

输入：拟合完成的 lens、Llama，以及原仓库六组评测全部 551 条：多跳、多语言、诗歌、运算顺序、联想、拼写纠错。

处理：同一输入位置分别读取 J-lens 和普通 Logit Lens；对每个目标概念统计跨 32 层的最小排名，得到 pass@1/5/10/25（0–30 层用拟合矩阵，31 层为两种 lens 共用的 J=I 真实输出）。诗歌在最后一个换行 token 读取，其余在 prompt 最后位置读取。保留单 token 的空格、大小写形式；运算顺序使用代码中固定的数字/运算同义词集合；没有单 token 名称的概念记录为不可读出并计为失败。

输出：`outputs/lens-validation.json`，包含逐样本概念/token 形式、最小排名、六类汇总和不可读出计数。该 tokenizer 和同义词处理不是对论文私有模型结果的逐字复现；也不能单凭短文本结果证明长上下文适用性。

### 阶段 4：完整长上下文 prefill 重放

输入：原四任务 KV Retrieval、En.MC、En.QA、Code.Debug 各 50 条，共 200 条；复用完整原始 token IDs、position IDs、attention mask，无重新渲染、无截断、无 chunked prefill。七种方法共 1400 次 prefill。

采集位置：前 16、末尾 128、全序列均匀 32 个位置，去重；KV Retrieval 额外加入答案值的原始文本 span 和前后各 2 个 token。答案标签只用于事后观测位置标注，**不传入模型、不参与构造 mask、不参与选头**。其余任务不假定均匀抽样一定覆盖相关证据。原 prompt 中没有找到答案 span 时明确记录。

处理：10 个 Dense workers 先保存参考，随后 60 个 sparse workers。每个 worker 20 条样本，每任务 5 条。通过 forward hook 读取所有 32 个 block 的输出残差，原始 attention dispatch/头选择/概率归一化不改动。完整 KV cache 保留，并采集同位置的 8-head K/V。

使用同一固定 lens，对完整词表计算 JS divergence、KL(Dense||Sparse)、Dense Top-10 在 Sparse 中的排名、Top-10 overlap；排名复用原仓库 `jlens.vis._ranks_of` 的完整词表 argsort，再转换成 1-based。记录原始残差和 lens 映射后残差的相对 L2/cosine，以及 K/V 的相对 L2/cosine。每次只处理一个层的 16 个位置，避免保存全部长序列词表 logits；不启用 `mask_display`。

输出：每个方法/样本的 JSON 指标和 `.pt` 抽样残差、K/V、最后 prompt logits；词表 Top-10 token IDs/文字；prefix/tail/uniform/检索证据位置标签；新旧 prefill logits 是否 bitwise 相同、最大差值及第一 token 是否相同。

本轮不重新生成回答、不生成新的任务分数。历史官方任务分数只作为关联标签；新旧 replay 不一致时独立标记，并提供 Dense 和 Sparse 两方 first-token 都与各自历史一致子集的描述性统计。该一致条件不证明后续生成完全一致。第 31 层 J=I 的单位置读出重建与真实 logits 对照，记录最大误差，正确性检查容许 BF16 数值误差（atol=.15/rtol=.02）。采集耗时包含 observer 开销，不与历史方法性能计时混用。

### 阶段 5：配对汇总与审计

输入：1400 条新观测及历史任务标签。

输出：`outputs/summary.json`（按任务/方法/层/位置组的 JS，VS−V 配对差异，和历史分数下降的关联）、`outputs/independent-audit.json`（输入、lens、config、结果文件及完整配对验证）。Bootstrap 按样本重采样 2000 次，固定 seed=42，结果为描述性统计，无多重比较显著性结论。

JS/排名变化是读出指标，不等于证明某概念因果损失；lens 映射后误差不等于非负稀疏 J-space 分量。Vertical 与 Vertical+Slash 不匹配预算，因此不单独归因于 pattern 形状。

## 环境与资源

保持既有 Python 3.12.13 / PyTorch 2.6.0 / Transformers 4.44.0 / Triton 3.0.0 / FlashAttention 2.7.4.post1 / NumPy 2.1.3。新目录 `deps/` 添加 PyArrow 19.0.1 和 pytest 8.3.5，不覆盖原 env，不安装原仓库声明的 Transformers>=5.5。

每 GPU worker：H100 80GB×1、8 CPU、128GB host memory、8 小时；GPU-shared/cis260309p/push。流水线依赖：1 个 smoke job → 32 个拟合 worker → 1 个合并+质量检查 job → 10 个 Dense worker → 60 个 sparse worker。总计 104 个作业/array elements，峰值 60 GPU，同时不运行全部阶段。8 小时为单作业墙钟上限，不是预计耗时；拟合耗时尚未实际测量。

所有记录原子写入；每 300 秒保存进度清单，并对最近已完成的拟合 checkpoint 保存不可变 hardlink、保留最近两份；最后退出时再次保存。长提示词处理期间 timer 也运行。当前 prompt 尚未完成的梯度图不保存，只能从最后完整 checkpoint 续跑。

## 启动条件

2026-10-06：用户明确批准 WikiText 拟合、InfiniteBench 测试，不限制 GPU 总使用量。第一次 GPU smoke 在 128256 个 logits 中有 1 个差值 .1640625 超过最初 atol=.15；该次失败及来源绑定归档在 `attempt-01/`，下游作业均未执行。随后将 Dense 实现兼容性检查的绝对容差改为 .2，并增加 JS 与首选 token 条件及 CPU 验证。原始 J-lens 拟合器、实际稀疏方法、正式输入、残差/KV指标及最后层重建容差均不改动。原生 FA2 与历史显式 FP32 scores 的差异另行记录，不作为稀疏误差隐藏或归零。

`approval.json` 必须明确批准并匹配 config、source-lock 和 input-manifest 的 SHA256，且 GPU/model 阶段需在 Slurm allocation 中运行。用户授权按 WikiText 拟合、InfiniteBench 测试的正式流程持续有效；每次必要修复均重新核验和绑定来源，失败记录保留，不重复请求相同范围的授权。

2026-10-06 验证修复：32 个拟合分片和 1,000 条 WikiText 序列已完成并合并。原 tokenizer 对换行返回零宽 offset，导致诗歌验证找不到读出位置。改为在原始 token 序列中选择最后一个解码包含换行的 token；若最后换行被截断则明确拒绝。全部 551 条输入（其中 98 条诗歌）的读出位置已用 CPU tokenizer 核对。拟合矩阵、拟合器、输入、评分规则和稀疏算法保持不变。失败记录归档在 `attempt-03-pre-validation-repair/`；`validation-retry.sbatch` 只重跑附加验证，不再次拟合或合并。
