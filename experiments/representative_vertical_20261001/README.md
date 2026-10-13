# 单代表头共享 mask 与 vertical token 扩展

当前状态：代码与 CPU 环境准备；**尚未批准或运行 GPU 实验**。
工作目录位于 `/ocean/projects/cis260309p/jzhang114/sparseAttention/experiments/representative_vertical_20261001`。

## 已同步的讨论结果

`handoff2.md` 与实际落盘的两个 `independent-audit.json` 一致：

| InfiniteBench 子任务，每任务 50 条 | Dense | 逐行共享注意力 A | 逐行共享 Top-p=0.95 mask |
|---|---:|---:|---:|
| KV Retrieval 准确率 | 64% | 0% | 0% |
| En.MC 准确率 | 78% | 0% | 2% |
| En.QA F1 | 0.3054 | 0.00667 | 0.00617 |
| Code.Debug 准确率 | 14% | 0% | 0% |

两次运行分别通过 400 条结果审计；Dense 的 200 条输出一致。
这些旧实验按 query 位置分配 head，新实验改为**每层一个固定代表头**。
Top-p 仅说明 mask 提供头的覆盖，不能推断同层每个成员头都达到该覆盖。
当前仅诊断质量，尚不能宣称共享有效或有加速收益。

## 仓库代码核对

用户指定仓库：<https://github.com/erly-zhang/sparseAttention/tree/autoblock>。
本地和远端 `autoblock` 分支均为 `8b21a6a18056eadc00b6248bb319a4af3c6ccc4d`。

- `../benchmark_shareprefill_ae3.py`：历史方法到参数配置的映射；原 fixed Top-k 默认 8192。
- `../token_compacted_sparse.py` 的 `RepresentativeTokenFirstBlockSelector`：
  `_score_representative_keys`、target mask、32/64/128 自适应块投影、CSR token 索引。
- 原 `project_target_to_blocks=False` 路径用于 token-only；不是 query 按 head 轮换。
- 原 `triton_token_compacted_prefill_attention`：共享 token 索引，各头独立 QK/softmax/GQA V。
- `../run_shared_layer_mask_experiment.py`：方向覆盖率、每样本候选代表头及跨样本投票规则。
- 已有 `member_vertical_slash` 为各成员头补 vertical+slash **blocks**，与本次共享 vertical token 列不同。

必要修复：fixed Top-k 分支原来 `sorted=False`，随后按 `min(query_end,K)` 截取。
CPU 复现 K=256、N=513 时第一 query tile 应保留 128 个合法 key，却仅保留 127 个并漏掉 key 0。
改为 `sorted=True`，保证合法分数位于未来位置的 `-inf` 前面。此修复影响使用该分支的旧 AutoBlock 输出，
新结果属于修复后的实现，不能宣称与历史 AutoBlock 输出逐 token 等同。
旧逐行 Top-p 实验使用另一份实现，不因此改动。

## 待批准配置

Llama-3.1-8B-Instruct，BF16；32 个 query heads、8 个 KV heads，head_dim=128。
仅修改偶数层 0、2、…、30 的 prefill；其余层及 decode 保持原始 attention，完整保存 KV。
采用相同的官方 prompt/scorer、chat template、原始位置，无截断、无训练、greedy seed=42。
生成上限依次为 KV 50、MC 40、QA 40、Code 5。
Dense 控制组在上述 16 层沿用历史 FP32 blocked explicit attention，其余层为 FA2。

### 代表头校准

新增 3 个与全部 200 条评测数据分离的样本：KV、MC、QA 各一个，按 seed=20261001 随机顺序选择完整可容纳样本。
在每个修改层，取最后 128 个 query（短输入取全部），每个候选头对完整 key 序列计算逐行 causal softmax。
用每行 Top-p=0.95 的支持集构建方向覆盖矩阵：

`S[provider,target] = mean_query sum_key A_target(query,key) * M_provider(query,key)`。

候选头为 `argmax_provider mean_target S`。三个样本投票，先比较票数，再比较得票样本的分数之和，
最终平票取较小 head ID；固定该层代表头用于所有评测方法和样本。
模型校准本身也是 GPU 实验，**尚未运行**；目前只冻结了校准输入。

### 三个 sparse 分支及 Dense 对照

公共 token 目标沿用原 AutoBlock 四 probe：1/3、2/3、最后一个 query 和 tile mean，
权重为 0.2/0.3/0.4/0.1；先加权 **logits**，再选 fixed Top-k=8192，按 tile 的合法 key 数截断。
执行 query tile 为 128，原始 token 位置用于最终 causal mask。

| 方法 | 原选择器设置 | 共享对象 |
|---|---|---|
| `representative_autoblock` | project=True；logical K sizes=32/64/128；block 密度阈值 0.125；coverage 容差 0.02；最终 whole-block token 预算 8192；原 sink/diagonal block 保护开启 | 每层代表头的 AutoBlock CSR 索引 |
| `representative_token` | project=False；同样 Top-k 目标；不扩展成整块；无额外 sink/diagonal 强制保护 | 每层代表头的 token CSR 索引 |
| `representative_token_vertical` | 与 token-only 完全相同的 base CSR，再合并 vertical key token | 同层所有头共用扩展后的 CSR 索引 |

AutoBlock 与 token-only 的保护机制也随原变体不同，二者不是仅改变 block 粒度的严格消融。
**token-only 与 token+vertical 具有相同 base mask**，这组对照可直接观察新增列的影响。
共享布尔支持/索引，不共享注意力概率 A 或输出 O；每个成员头仍计算自己的 QK、softmax、V 和 o_proj。

### Vertical 扩展

依据 [FlexPrefill Algorithm 3](https://arxiv.org/html/2502.20766v1#S3.SS3)：

1. 取本层代表头最后 128 个 query，对完整 K 计算 causal softmax，得到 `Ahat[128,N]`。
2. 每列平均，`a_v[j]=mean_query Ahat[:,j]`，再按总和归一化。
3. 降序排序，累积到 **gamma=0.95** 的最小前缀；等分按原 key 位置升序。
4. 将选中 key token 与每个 query tile 的 base CSR 去重合并。key 必须小于该 tile 的末端，
   执行 kernel 再逐 query 应用 `key_position <= query_position`。

当前 **无 vertical token 数量上限**，不增加 slash、不扩成完整 key blocks，
也不复制 FlexPrefill 的 block 预算与 first/last-block 规则。
gamma 是代表头最后 128 个 query 的聚合列分数覆盖，不能解释为全序列或全部头的逐行 95% 覆盖。
代码支持显式设定 `vertical_max_tokens`；若未来设 cap，实际质量覆盖可能低于 gamma，会记录 cap 和实际 mass。
新增列可使最终 token 预算高于 base 的 8192；报告实际增量和 pair density。

## 输入与资源安排

只读复用旧实验完整的 200 条输入及每个文件的哈希，不复制大型数据集、不重新抽评测集。
pilot 从冻结的每任务 50 条里按 seed=42 选 2 条：共 8 条输入 × 4 方法 = **32 次生成**。
另有 3 次只做 prefill 的代表头校准。

**建议首先批准 pilot：1 张 H100 80GB，8 CPU，128GB 内存，最长 4 小时，总上限 4 GPUh。**
完整实验预置为 4 张独立 H100、每张 8CPU/128GB/24h，总上限 96 GPUh；
200 输入 × 4 方法 = 800 次生成，复用 pilot 固定代表头。完整实验需要另外批准，pilot 不会自动扩展。

GPU 运行开始前，在批准的 allocation 内比较现有 Triton kernel 与独立 GQA 参考；
每条 Dense 都必须逐 token、prediction、stop reason、score 复现历史控制组。
每个输入完成后根据最慢观测 prefill/decode 估算剩余时间，乘 1.5 并留 600 秒；超预算停止并保留结果。
没有自动重试、重新申请资源、训练或自动提交完整实验。

## 环境、验证与运行入口

复用现有 Python 3.12.13、PyTorch 2.6.0+cu124、Transformers 4.44.0、Triton 3.0.0、
FlashAttention 2.7.4.post1、NumPy 2.1.3；不升级或覆盖旧环境。
`source-lock.json` 锁定本次代码、借用源码和环境文件；`cpu-validation.json` 记录验证。

CPU 准备（不加载真实模型权重、不用 GPU；adapter 单元测试使用微型随机初始化模型）：

```bash
source env.sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "$BASE/env/bin/python" prepare.py
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "$BASE/env/bin/python" setup.py
```

实验入口 `submit.py --profile pilot` 必须先有与 config/input/source 哈希匹配的 `approval.json`。
该文件只在用户明确批准后建立；当前不存在，直接运行或提交均被阻止。
本次没有调用 sbatch、GPU parity、模型校准或生成。

批准并运行后，`audit.py --profile pilot` 在 CPU 上独立重算官方得分、最终 prompt 词表 KL、
长度上限/重复标记，核验输入、源码、logits 和完成文件的哈希，以及代表头与共享契约。
还必须另行核对 Slurm `COMPLETED / 0:0`。
输出包括 causal attention pair keep ratio、新增 vertical pair 数，以及最后 128 个 query 的每头 retained mass。
token slot keep ratio 与 attention pair keep ratio 分开解释；不以全序列 unique token 比例代替 sparsity。

## 文件入口

- `selection.py`：调用原 AutoBlock 选择器、方向覆盖校准、vertical 列选择、CSR 并集。
- `attention.py` / `adapter.py`：prefill 接入现有 kernel、独立每头 QK/GQA、完整 KV/dense decode。
- `prepare.py` / `protocol.py`：冻结输入、隔离校准、配置与哈希约束。
- `run.py` / `submit.py`：批准后的 GPU parity、校准、pilot/formal、一次性提交与预算控制。
- `test_selection.py` / `setup.py`：CPU 回归验证与源文件锁定。
- `audit.py`：CPU 结果审计；`run-pilot.sbatch` / `run-formal.sbatch`：待批准的调度配置。
