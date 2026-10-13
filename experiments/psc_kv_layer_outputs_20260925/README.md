> Completed KL comparison: [run_kl_resume/RESULT.md](run_kl_resume/RESULT.md). Job47087068 completed successfully; all320records verified. The older incomplete-status notes below are historical.

> Latest repair outcome (2026-09-25): see [REPAIR_RESULT.md](REPAIR_RESULT.md). Five-method GPU correctness passed; real calibration maps are partially aggregated; full comparison remains blocked by the measured remaining-budget plan. Earlier status below is historical.

# PSC 新受控五方法层输出实验

> 已运行一次：job47069472，FAILED/1:0，76秒。MInference correctness编译失败，后续AE和真实样本比较未启动。详见 [RESULT.md](RESULT.md)。

本目录独立于所有历史实验。当前分支 autoblock，源码基准 `8b21a6a18056eadc00b6248bb319a4af3c6ccc4d`。不复用 Wulver 数据参照、groups 或硬件计时。

## 固定协议

数据来自用户指定 InfiniteBench revision `90f0394333616266d9fe85824ceaf505093cbaa5`。完整文件仅用于下载/格式与500条记录核验；仅索引0、1、2作校准，索引3作测试。输入身份在模型输出前固定。四条 chat 输入长度分别124048、123931、124136、123959，均未触发截断；maxcontext131072，按原 KV runner 保留128生成预算，输入上限130944，超长统一首尾中间截断。

`inputs/manifest.json` 保存数据SHA、原始行/规范JSON SHA、token长度和ID SHA。IDs存为无头部little-endian signed int64序列。原AE另使用校准原文本前1936 token（非chat、默认special tokens），独立保存为 `*.ae_ids.i64le`，与原提取脚本核对一致。测试索引3从不进入AE、分组或checkpoint选择。

模型 Llama-3.1-8B-Instruct、BF16、batch1、seed42。32层、Q32/KV8、head_dim128、hidden4096，GQA h//4，无Q/K norm、无投影bias，原Llama3 RoPE参数不变。checkpoint/tokenizer pinned revision `0e9e39f249a16976918f6564b8830bc894c89659`；完整文件重新hash见 model_provenance.json。

五方法为 dense FA2、C position-mod32组内causal/scatter/显式repeat32/完整o_proj、Shared6144+member2q4096、官方FlexPrefill、官方MInference。不是Q-only，也不包含NN。精确参数见 experiment_config.json。代表头仍Shared10240，其余29头Shared6144+4096；原sink128、当前query tile、原GQA与causal规则保留。AutoBlock调用原Selector与原compacted Triton内核，通过新PSC接口适配器接入。

主实验为各方法正常端到端32层前向，绝不逐层注入dense hidden states。每层记录 o_proj后/residual前及完整block输出；dense是五方法之一。全部prompt KV完整保留。仅prefill，不生成答案、不计算任务准确率。保存dense两处逐层BF16张量约60.53GiB；其他方法逐层比较后释放临时张量。

## 原AE流程与预算

复用 run_shareprefill_autoencoder_clustering.py 原实现：1936平方attention maps，三校准样本均值；卷积AE latent64，Adam lr.001、batch1、seed42、最多1000epoch、patience30、min_delta1e-7；1024个(layer,head)均值map按index%10划分921训练/103验证；归一化latent后Ward/cut_tree K=3、Euclidean medoid。该验证是head-map划分，见过全部三条校准数据，`grouping_independent_of_validation=False`，不是独立三折。分组标注PSC regenerated。

原始maps约21.45GiB，mean maps14.30GiB，完整test KV约15.13GiB。AE最早正常patience停止为31epoch；最坏1000epoch。无法仅凭GPU规格保证1小时完成。作业内先测原AE在真实校准mean map上的训练/验证步耗时（临时权重丢弃，不作为checkpoint），估计每epoch=921*t_train+103*t_val；最早31epoch加25%余量仍超剩余训练预算则停止并报告blocked_budget，绝不偷偷减epoch或改聚类。临时成本测量也计入总GPU预算。

单次提交 GPU-shared / cis260309p / 1 H100-80 / 8 CPU /128GiB host RAM /1小时。内部3500秒deadline、外部3550秒timeout、Slurm3600秒；不自动重提。预留900秒给聚类与五方法重放是计划估计，不是运行保证。GPU correctness失败、AE预算不足或后续超时均记录未完成，不冒充有效结果。

## 指标与计时

每层whole-layer Frobenius相对L2、所有有效token/features上的RMSE、逐token cosine均值、T=1输出特征softmax后的KL(dense||method)、有效causal pair数。KL单位nats，不是attention权重或next-token KL。batch1无padding。零范数：两个向量均零cosine=1，仅一方零=0；dense全零且误差非零时relativeL2=null。

每方法先33-token warmup，再一次无输出诊断hook的完整前向计时，最后另一次完整前向做张量保存/指标/pair诊断。诊断耗时单列、不混入计时。完整长度首次编译可能计入计时，因此不是稳态latency结论；算法实现自带的inline pair accounting仍在前向内，不能称无任何计数开销。5方法全部使用相同输入/精度/权重。额外前向也计入1小时预算。

## 环境与检查

独立 env：PSC Python3.12.13、只读共享Torch2.6.0+cu124、Transformers4.44.0/tokenizers0.19.1、FA2 2.7.4.post1官方匹配ABI wheel、Triton3.2.0、MInference0.1.6.0、GCC13.3.1、CUDA12.4。FlexPrefill复用原source；其原requirements标Triton3.0，而当前3.2是否数值兼容必须由GPU gates验证。

MInference为兼容旧Transformers，对其两处GLM-only import做可选导入保护；GLM调用明确失败，Llama选择器、pattern、kernel未改。补丁原文件、前后SHA在deps/。其模型key对应官方v2 pattern，已保存副本和SHA。CPU import/五方法installer构造/固定IDs重验已完成；这些不能代替GPU correctness。

run.sbatch包含全部GPU检查与实验。correctness.py检查小张量causal、repeat、GQA、原member2q masked-reference、最小Llama接口/cache/future invariance。检查使用结构固定的合成fixture group，绝不作为实际AE groups。实际groups只有原AE训练并完成原checkpoint选择后才输出并验证。

结果位置：run/state.json、各阶段log、ae/、run/dense/tensors/、run/<method>/layers.jsonl/timing.json；全部完成后合并320行（5×32×2），summarize.py生成CSV和图。Slurm终态另存sacct记录。缺少产物意味着未完成，不以历史成功日志替代。
