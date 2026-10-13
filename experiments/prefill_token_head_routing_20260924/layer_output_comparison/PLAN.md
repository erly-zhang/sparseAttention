# 逐层 O 对比：已确认的定义与待补齐条件

用户已确认：模型 Llama-3.1-8B-Instruct，同一 InfiniteBench KV retrieval 输入；所有方法每一层均接收 dense 轨迹的该层 hidden states。比较完整 o_proj 之后、residual 之前的 O。仅 prefill，无生成或评分，不运行 497 题完整评测。

对于每层每个有效 token，将 4096 维输出作 softmax(T=1)，计算 KL(softmax(O_dense)||softmax(O_method))，自然对数、单位 nats，按有效 token 平均。额外报告 whole-layer Frobenius 相对 L2 与逐 token 平均 cosine。padding 排除；不作额外标准化。此 KL 是人为定义的输出特征概率分布差异，不是 attention 权重 KL 或下一 token 分布 KL，对温度/输出尺度敏感。dense 自比较应为零。

方法：
- Dense 原始 attention。
- 本方法 C：position % 32、组内 causal、scatter 后重复拼接 32 次并完整 o_proj。
- AutoBlock 精确版本 `probmean_shared6144_member2q4096_allmembers`；原 AE K=3，每层 3 个代表头 Shared10240、29 个 member 共享核心6144+两 query 估计的补充4096；原 sink/local 规则及 GQA。
- FlexPrefill 原 Llama 设置：block128，gamma=.95，tau=.1，min_budget1024，max_budget=None。
- MInference 使用 `meta-llama/Llama-3.1-8B-Instruct` 对应 pattern，不能复用 Qwen pattern。

执行方案：先以原清单中固定首条真实输入作 bounded pilot，不根据结果挑样本。沿 dense forward 的 32 层逐层捕获进入 attention 的 hidden states、position embeddings、mask；分别运行各方法的该层 attention。每方法独立空 cache、同一位置/权重/精度；所有输出只用于指标，只有 dense 输出进入后续 residual/MLP/下一层。必须确认 patched 方法 hook 捕获的是 o_proj 后输出，不重复应用 o_proj。禁止让候选方法输出进入 dense 轨迹。逐层计算并释放临时输出，保存小型指标而非全部 hidden tensors。metrics.py 是指标模块；方法 replay 适配器尚未完成，不是可直接启动的实验入口。

数据必须包含相同 tokenizer/chat/truncation 后的 token IDs/hash；长度应进入 >10240 可见 key 的区间，否则目标 AutoBlock 在预算内退化为全 key，无法观察其稀疏近似误差。不能用之前 93-token 合成 smoke 冒充本任务。

结果：每个 sample/layer/method 一行 JSONL/CSV，包含 KL、相对 L2、cosine、长度、输入hash、源码/权重/group/pattern/config hash、backend/dtype；输出32层曲线。多样本时同时区分 sample均值与token加权均值。数值仅说明固定 dense 输入下的局部误差，不能代替端到端任务表现。

当前阻塞：
1. 全工作区文件名检索未发现原 kv_retrieval.jsonl 与 shareprefill_ae_k3_head_groups.json。原 schedule.py 仍指向 /home/ubuntu/work 下旧目录。需要原 filtered 数据、输入参照和精确每层 group 配置，不能临时重新聚类替代。
2. 仅已有 V100 allocation 47018302；目标 bf16/FA2 及原 sparse kernels 兼容 GPU/环境尚未具备。不会自动申请资源或降级后宣称设置一致。
3. MInference/FlexPrefill 完整 runtime 与当前 Transformers 5.7 接口兼容性尚未验证。原 online.install 会自动运行 >10k 随机 warmup，不能未经审查直接调用其有副作用入口。

未运行本比较、未产生 KL 结果。仅新建指标模块和本方案，不改变现有实验。补齐资产/兼容资源后先完成 replay 和小张量验证，再明确报告单输入GPU/时间上限执行配置；扩展范围或资源申请须同意。

## 提交授权更新

用户已授权提交计算任务。候选 pilot 资源为 GPU-shared、1 H100-80GB、1 小时上限（最多1 GPU-hour），固定首条原真实输入、32层、五方法、仅prefill。仍缺上述输入/group资产与完整适配器/运行环境，所以尚未提交；不提交必然因缺文件失败的任务。补齐前置条件后可按此有限范围提交，无需重复获取同一许可。不得擅自扩展为497题正式评测。详见 submission_authorization.json。
