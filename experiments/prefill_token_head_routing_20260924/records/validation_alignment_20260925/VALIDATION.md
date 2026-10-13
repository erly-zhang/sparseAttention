# 2026-09-25 对齐版本验证

用户在本轮明确授权“开始进行验证”。所有计算在已有 allocation 47018302、v009、空闲 V100-32GB 上进行；未申请新资源、未提交正式任务。源码快照见 source_snapshot.tar.gz，哈希见 provenance.json。未修改原 autoblock 跟踪文件。

已通过：

- `test_routing test_review_changes test_alignment` 共 14 项测试；Llama 小模型使用 CUDA/SDPA。包含独立 scalar reference、MHA/GQA、RoPE、padding/变长、N<H/非整除/单 token、scatter/repeat/bias/求和投影等价、未来不变性、实际 hidden states 形成完整 cache、多步原 dense decode、HF generate，以及结果损坏拒绝。
- 官方 InfiniteBench synthetic adapter：A/B/C loader、generate、记录与 grader 接口均通过；使用合成 passkey 和随机小 Llama，不是 benchmark 准确率。
- 新增 `validate_aligned_recorder.py`：真实 canonical GenerationMetricsRecorder 与 deferred GenerationBridge，在 CUDA/SDPA 小 Llama 上每条件 warmup 后连续两次三-token生成。A/B/C 均通过 warmup 排除、两次明细落盘、pair 累计和 decode event 断言。mock 测试另检查明细写入在 canonical timer 之后。
- 14 个 Python 文件 AST 语法检查通过。

限制：真实 FA2 尚未验证。FA2 的测试使用 mock 实现核对等长组内 causal 调用契约，不证明实际 FA2 内核可运行或数值一致。现有 V100 不支持目标 native bf16/FA2；不会用 fp16/SDPA 成功替代该项。原 filtered 数据/输入参照也尚未核验，正式入口与全数据评测未运行。

实际 8B smoke 结果见同目录 `llama8b_smoke/results.json`（若不存在则未完成），仅两条原合成输入、每条件最多 15 token、fp16/SDPA。smoke 手动生成验证 routing/cache 接口，不能替代完整官方真实数据入口或 bf16/FA2 验证。计时不用于与原 AutoBlock 比较，不用于选正式配置。

## 真实模型 smoke 已完成

6/6 记录完成，32 层完整 prompt cache 均为 [1,8,93,128]。A 的两条输入均生成 9 token 后 EOS，回答合成 passkey；B/C 均生成 15 token 达到 smoke 上限，分别出现重复反斜杠/重复 U+200D。相对 A，首步 logits 最大绝对差 B 约 23.59/23.31，C 约 21.73/21.63。结果说明实现可运行，但路由输出与 dense 明显不同，不能据此声称表现保留或已证明有效。具体时间与显存见 results.json，不作正式性能比较。
