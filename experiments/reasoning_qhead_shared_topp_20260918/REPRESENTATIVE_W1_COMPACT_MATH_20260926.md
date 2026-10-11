# W1 代表头 TopK 压缩：MATH500

用户选定 W1。新方法名 `qhead_shared_representative_compact_recent_gpu`；独立根目录 `results/representative_w1_compact_math_20260926`。原随机压缩版和正在运行的多数据集实验保持不变。

## 算法

复用 `RandomCompactRecentLayer` 的存储、attention、指标、物理 gather 与压缩调用顺序，仅覆写 query 保存和位置选择。完整 prompt 长度 P 不占 reasoning1024；recent64 计入 reasoning1024。prefill dense；每次 decode 先对全部存活 KV（含新 KV）计算 attention，再在 reasoning 达到1088时压回1024，之后每64步重复。

每次压缩、每层抽一个代表 Q head，使用独立 `rep_seed=0 / sample_idx / layer` 随机流；只在压缩事件推进该流，同一事件的两个活跃 rollout 使用相同代表头。该流不影响生成随机状态。正确 GQA 映射 `h_kv = h_q // (H_q/H_kv)`。

当前 post-RoPE query 对本层全部存活 key 打分，沿用历史 W1 的 QK 缩放与 FP32 softmax。排除 prompt 和 recent64 后，取分数最大的960个旧 reasoning 位置；概率并列时选择较早的原始位置。最终按时间顺序拼接 prompt、TopK960、recent64，同层全部 KV/Q heads 共享这些位置，各自 KV 数值保持独立。

代表头 seed 规则沿用历史私有层/输入流，但抽取频率改为每次压缩，故不承诺与历史每步 W1 使用相同代表头序列。Top-p 选择已替换为固定 TopK；生成采样 Top-p0.95 不变。永久删除的 KV 不可恢复。

## 设置与门控

Qwen3-4B、BF16、FlashAttention2、A100 80GB、MATH500原500条输入、batch2（每题两次采样）、temperature0.6、生成Top-p0.95、seed42、总长度32768、两token不计时warmup。与已完成随机压缩版逐题核验 input_tokens 和 input_ids_sha256。

先做小张量正确性/GPU BF16参考及回归测试，然后完成同三题（idx0/1/2）共六条完整生成。核验覆盖、输入、exact pair预算、累计RoPE位置、退出状态和评分后自动提交全500题，共1000条输出。全量8分片、最多4GPU并发，Slurm安排资源，不终止其他作业。

新增测试覆盖：当前query更新、正确GQA、独立参考TopK、保护位置预排除、概率并列、预算与短历史、长prompt、固定存储、batch过滤、不可恢复删除、连续多次压缩、BF16 attention独立数值参考与全局随机状态不变。旧缓存相关测试也在GPU smoke前运行。

报告与已完成随机压缩版直接对照：准确率、逐题配对结果、EOS/长度上限、输出长度、prefill/decode/total均值/中位数/P90、显存、实际selected/causal pairs、global sparsity。固定相同预算和周期不保证实际pair总数相同，因为生成长度可能变化；每条均由相同预算公式独立核验。单seed、不同A100节点，完整生成耗时不能独立解释为选择器加速。

## 追踪

- `pipeline_tests.json`：提交前模型哈希、CPU选择器和500条历史预算指标回归。
- `correctness.log/json`：分配GPU上的完整正确性测试；不通过则禁止生成。
- `smoke_submission.json`、`smoke_collection_status.json`、`smoke_gate.json`：三题试运行与门控。
- `formal_submission.json`、`formal_collection_status.json`：全量自动提交及核验。
- `SMOKE_REPORT.md` / `FORMAL_REPORT.md`：阶段结果及与随机基线比较。
- `math/<phase>/shard_*`：原始输出、指标、硬件、退出码；`math/<phase>/summary.json`、`grades.jsonl`、`comparison.json` 为汇总。
- `implementation_alignment.json` / `inference_adapter.diff`：与已完成随机版的源码差异；已有推理文件仅增加factory/config/metrics分支，基类压缩实现未改。

2026-09-26 23:51 EDT：提交前检查通过。GPU correctness + smoke作业 `1346159_0` 已在 n0003 启动；CPU评分/gate为 `1346160`。正式实验尚未启动，需GPU correctness和三题smoke核验通过后自动提交。

GPU验证更新：42项 correctness/回归全部通过（含新增W1压缩5项），模型和源码preflight通过，完整生成smoke已完成warmup并开始运行。尚无正式准确率结果。
