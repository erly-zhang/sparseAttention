# 新压缩版：GPQA-D / AIME / HMMT / LiveCodeBench

实验根目录：`results/compact_multibench_20260926`。本轮是用户授权的新压缩版跨数据集实验，保留历史结果；不新增 dense、random_pp 或其他预算对照。

## 数据范围

| 数据集 | 本轮范围 | 题数 | 正式生成次数 |
|---|---|---:|---:|
| GPQA | 官方 Diamond | 198 | 396 |
| AIME | 2025 I+II | 30 | 60 |
| AIME | 2026 I+II | 30 | 60 |
| HMMT | 2025 February+November | 60 | 120 |
| LiveCodeBench | release_v6，medium 子集 | 383 | 766 |

正式共701题、1402条输出，另有每个任务分组前三题的完整 smoke（15题、30条输出）。LCB 的结果必须标为 medium 子集，不能代表全 release。

GPQA 官方访问已在用户授权后成功。发现原 VaSE 转换版本将所有正确答案固定为A；本轮直接从官方 CSV 构建，使用 SHA256(seed42:RecordID) 初始化独立 Python Random，打乱四个选项；答案分布 A50/B47/C48/D53。不会将题目或答案上传外部网站。

AIME/HMMT/LCB 题目来自固定 VaSE commit `bfa2692112c45da13e637b3eca03ee6516f9e0bb`。LCB 测试源为官方 `livecodebench/code_generation_lite` commit `0fe84c3912ea0c4d4a78037083943e8f0c4dd505` 的6份测试数据；383题测试文件 MD5 全部与固定项目题目记录一致，共10267个测试。数据文件 SHA256、下载 URL 和转换脚本另存于根目录。

## 保持的方法和生成设置

推理源码逐文件复制自已完成 MATH500 的 `compact_kv_random_reason1024_recent64_20260926/source`，保持原37项正确性/回归验证记录和实现哈希。Qwen3-4B、BF16、FlashAttention2、A100 80GB、batch2，每题两次采样，seed42、temperature0.6、generation top-p0.95、总上下文上限32768。两token warmup排除于计时。

全 prompt 受保护，位于 reasoning1024 预算之外；recent64 计入 reasoning1024。prefill为dense，decode将KV物理保存为每层共享位置的缓存；attention后 reasoning达到1088时压回1024，即此后每64步压缩一次。各KV head保留相同位置但不同KV数值，GQA映射与原始RoPE位置不变。压缩采样、种子坐标及其他实现完全沿用完成版MATH500。

## 执行和门控

1. 数据完整性核验；用冻结 tokenizer/chat 模板保存每题 input_tokens 和 input_ids_sha256。
2. 评分器自测：GPQA boxed选项、数学答案与LCB stdin/functional/错误/超时；新 pair 检查器对已完成500条MATH指标回归，并故意破坏指标确认拒绝。
3. 5个单GPU smoke worker，各自完整生成前三题（每题两条）。
4. CPU gate检查退出码、逐题输入哈希、无重复缺失、EOS或到达长度上限、精确pair预算和缓存累计位置，并评分。模型答错不属于技术smoke失败；缺数据、预算偏差或无法评分会阻止正式提交。
5. 所有smoke通过后，自动提交16个正式GPU分片，最多8个GPU同时运行；另有依赖正式array完成的CPU汇总任务。任何缺失/失败均不得报为已完成。

GPQA严格提取最终完整 boxed A-D；AIME/HMMT使用已有数学评分器；LCB只从生成后缀提取代码，避免把prompt中starter code作为输出。LCB通过全部可用测试才算正确，20秒/测试、2048MiB/候选进程，4个评分进程。采用项目本地 grader，外部榜单比较须先匹配评估器与模板。

报告逐数据集正确数/生成数、每次rollout准确率、均值准确率（LCB为pass@1估计，非pass@2）、EOS/长度上限、输出长度、prefill/decode/total统计、显存、selected/causal原始pair总数及global sparsity。不汇总不同数据集为单一总准确率。

## 状态与检查入口

- `smoke_submission.json`：smoke GPU和CPU gate作业号。
- `pipeline_tests.json`：运行前回归检查。
- `smoke_collection_status.json`、`smoke_gate.json`、`SMOKE_REPORT.md`：smoke检查和评分。
- `formal_submission.json`：只在全部smoke通过后产生。
- `formal_collection_status.json`、`FORMAL_REPORT.md`：正式完成检查和结果。
- 每个任务的 `smoke/`、`formal/`：原始输出、指标、逐题评分和summary。
- `logs/grade_<phase>_<task>.log`：评分/核验失败详情。

2026-09-26 23:20 EDT：提交前验证全部通过，smoke GPU array `1346032`（5个worker）已运行；CPU评分/gate `1346037` 等待依赖。正式实验尚未启动，须全部smoke通过后由gate提交。
