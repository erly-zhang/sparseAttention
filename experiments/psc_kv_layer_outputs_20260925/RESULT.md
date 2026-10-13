# 运行结论：未完成，correctness gate阻止继续

Slurm job **47069472**，w002，1×H100-80GB，终态 **FAILED**，退出码 **1:0**，实际分配运行 **76秒**（2026-09-25 11:51:05–11:52:21 America/New_York）。详见 slurm_terminal.psv。没有自动重提、加时、加卡或删减方法。

## 已完成

- autoblock commit 8b21a6a18056eadc00b6248bb319a4af3c6ccc4d 审计；新代码、env、配置、下载与日志均在本目录。旧tracked代码未改。
- 指定数据revision下载、SHA与500条JSONL/schema核验。仅索引0/1/2/3构造输入，测试3固定为123959 tokens；未缩短。模型/tokenizer完整文件重新hash一致。
- 原AE提取/训练/Ward聚类/medoid流程已定位并接入编排，参数保持原配置；AE尚未运行，因此没有伪造groups/checkpoint。
- 五方法安装接口检查通过；3项CPU指标测试通过。
- H100/BF16/真实FA2：dense、C、Shared6144+member2q4096、FlexPrefill的小模型cache/future-invariance检查通过。C通过独立逐token参考与显式拼接测试。AutoBlock额外通过缩小预算下真实member2q残差TopK及masked-attention参考检查。

## 阻塞与未完成

MInference在最小模型前向时触发LLVM断言：`cast<mlir::ShapedType> ... incompatible type`，子进程SIGABRT。主流程立即停止，未执行AE提取、AE训练、聚类、真实索引3前向或任何32层对比。因此没有逐层KL/L2/RMSE/cosine或正式准确率结果；已通过的四个方法不能替代五方法对比。

目前只能确认编译阶段失败，尚不能确定具体kernel/root cause。候选原因是MInference/Triton3.2兼容性；原FlexPrefill依赖标注Triton3.0，而四个已过方法不能证明MInference兼容。可行修复方向是单独恢复并验证MInference支持的Torch/Triton/FA2组合，保留Llama pattern、算法和固定样本；或最小复现后修正编译兼容接口并保留原kernel语义。不能直接用其他sparse方法替代。

## 预算与可行缩减

本次GPU模型计算在这一个H100作业内；Slurm allocation为76秒。AE尚未测得epoch成本，因此不存在可靠的完整五方法总耗时实测估计。原AE每epoch921训练+103验证、最早patience停止31epoch、最多1000epoch；准备了真实校准map上的成本门槛和硬时限。若完整原配置超预算，可减少诊断I/O（只保存dense参考、逐层流式统计，当前已采用）；不能保证仅此足够。减少AE epoch/map_size或缩短测试输入都改变已约定配置，本次未这样做。

**预检记录更正**：MInference源码在import时创建CUDA arange/mask常量，所以此前把导入检查称为“纯CPU”不准确。该导入发生在已有开发节点；没有在该节点运行本实验模型forward或AE训练，但少量导入CUDA操作未单独计时。不能将严格的全部GPU活动统计伪称只有76秒；76秒是新H100作业的准确Slurm时长。后续应把这种import检查也放入目标H100 allocation。相关metadata已更正。

用户要求单次提交，故保留失败现场和提交锁，未利用剩余时长自动再提交。输出定义和端到端语义未改变。任何未生成文件均记为未完成。
