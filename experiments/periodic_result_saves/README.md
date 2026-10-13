# 定时保存实验结果

独立 CPU 服务，每 300 秒立即保存一份快照。当前运行目录为
`../representative_patterns_fast_retry03_20261001`，快照保存到该目录的 `periodic-saves/`。
服务不修改已冻结的配置、算法源码、正式输出或 Slurm worker，也不占用 GPU。

- `latest.json`：最近一次快照的时间、已保存样本数、各方法/任务进度、临时评分和文件清单。
- `snapshots/*.json`：每次保存的独立清单，保留历史进度。
- `objects/<前两位>/<sha256>`：原始结果 JSON、配套 logits NPY、校准结果、完成/失败记录、配置和来源信息的原始字节；相同文件只存一份。
- `service-state.json`：服务 PID、主机、间隔、最近保存时间和错误。
- `service.log`：后台保存日志。

正式样本只有在配套 logits 完整且 SHA256 与结果记录一致时才计入快照；正在写入或损坏的文件记录为 skipped，下次重新检查。
结果文件仍然由原 worker 每完成一个样本立即保存；快照不会保存尚未完成的生成过程或 KV 状态。
最终 `outputs/independent-audit.json` 出现后，服务会在 5 秒内触发最后一次保存并退出。
中途停止可在 `periodic-saves/` 创建 `STOP` 文件；服务先保存当前结果再退出。

启动命令（先删除用于主动停止的 STOP 文件，确认没有旧进程运行）：

```bash
python snapshot.py --run ../representative_patterns_fast_retry03_20261001 \
  --destination ../representative_patterns_fast_retry03_20261001/periodic-saves --interval 300
```

进程使用独占文件锁，阻止重复启动。`--once` 可单次保存；`--interval` 可调整周期。
保存进程默认在当前登录节点后台运行，状态文件记录所在主机；节点重启后需要按以上命令重启。
现有原始输出仍由 Slurm 作业按样本保存，独立于该进程。

恢复任意快照（将路径替换为需要恢复的历史清单，目标必须使用新的目录）：

```python
import json, pathlib, shutil
backup = pathlib.Path('/absolute/path/to/periodic-saves')
target = pathlib.Path('/absolute/path/to/new-restored-directory')
manifest = json.loads((backup / 'latest.json').read_text())
target.mkdir(parents=True, exist_ok=False)
for relative, item in manifest['files'].items():
    output = target / relative
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(backup / item['object'], output)
```

保存与原始数据位于同一项目文件系统，提供历史结果版本，不等同于异地备份。
