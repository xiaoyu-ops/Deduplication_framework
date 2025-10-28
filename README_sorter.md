# sorter.py 使用说明

`sorter.py` 是一个面向多模态去重/分类框架的轻量级文件路由器，可将混合目录中的文件自动分类到 `audio/`, `image/`, `text/` 三个子目录，或在评估模式下输出预测结果。最新版本强化了对缺失/错误扩展名、模糊容器以及空文件的判定能力，适用于大规模离线管线中的预分类阶段。

## 功能特性

- **多层判定策略**：
  - 基于文件扩展名的快速路径。
  - 魔数 / 文件头嗅探（PNG、GIF、RIFF/WAV、ID3、FLAC、SVG 等）。
  - 文本可打印度检测，区分二进制与结构化文本。
  - JSON 内容解析与投票，识别 `audio_url`、`image`、`text` 等字段。
- **鲁棒性增强**：
  - 处理缺失或错误扩展名的“tricky”样本。
  - 容错 JSON 解码，对 BOM、非 UTF-8 字符做忽略处理。
  - 对空文件、损坏文件返回 `unknown` 或 `error`，不会导致进程终止。
- **双模式**：
  - 默认模式：将文件移动到目标目录。
  - 评估模式 (`--eval`)：不移动文件，仅输出预测 CSV，与评估脚本无缝衔接。

## 依赖

- Python 3.9+
- 第三方库：[`tqdm`](https://tqdm.github.io/)

可通过 `pip install -r requirements/base.txt`（若存在）或单独安装 `tqdm`。

## 命令用法

```cmd
C:/Users/sysu/anaconda3/Scripts/conda.exe run -p C:/Users/sysu/anaconda3 --no-capture-output python sorter.py [--input <dir>] [--eval] [--predictions <csv>] [--input-root <path>] [--move-base <dir>]
```

常用参数说明：

- `--input`: 待分类目录，默认 `./mix_dataset`。
- `--eval`: 评估模式，不执行移动操作。
- `--predictions`: 评估模式下保存预测结果的 CSV 路径（默认 `predictions.csv`）。
- `--input-root`: 预测 CSV 中相对路径的基准，默认等于 `--input`。
- `--move-base`: 文件移动时的目标根目录（覆盖默认的项目根路径）。

### 示例

1. **批量分类并移动**
   ```cmd
   C:/Users/sysu/anaconda3/Scripts/conda.exe run -p C:/Users/sysu/anaconda3 --no-capture-output python sorter.py --input mix_dataset --move-base .
   ```
2. **评估模式输出预测**
   ```cmd
   C:/Users/sysu/anaconda3/Scripts/conda.exe run -p C:/Users/sysu/anaconda3 --no-capture-output python sorter.py --input mix_dataset_10k --eval --predictions evaluation/predictions_sorter.csv
   ```

## 评估结果摘要（2025-10-20）

- 数据集：`mix_dataset_10k`，10,000 个样本（image 3,333 / audio 3,083 / text 3,084 / unknown 500）。
- 指标：Accuracy 0.985，Macro F1 0.961，Weighted F1 0.968。
- 误分类：150 个样本，全部为截断的裸 PCM（`tricky/ambiguous/ambig_audio_*.bin`）。
- 详细报告：参见 `evaluation/comparative/report_20251020.md` 或 `evaluation/comparative/20251020-224637/` 目录。

## 故障排查

- **预测结果为 `unknown`**：
  - 检查文件是否为空或损坏。
  - 对裸 PCM、容器不明确的文件，可补充波形特征检测（采样率、能量）。
- **JSON 解码失败**：脚本已内置容错，若仍失败，可先用外部脚本清理格式或 BOM。
- **文件无法移动**：确保目标目录未被占用（关闭 VS Code/播放器等），或改用 `--eval` 模式产出预测后再人工处理。

## 扩展建议

- 为“ambiguous audio”类添加频域/能量分析，进一步降低剩余误判。
- 将 `sorter()` 函数封装成可复用模块，以便在批量任务或 Web 服务中调用。
- 将评估流程加入 CI，自动生成比较报告以监控回归。

如需进一步定制或接入上游管线，可结合评估脚本 `tools/run_comparative_evaluation.py` 与数据生成脚本 `tools/generate_mix_dataset_10k.py` 进行端到端测试。
