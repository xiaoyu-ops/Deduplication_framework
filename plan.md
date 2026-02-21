# 实验计划 — 并发与流水线吞吐量对比

目的：通过一系列控制实验收集吞吐量数据，找出并发与流水线（orchestrator）对系统吞吐的影响，并测试对数据中三模态（image/audio/text）比例的影响，寻找优化/创新点。

说明：
- 你会手动运行每个实验；此文件给出每次实验需要的配置变更和运行建议（在 `pipelines/multimodal_runner.py` 所使用的配置文件，即 pipeline 配置 YAML/JSON）。
- 记录项：每次运行请记录 `run_manifest.json`（位于 artifacts/<run_id>/run_manifest.json）以及每个 modality 的 `*_runner_summary.json`，并记录总耗时与处理条目数以计算吞吐（items/sec 或 MB/sec）。
- 建议重复每个配置 3 次取平均。

实验一：有流水线，关闭并发（基线）
- 目的：测量在 orchestrator 全流程下（包含 sorter + 各 modality stage 串行执行）吞吐。
- 配置修改（pipeline YAML 中 `general` 段）:

general:
  parallel_modalities: false
  parallel_workers: 1  # 可不设置
  batch_size: 1000      # 根据数据规模调整

- 运行：使用现有多模态入口运行完整流水线：

powershell:
python -m pipelines.multimodal_runner --config configs/my_pipeline.yaml

实验一（已执行示例 & 结果）：
- 运行标识: `run_id = 20251228-152819`（位于 `artifacts/big_run/20251228-有流水线无并发`）
- 关键文件: `artifacts/big_run/20251228-有流水线无并发/run_manifest.json`
- 总体（sorter）:
  - sorter elapsed: 1365.6966 s
  - 总条目 (manifest rows): 181,756 (success 179,366; fail 2,390)
  - 总字节: 4,884,761,391 bytes
  - sorter throughput: 4,884,761,391 B / 1365.6966 s ≈ 3,576,968 B/s (≈ 3.41 MiB/s)
- 各模态处理（来自 `run_manifest.json` 的 `modality_results`）:
  - image: files=24,162; bytes=1,086,022,707; stage elapsed=703.532 s; throughput ≈ 1,543,760 B/s (≈ 1.47 MiB/s)
    - runner internal elapsed (合并子任务): 544.7883 s (在 `runner_summary.stats.elapsed_seconds`)，如按该值计算 throughput ≈ 1,994,726 B/s (≈ 1.90 MiB/s)
  - audio: files=10,601; bytes=3,668,934,262; stage elapsed=2362.139 s; throughput ≈ 1,553,540 B/s (≈ 1.48 MiB/s)
  - text: files=144,603; bytes=105,470,403; stage elapsed=3384.343 s; throughput ≈ 31,168 B/s (≈ 0.03 MiB/s)

- 说明与建议:
  - 上述 throughput 以各模态 `modality_results.elapsed_seconds` 作为分母得到（image 有时也可用 runner_summary 内的 `elapsed_seconds`，见备注）。
  - 这些数字为有流水线（orchestrator）情形的基线测量；与实验二的并发测量对比时请注意计数语义与 manifest 分片策略的一致性。


实验二：无流水线，仅并发（模态并行直接运行 runner）
- 目的：去掉 orchestrator 的调度/锁/汇总开销，仅并发发起各模态 runner，测最大并发启动下的吞吐。
- 方法（两种可选）:
  A) 手工并行启动模态 runner：分别为 `image`、`audio`、`text` 的 entrypoint 启动进程，使用同样的输入清单（或各自子集），并在 shell 中并行运行。例如在 PowerShell 中使用 `Start-Process` 或多窗口并行启动。
  B) 使用一个临时脚本并行调用它们（示例见下）。
- 推荐并发数：按机器核数/IO 上限调整（例如 3、6、12）。

示例临时脚本（PowerShell）:

powershell:
Start-Process python -ArgumentList "D:/Deduplication_framework/pipelines/modalities/image_runner.py --input manifest_image.txt --output out_image" -NoNewWindow
Start-Process python -ArgumentList "D:/Deduplication_framework/pipelines/modalities/audio_runner.py --input manifest_audio.txt --output out_audio" -NoNewWindow
Start-Process python -ArgumentList "D:/Deduplication_framework/pipelines/modalities/text_runner.py --input manifest_text.txt --output out_text" -NoNewWindow

- 注意：此实验不通过 `orchestrator`，因此不会产生 run_manifest；请在各 runner 输出目录寻找 `*_runner_summary.json` 或在启动脚本中汇总 stdout/stderr 时间戳用于吞吐计算。

实验二（已执行示例 & 结果）：
- 运行参数（本次实验）: 使用仓库内临时 launcher 并发启动每个模态的 3 个实例。
  - 启动命令: `python .\scripts\run_parallel.py --instances 3 --out-root outputs/parallel_test`
  - 并发实例: `image`、`audio`、`text` 各 3 个（总 9 个子进程）
  - 输入清单: `manifests/manifest_image.txt`, `manifests/manifest_audio.txt`, `manifests/manifest_text.txt`（同一清单在多个实例中被重复使用，未做分片）
  - 输出目录: `outputs/parallel_test/20260103_214949`

- 关键测量（注意：因为清单被重复使用，计数可能重复统计；audio runner 在本次环境下出现 `processed=0` 的问题但仍报告了 `processed_bytes` —— audio 字节数需谨慎解读）
  - 单实例 processed_bytes（来自各模态第1个实例的 `*_runner_summary.json`）与对应吞吐（使用 launcher 中每进程 wall-clock elapsed 计算）:
    - image: 1,086,022,707 bytes; elapsed ≈ 190.63 s; throughput = 1,086,022,707 / 190.63 ≈ 5,697,019 B/s (≈ 5.43 MiB/s)
    - audio: 3,668,934,262 bytes; elapsed ≈ 190.61 s; throughput = 3,668,934,262 / 190.61 ≈ 19,248,383 B/s (≈ 18.36 MiB/s)
      - 警告：audio runner 当次运行 `processed=0`（指纹/配置问题），上述 bytes 来源于文件总量，请先确认指纹计算有效性或使用预计算指纹再做最终结论。
    - text: 105,470,403 bytes; elapsed ≈ 635.57 s; throughput = 105,470,403 / 635.57 ≈ 165,946 B/s (≈ 0.16 MiB/s)
  - 启动器 wall-clock 总耗时（从 launcher 的 `parallel_summary.json`）: 635.62 s

- 说明与建议:
  - 计数语义: 本次实验为了并发性测试复用了相同 manifest，导致多实例间存在重复处理路径；若需精确的系统级吞吐（去重后的 bytes/s），请在 launcher 中按实例对 manifest 进行分片（shard），或事先生成不重叠的子清单后重跑。
  - audio runner 的 `processed=0` 问题是配置/依赖导致指纹计算回退；可通过在 pipeline 配置中设置 `embedding.precomputed_fingerprints` 或安装缺失依赖（例如 `librosa`、`scikit-image`）并重跑来获得可信的 audio items 计数。
  - 如果你要我把本次测量写成 CSV 或自动化将 manifest 分片并重跑以得去重吞吐，我可以继续实现并执行。

实验三：两个都没有（既无流水线也无并发）
- 目的：测量单一 runner 串行执行（单模态、单进程）吞吐，作为最低开销基线。
- 方法：只运行单个模态的 entrypoint，处理同样数量的数据（或完整 manifest），记录耗时。

运行示例：

powershell:
python path/to/text_entrypoint.py --input manifest_text.txt --output out_text


实验四：两个都有（流水线 + 并发）
- 目的：在 orchestrator 负责流程管理的同时开启模态并发，比较 orchestrator 管理开销与并发获益。
- 配置修改（pipeline YAML 中 `general` 段）:

general:
  parallel_modalities: true
  parallel_workers: 3   # 或者根据机器调整
  batch_size: 1000

- 运行：同实验一（使用 `pipelines.multimodal_runner` 运行），但确保 `parallel_modalities: true`。


模态比例调整实验（对数据内模态比例做参数化测试）
- 目的：观察不同模态比例（image:audio:text）下对整体吞吐与并发扩展性的影响，找出瓶颈模态。
- 测试集设定：准备若干 manifest（或从原 manifest 过滤/采样）以构造不同比例，例如：
  - A: 70% image / 15% audio / 15% text
  - B: 40% image / 30% audio / 30% text
  - C: 10% image / 45% audio / 45% text
  - D: 均衡 33/33/34
- 对每个比例分别运行以上 4 种并发/流水线组合，记录吞吐与资源占用（CPU、内存、磁盘/网络 IO if possible）。

配置片段（示例 pipeline YAML）
- 基本模板（只展示相关字段）：

general:
  input_root: "path/to/input_root"
  output_root: "outputs"
  temp_root: "artifacts"
  batch_size: 1000
  parallel_modalities: false   # 改为 true 用于并发试验
  parallel_workers: 1
  retry:
    max_retries: 0
    delay_seconds: 0

image:
  entrypoint: "path/to/image_runner.py"
  output_dir: "outputs/image"
  batch_size: 1000
  workdir: "path/to/working_dir"
  env: {}

audio:
  entrypoint: "path/to/audio_runner.py"
  output_dir: "outputs/audio"
  workdir: "path/to/working_dir"

text:
  entrypoint: "path/to/text_runner.py"
  output_dir: "outputs/text"
  workdir: "path/to/working_dir"

- 说明：把 `general.parallel_modalities` 与 `general.parallel_workers` 调整为实验所需值；通过 `batch_size` / `manifest_subset_count` 调整每个 runner 的批量大小。

记录模板（建议 CSV/TSV）
- columns: experiment_id, run_id, pipeline_enabled (yes/no), parallel_modalities (true/false), parallel_workers, modality_ratio, batch_size, total_items, total_bytes, elapsed_seconds, items_per_second, errors, notes

示例实验矩阵（建议顺序）
- 对每个模态比例 (A,B,C,D):
  - Run exp1 (pipeline, serial)
  - Run exp2 (no pipeline, parallel)
  - Run exp3 (no pipeline, serial single-runner)
  - Run exp4 (pipeline, parallel)

执行建议与注意事项
- 每次实验前清空或使用新的 `artifacts` 输出目录以避免锁/缓存影响。
- 对并发试验，逐步增加 `parallel_workers`（例如 1,3,6,12），观察扩展性与退化点。
- 记录系统资源（建议使用 `perfmon` 或 `psutil` 脚本）以便分析瓶颈（CPU-bound、IO-bound 或 runner 本身限制）。
- 如果 runner 使用 GPU/专用资源，确保实验间隔离，避免资源争用导致测量噪声。

后续：
- 如果你需要，我可以为你生成：
  - 一组可直接替换的 YAML 配置文件样板（四种并发/流水线组合 × 若干模态比例），放入 `configs/` 目录；或
  - 一个小脚本用于启动“无流水线并发”情形的并行 runner（PowerShell 或 Python 版本），便于你重复实验。

请告诉我是否需要我把这些配置样板写入 `configs/` 下的文件，或只保留当前 `plan.md` 即可。