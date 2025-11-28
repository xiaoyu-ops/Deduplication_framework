# 多模态去重流水线筹备概要

本概要用于阶段 0 的分析与准备，梳理现有模块、可复用入口、计划中的配置结构，以及后续阶段需要注意的关键点。

## 当前进展快照（2025-11）

- 新增 `pipelines/multimodal_runner.py` 作为统一 CLI 入口，可通过 `python -m pipelines.multimodal_runner --config <配置文件>` 启动整套流程。
- Orchestrator 现在在 `artifacts/<run_id>/run_manifest.json` 中实时记录阶段计划、执行结果与时间戳，便于调试与断点续跑。
- 提供示例配置 `pipelines/configs/example_pipeline.yaml`（默认仅跑 sorter 阶段且无需 Conda 环境）。需要启用具体模态时，可把 `enabled: false` 改为 `true` 并补齐相应 `output_dir`、`config_file` 与环境配置。

## 现有模块梳理（可复用入口）

- **Sorter (`sorter.py`)**

  - 函数 `sorter(files, eval_mode=False, prediction_path=None, input_root=None, move_base_dir=None)` 可直接复用。
  - 需要新增参数控制是否移动文件（默认保持原位），并输出分类 manifest。

- **图像管线 (`image/method/`)**

  - `main.py`: 生成图像嵌入，输入来源配置于 `image_config.json`。
  - `load_and_convert_embeddings.py`: 转换嵌入（FAISS 兼容）。
  - `run_clustering_local.py`: KMeans 聚类，生成 `kmeans_centroids.npy` 等文件。
  - `sort_clusters_in_windows.py` & `sort_clusters.py`: 对聚类结果进行排序。
  - `simple_semdedup.py`: 根据簇内相似度执行去重，依赖 `semdedup_configs.yaml`。
  - `create_deduped_dataset.py`: 根据保留列表导出去重数据集。
  - 以上脚本均通过 CLI 调用，需要封装为函数并在流水线中按照阶段顺序串联。

- **音频管线 (`audio/method/`)**

  - `spectrum_fingerprint.py`: 生成指纹（`binary_array_dict.npy`）。
  - `audio_dedup_main.py`: 批量计算 LSH 配置、查找相似对、执行去重。
  - `caculate_dedup.py`: 提供去重辅助函数（构建组合、生成报告等）。
  - 需要改造 `audio_dedup_main.py`，拆分为：指纹检查/生成、相似对计算、去重执行、结果导出。

- **文本管线 (`text/method/`)**

  - `jaccard_deduplication.py`: Jaccard 去重核心逻辑。
  - `clean_the_dataset.py`: CLI 包装，含数据加载、交互、并行策略。
  - 计划保留 `DatasetCleaner` 类，剥离交互逻辑，启用配置驱动执行。

- **中心管理器 (`center_manager.py`)**
  - 当前 CLI 仅负责环境选择和脚本调用；后续 orchestrator 将取代其职责。

> 图像、音频、文本三个模态会在 sorter 阶段完成后并行触发：流水线会把各自的去重子流程交给独立执行器实例，确保互不阻塞，同时在所有模态完成后汇总结果。

## 计划中的配置结构

新建顶层配置（示例 `configs/pipeline.yaml`）：

```yaml
general:
  run_name: "demo-run"
  input_root: "./mix_dataset"
  output_root: "./outputs"
  temp_root: "./artifacts"
  resume: true
  retry:
    max_retries: 1
    delay_seconds: 10
  only_modalities: [image, audio, text] # 可选
  parallel_modalities: true

executor:
  type: "local" # 预留 future: "submitit"
  envs:
    sorter: "base" # 可选，若 sorter 有独立环境
    image: "tryamrosemdedup"
    audio: "audio"
    text: "text-dedup"

sorter:
  move_files: false
  manifest_name: "sorter_manifest.csv"
  prediction_path: "predictions.csv"
  unknown_policy: "log" # 或 "skip" / "copy"

image:
  enabled: true
  entrypoint: "pipelines/modalities/image_runner.py"
  workdir: "."
  args: []
  manifest_subset_count: 100
  env:
    PIPELINE_IMAGE_MODE: "batch"
  config_file: "image/method/image_config.json"
  semdedup_config: "image/method/semdedup_configs.yaml"
  eps_list: [0.15]
  keep_policy: "hard"
  max_clusters: null
  checkpoints:
    embeddings: "embeddings/image_embeddings.npy"
    centroids: "image/method/dedup_results/kmeans_centroids.npy"

audio:
  enabled: true
  entrypoint: "pipelines/modalities/audio_runner.py"
  workdir: "."
  args: []
  manifest_subset_count: 100
  env:
    PIPELINE_AUDIO_MODE: "batch"
  config_file: "audio/method/audio_config.json"
  thresholds: [0.8]
  ensure_fingerprint: true
  fingerprint_path: "audio/binary_array_dict.npy"

text:
  enabled: true
  entrypoint: "pipelines/modalities/text_runner.py"
  workdir: "."
  args: []
  manifest_subset_count: 500
  env:
    PIPELINE_TEXT_MODE: "batch"
  dataset_path: "text/dataset/dataset.json"
  output_format: "json"
  threshold: 0.8
  ngram_size: 3
  mode: "standard" # 或 "fast" / "parallel"
  parallel_workers: 4

report:
  summary_file: "summary.json"
  markdown_file: "report.md"
```

> 说明：
>
> - 每个模块可在配置中单独开启/关闭，支持 `only_modalities` 覆盖。
> - `executor.envs` 用于按阶段调用对应的 Conda 环境；后续扩展集群执行时新增 `submitit` 等实现。
> - 图像阶段入口脚本需读取环境变量 `PIPELINE_IMAGE_INPUT_LIST`（待处理文件列表）与 `PIPELINE_IMAGE_TOTAL`（总候选数），并可结合自定义 `args`、`env` 进行扩展。
> - 音频阶段入口脚本需读取环境变量 `PIPELINE_AUDIO_INPUT_LIST` 与 `PIPELINE_AUDIO_TOTAL`，其他参数可通过 `args` 与 `env` 注入。
> - 文本阶段入口脚本需读取 `PIPELINE_TEXT_INPUT_LIST` 与 `PIPELINE_TEXT_TOTAL`，以 manifest 子集驱动去重流程。
> - `general.retry` 可定义模态阶段的最大重试次数与重试间隔；单个模态配置也可覆盖 `max_retries`。
> - 音频阶段入口脚本需读取环境变量 `PIPELINE_AUDIO_INPUT_LIST` 与 `PIPELINE_AUDIO_TOTAL`，其他参数可通过 `args` 与 `env` 注入。
> - `checkpoints` 字段用于断点续跑时的文件检测与验证。

## 阶段产物约定（初稿）

- `artifacts/
├─ <run_timestamp>/
│   ├─ stage0_ingest/manifest.json
│   ├─ stage1_sorter/manifest.csv
│   ├─ stage2_image/summary.json
│   ├─ stage3_audio/summary.json
│   ├─ stage4_text/summary.json
│   └─ logs/pipeline.log
└─ latest -> <run_timestamp>`

- `outputs/
├─ report.md
├─ summary.json
├─ image/
├─ audio/
└─ text/`

## 断点与幂等策略

- 每个 stage 在执行前检查对应 `manifest`/关键文件是否存在且完备，默认跳过；可通过 `--force-stage stage2_image` 重跑。
- 重要文件写入采用临时文件 + 原子重命名。
- 统一记录阶段耗时、输入/输出路径、关键参数，写入 `summary.json`。

## 后续阶段关注点

- Stage 1（Sorter）需要调整函数签名并输出统计数据。
- Stage 2/3/4 在封装时谨慎处理外部配置依赖，避免硬编码路径，确保可通过 pipeline config 注入。
- 并行调度 Stage 2/3/4 时，需要合理设置资源上限（线程/进程数、GPU 使用），并在 orchestrator 中等待所有模态完成再进入汇总阶段。
- 需要为大型数据集设置内存/并行参数，例如图像聚类的批量大小、文本去重的块大小。
- 自构小样例数据集用于测试，并在 CI 或本地脚本中验证流水线正确性。
