# 多模态去重流水线使用手册

本手册覆盖 Deduplication Framework 的整体结构、配置方法与运行步骤，帮助你快速上手并排查常见问题。该框架以 `PipelineOrchestrator` 为核心，串联排序、图像、音频、文本三个模态的去重流程，最终生成汇总报告。

---

## 1. 术语与组件

- **排序器 (Sorter)**：`sorter.py`，负责扫描输入目录并按模态拆分文件，输出 manifest 与统计信息。
- **Orchestrator**：`pipelines/orchestrator.py`，根据配置依次运行排序器与各模态阶段，并写入 `artifacts/<run_id>/`。
- **模态 Runner**：位于 `pipelines/modalities/`，分别为 `image_runner.py`、`audio_runner.py`、`text_runner.py`，使用环境变量接收 manifest、输出目录与配置路径。
- **Pipeline API**：`image/audio/text/method/pipeline_api.py`，封装对应模态的去重逻辑，可单独调用或在 Runner 中引用。
- **报告阶段**：聚合各阶段统计，生成 `outputs/summary.json` 与可选 `report.md`。

---

## 2. 快速开始

1. **准备运行环境**

   - Python 3.9+。
   - 建议在 Conda 虚拟环境中安装依赖，可按需选择：
     - 图像模态（可选 GPU）: `pip install -r requirements/image_requirements.txt`
     - 音频模态: `pip install -r requirements/audio_requirements.txt`
     - 文本模态: `pip install -r requirements/text_requirements.txt`
   - 图像阶段若需 CLIP 向量，请安装 `open_clip` 与 `torch`，否则默认回退到 `average_rgb` 提取。

2. **准备输入数据**

   - 将待处理文件放入某个根目录，例如 `./mix_dataset`。目录内可以含有任意层级的混合文件。
   - 可先运行排序器了解数据分布：
     ```cmd
     python sorter.py --input mix_dataset --eval --predictions outputs/sorter_predictions.csv
     ```

3. **复制并修改示例配置**

   - 推荐以 `pipelines/configs/example_pipeline.yaml` 为模板，放到自定义位置（例如 `configs/my_pipeline.yaml`），再按照下节说明调整路径与开关。

4. **执行流水线**
   ```cmd
   python -m pipelines.multimodal_runner --config configs/my_pipeline.yaml
   ```
   - 运行过程中会在 `artifacts/<run_id>/` 下生成阶段日志、manifest 与 `_SUCCESS`/`_FAILURE` 标记。
   - 成功后可在配置中 `report.summary_file` 指向的位置查看汇总 JSON。

---

## 3. 配置文件详解

配置文件使用 YAML/JSON，根节点由以下部分组成：

| 节点                       | 关键字段                                                                       | 说明                                                                                                                        |
| -------------------------- | ------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------- |
| `general`                  | `input_root`, `output_root`, `temp_root`, `resume`, `retry`                    | 指定输入/输出/临时目录与断点续跑策略。`retry` 包含 `max_retries`、`delay_seconds`。                                         |
| `executor`                 | `type`, `conda_executable`, `envs`                                             | 默认 `type: local`。`envs` 可为 `sorter`, `image`, `audio`, `text` 指定 Conda 环境名称。                                    |
| `sorter`                   | `enabled`, `manifest_name`, `prediction_path`, `move_files`                    | 控制排序阶段是否执行、输出 manifest 文件名、是否移动文件。                                                                  |
| `image` / `audio` / `text` | `enabled`, `entrypoint`, `workdir`, `output_dir`, `config_file`, `args`, `env` | 定义对应 Runner 的脚本位置、工作目录、输出路径、可选参数与额外环境变量。`config_file` 会传给各自的 `load_pipeline_config`。 |
| `report`                   | `summary_file`, `markdown_file`                                                | 设定汇总 JSON 与 Markdown 报告的输出路径。若在 CLI 加 `--no-report`，该阶段会被跳过。                                       |

### 3.1 通用配置字段

- **`general.input_root`**：必填。待处理数据所在的根目录，排序器会递归扫描该路径。
- **`general.output_root`**：必填。模态去重结果、汇总报告、拷贝副本都写入此目录。
- **`general.temp_root`**：可选。默认 `./artifacts`，Orchestrator 会在其中按 `run_id` 建立阶段目录、日志和状态文件。
- **`general.resume`**：布尔值。开启后会在阶段开始前检查 `_SUCCESS` 与配置哈希，相同则跳过执行，实现断点续跑。
- **`general.retry`**：包含 `max_retries`、`delay_seconds`。模态阶段失败时按照该策略重试，可在单个模态配置中覆盖。

### 3.2 执行器配置

- **`executor.type`**：当前支持 `local`。后续若扩展到队列/集群，可在此切换实现。
- **`executor.conda_executable`**：指定 Conda/Mamba 可执行文件，留空则使用系统 PATH 中的默认值。
- **`executor.envs`**：映射 `{stage: env_name}`。常见键包括 `sorter`、`image`、`audio`、`text`、`report`，Orchestrator 会在运行阶段前自动切换到对应 Conda 环境。

  - **示例**：

    ```yaml
    executor:
      type: local
      envs:
        sorter: Deduplication_Framework
        image: tryamrosemdedup
        audio: audio
        text: text-dedup
    ```

    在此示例中，每个阶段使用独立 Conda 环境隔离依赖；请确保这些环境已通过 `conda env list` 可见，并且音频/图像阶段所需的第三方包都安装在对应环境里（orchestrator 只负责 `conda run -n <env>`，不会自动安装依赖）。

### 3.3 排序器配置

- **`sorter.enabled`**：关闭后需自行提供 manifest，否则模态阶段将读取到空列表。
- **`sorter.move_files`**：为 `true` 时会把文件移动到项目内的 `image/audio/text/dataset/` 目录。为 `false` 时仅输出 manifest 与预测文件。
- **`sorter.prediction_path`**：CSV 文件位置，记录每个文件的模态预测，可用于离线评估。
- **`sorter.manifest_name`**：manifest 文件名，实际输出位于 `artifacts/<run_id>/stage1_sorter/<manifest_name>`。
- **`sorter.move_base_dir`**：可选，覆盖默认的移动目标根目录。

### 3.4 模态阶段配置

三个模态的字段含义一致：

| 字段                    | 说明                                                              |
| ----------------------- | ----------------------------------------------------------------- |
| `enabled`               | 是否执行该模态阶段。                                              |
| `entrypoint`            | Runner 脚本路径，通常位于 `pipelines/modalities/`。               |
| `workdir`               | 运行子进程时的工作目录。                                          |
| `output_dir`            | 模态输出、duplicates 明细的写入位置。                             |
| `config_file`           | 传递给 `load_pipeline_config` 的 YAML/JSON，决定细粒度算法行为。  |
| `args`                  | 追加给 runner 的命令行参数列表。                                  |
| `env`                   | 附加环境变量，可用于覆盖默认参数（如 `PIPELINE_*_CONFIG_FILE`）。 |
| `manifest_subset_count` | 可选，仅使用 manifest 前 N 条进行抽样调试。                       |
| `max_retries`           | 可选，覆盖 `general.retry` 的设置。                               |

Orchestrator 会为每个模态注入以下环境变量：

- `PIPELINE_<MODALITY>_INPUT_LIST`：排序器生成的 manifest 子集路径。
- `PIPELINE_<MODALITY>_TOTAL`：该模态候选的原始数量。
- `PIPELINE_<MODALITY>_OUTPUT_DIR`：输出目录，若未配置则为空。
- `PIPELINE_<MODALITY>_CONFIG_FILE`：当 `config_file` 字段存在时自动注入，也可通过 `env` 重新指定。

### 3.5 模态内部配置

各模态的 `config_file` 由对应的 `pipeline_api` 解析，常用字段如下：

- **图像 (`image/method/pipeline_api.py`)**

  - `embedding.backend`：`auto`（优先 open_clip，其次 `fallback`）、`open_clip`、`average_rgb` 等。
  - `embedding.precomputed_embeddings` / `precomputed_index`：指定 legacy `.npy` 文件，实现免计算模式。
  - `embedding.save_embeddings_dir`：可选。完成后将实时嵌入持久化，便于下游使用。
  - `dedup.method`：`pairwise`（默认）或 `sem_dedup`（需提供 legacy keep/cluster 文件）。
  - `dedup.eps`、`dedup.max_candidates`：控制相似度阈值、最大候选数量。

- **音频 (`audio/method/pipeline_api.py`)**

  - `embedding.fingerprint_backend`：`auto`、`precomputed` 或 `compute`。无外部依赖时推荐预生成指纹并配置路径。
  - `embedding.precomputed_fingerprints`：指向 `.npy` 格式的指纹字典，键可为文件名或绝对路径。
  - `embedding.save_fingerprints_dir`：可选。执行后将指纹存盘。
  - `dedup.method`：目前实现 `jaccard`，`threshold` 控制相似度下限，`max_candidates` 限定批量规模。

- **文本 (`text/method/pipeline_api.py`)**
  - `embedding.ngram_size`、`lowercase`、`strip_non_alnum`、`collapse_whitespace`：控制文本预处理与特征粒度。
  - `dedup.method`：当前支持 `jaccard`。通过 `threshold` 设定保留标准，`max_candidates` 用来限制单批规模。

> 建议先在交互式环境调用 `load_pipeline_config` + `run_*_pipeline` 进行烟测，再交由 orchestrator 执行，可快速验证配置有效性。

### 最小可运行配置示例（仅文本模态）

```yaml
general:
  input_root: ./mix_dataset
  output_root: ./outputs/demo_run
  temp_root: ./artifacts/demo_run
  resume: false

executor:
  type: local
  envs: {}

sorter:
  enabled: true
  manifest_name: manifest.csv
  move_files: false

image:
  enabled: false

audio:
  enabled: false

text:
  enabled: true
  entrypoint: ./pipelines/modalities/text_runner.py
  workdir: .
  output_dir: ./outputs/demo_run/text
  args: []

report:
  summary_file: ./outputs/demo_run/summary.json
  markdown_file: ./outputs/demo_run/report.md
```

> 路径既可绝对也可相对，框架会在加载时自动转换为绝对路径。

---

## 4. 阶段执行流程

### 4.1 排序阶段

- 从 `general.input_root` 递归收集文件，调用 `sorter.sorter()` 判断模态。
- 输出 CSV manifest（字段：源路径、相对路径、类别、状态等），并在 orchestrator 中记录统计。
- 若 `move_files: true`，会把文件移动到项目根下的 `image/audio/text/dataset/`。

### 4.2 模态阶段

每个模态读取 sorter 产出的 manifest 子集，并由 Runner 负责：

| 模态 | Runner                                 | 输入环境变量                                                                                                   | 输出内容                                                                                    |
| ---- | -------------------------------------- | -------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| 图像 | `pipelines/modalities/image_runner.py` | `PIPELINE_IMAGE_INPUT_LIST`, `PIPELINE_IMAGE_TOTAL`, `PIPELINE_IMAGE_OUTPUT_DIR`, `PIPELINE_IMAGE_CONFIG_FILE` | 在输出目录写入 `image_runner_summary.json` 与 `image_duplicates.json`，并可复制保留的文件。 |
| 音频 | `pipelines/modalities/audio_runner.py` | `PIPELINE_AUDIO_*`                                                                                             | 同上，统计字段包含指纹后端、Jaccard 相似度。                                                |
| 文本 | `pipelines/modalities/text_runner.py`  | `PIPELINE_TEXT_*`                                                                                              | 生成文本去重统计，支持 N-gram Jaccard。                                                     |

> 各 Runner 底层调用对应的 `run_*_pipeline()`，可在 `image/audio/text/method/pipeline_api.py` 中调整阈值、回退策略及持久化行为。

### 4.3 报告阶段

- 汇总 sorter 与模态统计，计算吞吐率、重复率、缺失数等指标。
- 输出 `summary.json`，结构大致包含 `aggregated.sorter`, `aggregated.modalities`, `aggregated.overall` 等节点。
- 若配置 `report.markdown_file`，还会生成 Markdown 报告，便于人工复核。

---

## 5. 运行产物结构

运行后目录示例：

```
artifacts/
  20251124-153012/
    logs/pipeline.log
    stage1_sorter/manifest.csv
    stage2_image/{input_manifest.txt, stdout.log, stderr.log, summary.json}
    stage2_audio/...
    stage2_text/...
outputs/
  summary.json
  report.md
  image/
    image_runner_summary.json
    image_duplicates.json
  audio/
    ...
  text/
    ...
```

关键文件说明：

- `pipeline.log`：全局日志，包含每个阶段的开始/结束时间与命令。
- `stageX_*/summary.json`：阶段元数据（执行命令、耗时、配置哈希等）。
- `_SUCCESS` / `_FAILURE`：手动重跑时可检查或删除的标记文件。
- `*_runner_summary.json`：模态统计输出，`duplicates_file` 字段指向重复项明细。

---

## 6. 调试与故障排查

1. **阶段失败**

   - 查看 `artifacts/<run_id>/stage*/stderr.log` 与 `stdout.log` 获取 Runner 实际输出。
   - `artifacts/<run_id>/stage*/summary.json` 中的 `metadata.error` 字段通常包含简要原因。
   - 若需要重跑，删除对应阶段目录下的 `_SUCCESS`/`_FAILURE` 后再执行 orchestrator，或将配置中的 `resume` 设为 `false`。

2. **排序器未发现文件**

   - 确认 `general.input_root` 指向的目录存在，并具备读取权限。
   - 若目录为空，可先运行 `python sorter.py --input <dir> --eval` 进行快速诊断。

3. **模态 Runner 缺少依赖**

   - 图像阶段报 `open_clip`/`torch` 未找到：安装对应包或在 `image` 配置里显式指定 `config_file` 使用 `average_rgb` 后端。
   - 音频阶段若提示 `spectrum_fingerprint` 缺失，请先安装 `requirements/audio_requirements.txt` 中的依赖并确保输入为可解析的音频格式。

4. **输出目录缺失或无写权限**

   - Runner 在创建输出目录前会调用 `ensure_output_dir`，若失败请检查路径合法性与权限。
   - 推荐在配置中使用绝对路径或调用 `Path.resolve()` 后的结果。

5. **报告阶段缺少统计数据**

   - 当某模态被禁用或未生成 `runner_summary` 时，`aggregated.modalities` 会记录 `enabled: false`。若期望包含统计，需要确保该模态成功写入 `*_runner_summary.json`。

6. **并行调试建议**
   - 先用单模态配置验证，再逐个启用其他模态，可以更快定位问题来源。
   - 使用小型样本数据（例如 `tests/smoke_dataset`）可在几秒内验证整体流程。

---

## 7. 验证与测试

- **端到端冒烟测试**：

  ```cmd
  python -m pytest tests/test_pipeline_end_to_end.py
  ```

  该测试会在临时目录执行完整文本模态流程，确保 orchestrator、排序汇总与报告逻辑正常。

- **自定义数据验证**：
  - 在 `tests/smoke_dataset` 基础上加入图像/音频样本，修改测试或手动运行流水线确认统计字段。
  - 若需要集成测试，可在 `tests/` 目录下新增用例并复用 `_write_pipeline_config` 辅助函数。
  - 已内置 `test_pipeline_end_to_end_image_modality`、`test_pipeline_end_to_end_audio_modality` 与 `test_pipeline_end_to_end_all_modalities` 三个用例，分别覆盖单模态和三模态回归。可执行 `python -m pytest tests/test_pipeline_end_to_end.py::test_pipeline_end_to_end_all_modalities` 快速验证整套流程。

---

## 8. 常用脚本速查

| 位置                                      | 用途                                                            |
| ----------------------------------------- | --------------------------------------------------------------- |
| `sorter.py`                               | 扫描目录并按模态分类，支持评估/移动两种模式。                   |
| `pipelines/multimodal_runner.py`          | CLI 入口，通过 `--config` 指定 YAML/JSON 配置执行整条流水线。   |
| `pipelines/modalities/common.py`          | Runner 共用的 manifest 读取、输出目录创建、重复写入等工具函数。 |
| `image/audio/text/method/pipeline_api.py` | 各模态的核心去重逻辑，可单独调用或在 Runner 中引用。            |
| `tools/run_comparative_evaluation.py`     | 评估 sorter 分类效果，生成对比报告。                            |
| `tools/generate_mix_dataset_10k.py`       | 生成或采样混合模态数据集，便于实验与回归测试。                  |

---

## 9. 常见问答

- **是否必须启用所有模态？**
  否。将对应模态的 `enabled` 设为 `false` 即可跳过，Orchestrator 会在最终汇总中标记为未启用。

- **如何在同一台机器上使用多个 Conda 环境？**
  在配置的 `executor.envs` 中为各阶段填入环境名称，例如：

  ```yaml
  executor:
    type: local
    envs:
      sorter: base
      image: image-env
      audio: audio-env
      text: text-env
  ```

  Orchestrator 会在调用执行器时自动切换。

- **可以断点续跑吗？**
  可以。设置 `general.resume: true` 后，若阶段已经存在 `_SUCCESS` 标记且配置未变化，会直接跳过。若需要强制重跑，请删除对应阶段目录或修改配置（改变 `config_hash`）。

- **如何获取详细的重复样本列表？**
  在每个模态的输出目录查看 `*_duplicates.json`。文件结构包含 `original`、`duplicates`（带相似度）等字段，便于后续审查或二次处理。

---

如有更多需求，可阅读 `docs/pipeline_overview.md` 与阶段笔记（`pipeline_stage*.md`），了解设计背景与后续扩展计划。欢迎在新增功能时补充本手册相应章节，保持文档与代码同步。
