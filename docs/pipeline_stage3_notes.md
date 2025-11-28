# 阶段 3 记录：图像与音频模态接入

- **模态调度骨架**：`PipelineOrchestrator` 新增 `_run_single_modality_stage`，统一处理锁文件、配置哈希与 resume 场景，并根据 `modality` 分发至 `_run_image_stage` / `_run_audio_stage`。
- **图像阶段实际执行**：读取 sorter manifest 生成 `input_manifest.txt`，通过环境变量 `PIPELINE_IMAGE_INPUT_LIST` / `PIPELINE_IMAGE_TOTAL` 传给入口；执行后落盘 stdout/stderr 并在 summary 中记录 `processed_files`、耗时、命令。
- **音频阶段接入**：逻辑与图像阶段一致，环境变量改为 `PIPELINE_AUDIO_INPUT_LIST` / `PIPELINE_AUDIO_TOTAL`；保持子集抽样与自定义 `env`、`args` 支持。
- **文本阶段接入**：新增 `_run_text_stage`，同样写入 manifest 子集并通过 `PIPELINE_TEXT_INPUT_LIST` / `PIPELINE_TEXT_TOTAL` 提供给入口脚本，执行结果与日志统一落盘。
- **配置字段扩展**：`pipelines/config.py` 会规范化 `entrypoint`、`workdir` 等路径；图像/音频配置均可指定 `manifest_subset_count`、`args`、`env`。
- **模态入口封装**：`pipelines/modalities/{image,audio,text}_runner.py` 读取 `PIPELINE_*_INPUT_LIST`、`PIPELINE_*_TOTAL`、`PIPELINE_*_OUTPUT_DIR` 等环境变量；图像入口现委托 `image.method.pipeline_api.run_image_pipeline`，优先尝试 `open_clip` 嵌入 + 余弦相似度去重并在缺失依赖时退回平均 RGB 统计，音频/文本仍保留基于文件哈希的占位逻辑，统一拷贝唯一文件并写入 `<modality>_duplicates.json`。
- **脚本执行兼容性**：runner 在启动时会将项目根目录注入 `sys.path`，避免以脚本模式运行时出现 `ModuleNotFoundError: pipelines`。
- **示例配置**：`docs/examples/pipeline_image_stage_example.yaml`、`docs/examples/pipeline_audio_stage_example.yaml`、`docs/examples/pipeline_text_stage_example.yaml` 已切换为上述入口并提供参考模板。
- **烟囱测试配置**：补充 `tests/pipeline_smoke_config.yaml`，指定极小样本、单次重试与独立输出目录，可用于验证 orchestrator + 轻量 runner 的端到端流程（仍需手动准备输入样例）。
- **烟囱测试验证**：`python -m pipelines --config tests/pipeline_smoke_config.yaml` 已跑通，图像/文本阶段成功写 summary，音频阶段在无样本时自动跳过。
- **输出归档**：summary JSON 现按 `report.summary_file` 名称写入 `output_root`，若未配置 markdown 报告则记录并跳过。
- **通用重试机制**：`general.retry` 可设置 `max_retries`、`delay_seconds`，模态配置也能局部覆盖；执行失败会记录每次错误并按需等待后重试。
- **未完事项**：仍需实现并行执行上限、端到端小样本测试与报告阶段落地。
- **可选依赖治理**：`image.method.pipeline_api` 会在运行时判定 `open_clip`、`torch`、`Pillow` 的可用性；若仅缺少前两者则自动回退到平均 RGB 特征，缺少 `Pillow` 时直接提示安装需求，确保 orchestrator 在缺省环境下也能出具可读错误。
- **异常样本容错**：当 smoke 数据集中包含损坏/非图像文件时，pipeline 会记录失败条目并把它们记入 `missing`，整体 stage 不再因平均 RGB 后端找不到合法样本而直接报错。
- **烟囱数据替换**：`tests/smoke_dataset/image/sample1.png`、`sample2.jpg` 已换成真实图像，方便验证嵌入+去重流程；旧的损坏样本仍可在 Git 历史中找到备份。
- **预计算嵌入复用**：`EmbeddingConfig` 新增 `precomputed_embeddings` / `precomputed_index` 字段，可直接将旧管线产出的 `.npy` 向量与路径索引映射进 orchestrator，缺失条目会自动计入 `missing` 并对剩余项走去重；若指定 `save_embeddings_dir` 则会把本次计算得到的嵌入落盘。新建 `image/method/legacy_integration.py` 封装加载/保存逻辑，为后续聚类与 SemDeDup 的迁移打基础。
- **SemDeDup 迁移**：`DedupConfig.method` 现支持 `sem_dedup`，结合 `legacy_config_file` / `legacy_keep_indices_file` / `legacy_cluster_dir` 可直接读取旧流程的 `all_kept_samples.txt` 与簇分布，按簇归属为当前输入构建去重结果并保留相似度分数；缺失配置或索引不匹配时自动回退到现有 pairwise 去重。
