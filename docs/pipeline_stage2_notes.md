# 阶段 2 记录：Sorter 集成与恢复机制

- **Sorter 阶段落地执行**：`PipelineOrchestrator.run_sorter_stage` 现实际调用 sorter，生成 manifest，并记录成功/失败统计与样本路径。
- **恢复与锁控制**：引入 `_LOCK` 文件管理，避免并发触发；在 resume 模式下校验配置哈希、遇到 `_FAILURE` 或残留锁会给出明确提示。
- **Manifest 读取工具**：新增 `pipelines/manifest_utils.py`，统一解析 sorter 输出的 CSV，按模态拆分、校验字段与空文件；orchestrator 会在成功或 resume 时加载 manifest 快照。
- **汇总信息丰富化**：流水线 summary 记录 manifest 路径与各模态数量，日志中打印总量与分模态计数，便于后续阶段调度资源。
- **模态阶段预规划**：`run_modality_stages` 基于 manifest 生成 image/audio/text 的任务草案，记录文件数、目标环境与输出目录，占位 artifact 用于后续实际执行时覆盖。
- **后续待办**：基于 manifest 的模态阶段执行器、资源配额控制、失败重试策略以及并行调度仍需在阶段 3 及以后完成。
