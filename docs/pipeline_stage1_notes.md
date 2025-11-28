# 阶段 1 记录：基础框架搭建

- 新建 `pipelines/` 包并加入：

  - `executor.py`：实现 `BaseExecutor` 接口与 `LocalExecutor`，支持 `conda run -n <env>` 调用，同时提供结果封装与错误处理。
  - `artifacts.py`：定义 `StageArtifact` 数据结构与 JSON 保存方法。
  - `config.py`：统一加载 YAML/JSON 配置，封装 `ExecutorConfig` / `PipelineConfig`。
  - `logger.py`：集中设置日志输出（控制台 + 文件）。
  - `orchestrator.py`：构建 orchestrator 主体，负责初始化配置/执行器、管理 run id、记录阶段输出、生成 summary。
  - `__main__.py`：提供 CLI 入口 `python -m pipelines --config ...`。

- 约定 artifacts 目录结构：`artifacts/<run_id>/stageX_*/summary.json` + `logs/pipeline.log`。

- Orchestrator 暂时提供占位方法：`run_sorter_stage` / `run_modality_stages` / `run_report_stage`，后续阶段会补充具体实现。

- 现有代码尚未整合并行执行与阶段跳过逻辑，下一步（阶段 2）将从 sorter stage 开始补全。
