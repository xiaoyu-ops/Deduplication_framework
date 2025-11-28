# Current Pipeline Blocker (2025-11-26)

## Run Context

- **Command**: `python -m pipelines.orchestrator --config configs\my_pipeline.yaml`
- **Environment**: `conda` env `tryamrosemdedup` (Windows)
- **Artifacts**: `artifacts/test_run/20251126-141241/`
- **Primary log**: `artifacts/test_run/20251126-141241/logs/pipeline.log`

## Observed Failure

- Orchestrator initializes and sorter stage completes (120 inputs; image/audio each 53, text 7, unknown 7).
- Image stage launch via `conda run -n tryamrosemdedup python pipelines/modalities/image_runner.py` fails immediately.
- Error captured in stage stderr and pipeline log:
  ```
  OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
  ... multiple OpenMP runtimes linked ... set KMP_DUPLICATE_LIB_OK=TRUE (unsafe workaround)
  ```
- Executor raises `pipelines.executor.ExecutorError` (exit code 3), causing pipeline abort before audio/text stages run.

## Impact

- `stage2_image` fails and blocks the entire pipeline; no downstream stages complete.
- Outputs at `outputs/test_run/*` remain incomplete; summary only records failure metadata.

## Attempts / Findings

1. Running orchestrator as a module (`python -m pipelines.orchestrator ...`) resolves earlier relative-import errors.
2. Image runner can complete when executed manually with `average_rgb` backend in the same env, suggesting issue triggers when OpenMP-heavy dependencies load under orchestrator (likely due to mixed pip/conda installs or multiple MKL/OpenMP libraries).
3. Residual `~orch*` site-packages entries already moved to `backup_invalid_dist_20251126224438`.
4. Environment currently has pip-installed `torch==2.9.1+cpu` and `open_clip_torch==3.2.0`, leading to dependency mismatches (e.g., `torchaudio` wants `torch==2.5.1`).

## Recommended Next Steps

1. **Quick verification (unsafe)**: Temporarily set `KMP_DUPLICATE_LIB_OK=TRUE` before rerunning to confirm no additional blockers exist.
2. **Proper fix**: Reconcile OpenMP providers in `tryamrosemdedup`:
   - Prefer conda-forge packages (e.g., install `intel-openmp`, ensure only one `libiomp5md.dll`).
   - Reinstall PyTorch/torchvision/torchaudio + `open_clip_torch` using consistent channels (`pytorch` + `pytorch-cuda=12.1`) or recreate env from scratch.
3. **Diagnostics**: Locate all `libiomp5md.dll` copies under `%CONDA_PREFIX%` (`where /R %CONDA_PREFIX% libiomp5md.dll`) to identify conflicting sources.
4. After environment is consistent, rerun orchestrator and capture fresh logs for confirmation.
