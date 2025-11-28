from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .orchestrator import PipelineOrchestrator


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="multimodal-runner",
        description="Run the multimodal deduplication pipeline",
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to pipeline configuration file (YAML or JSON)",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Skip report stage even if configured",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    orchestrator = PipelineOrchestrator(args.config)

    try:
        orchestrator.logger.info(
            "Starting multimodal pipeline (config=%s, report=%s)",
            args.config,
            "disabled" if args.no_report else "enabled",
        )
        if args.no_report and orchestrator.config.report:
            orchestrator.config.report = {}
            orchestrator.logger.info("Report stage requested but disabled via flag")
            orchestrator.summary.setdefault("report_outputs", {})["skipped"] = True
            orchestrator._write_run_manifest()
        orchestrator.run()
    except Exception as exc:  # pragma: no cover - top-level guard
        orchestrator.logger.error("Pipeline run failed: %s", exc, exc_info=True)
        return 1

    return 0


if __name__ == "__main__":  # pragma: no cover - script entry point
    sys.exit(main())
