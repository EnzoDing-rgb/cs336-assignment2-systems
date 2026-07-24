"""CLI: sweep scaled dot-product attention benchmark + figures."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import torch

from cs336_systems.attention_operator.benchmark import BenchmarkResult
from cs336_systems.attention_operator.plots import make_figures
from cs336_systems.attention_operator.sweep import DEFAULT_D_VALUES, DEFAULT_S_VALUES, run_sweep

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_ROOT = REPO_ROOT / "artifacts" / "attention_operator"
RESULTS_PATH = ARTIFACTS_ROOT / "results.json"
REPORT_PATH = REPO_ROOT / "reports" / "benchmarking-scaled-dot-product-attention.md"
FIGURES_DIR = REPO_ROOT / "reports" / "figures"


def main() -> None:
    p = argparse.ArgumentParser(description="Scaled dot-product attention operator benchmark")
    p.add_argument("--skip-run", action="store_true", help="Rebuild figures/report from results.json")
    p.add_argument("--no-report", action="store_true", help="Skip report regeneration")
    args = p.parse_args()

    if args.skip_run:
        manifest = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
        results = [BenchmarkResult(**r) for r in manifest["results"]]
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required")
        ARTIFACTS_ROOT.mkdir(parents=True, exist_ok=True)
        device = torch.device("cuda")
        print("── Attention operator sweep ──", flush=True)
        results = run_sweep(device=device)
        RESULTS_PATH.write_text(
            json.dumps(
                {
                    "results": [r.to_dict() for r in results],
                    "d_values": list(DEFAULT_D_VALUES),
                    "s_values": list(DEFAULT_S_VALUES),
                    "generated_utc": datetime.now(timezone.utc).isoformat(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Wrote {RESULTS_PATH}", flush=True)

    figs = make_figures([r.to_dict() for r in results])
    print("Figures:", ", ".join(str(p) for p in figs.values()), flush=True)

    if not args.no_report:
        write_report(results)
        print(f"Wrote {REPORT_PATH}", flush=True)


def write_report(results: list[BenchmarkResult]) -> None:
    from cs336_systems.attention_operator.report import render_report

    REPORT_PATH.write_text(render_report(results), encoding="utf-8")


if __name__ == "__main__":
    main()
