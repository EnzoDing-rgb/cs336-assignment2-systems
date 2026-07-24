"""Suite entry: `uv run python -m cs336_systems.e2e_timing`"""

from __future__ import annotations

import sys
import time


def _log(msg: str) -> None:
    print(f"[suite {time.strftime('%H:%M:%S')}] {msg}", flush=True)


_log("process started")
_log(f"python={sys.executable}")
_log("importing e2e_timing (torch / CUDA init can take 30–120s, please wait) …")

from cs336_systems.e2e_timing.e2e import run_suite  # noqa: E402

_log("imports done; entering run_suite()")
run_suite()
