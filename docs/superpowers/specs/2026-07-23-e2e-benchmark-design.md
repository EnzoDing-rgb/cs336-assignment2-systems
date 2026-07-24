# End-to-End Benchmarking Design

**Date:** 2026-07-23  
**Scope:** Assignment 2 `benchmarking_script` parts (a)(b)(c)

## Layout

```
cs336_systems/benchmark/e2e.py       # all logic
cs336_systems/benchmark/__main__.py  # suite entry
```

- Single run: `uv run python -m cs336_systems.e2e_timing.e2e --model-size … --mode …`
- Suite (b+c): `uv run python -m cs336_systems.e2e_timing`
- No `scripts/` directory.
- Do not modify `cs336-basics`.

## Locked decisions

- Modes: `forward` | `forward_backward` | `train` | `timed_train`
- `timed_train` segments: forward / loss / backward / optimizer
- Suite matrix: 5 sizes × timed_train × warmup=5; then same × warmup∈{0,1,2}; reuse wu=5
- Deliverables: `reports/end2end-benchmark.md` + 3 PNGs under `reports/figures/`
  (mean times, segment std, warmup ablation). Tables use mean±std.
- Artifacts under `artifacts/e2e_benchmark/` (gitignored)
