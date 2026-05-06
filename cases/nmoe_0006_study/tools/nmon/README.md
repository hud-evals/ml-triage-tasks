# nmon

Terminal-first monitoring for NMoE training runs.

Design goals:
- Single-run-first (fast follow: multi-run compare).
- Rendering via `lipgloss` widgets (nvitop-style tables + compact charts).
- Reads the existing NMoE metrics/artifacts:
  - Parquet: `/data/metrics/{run_id}/step_*.parquet`
  - SQLite: `/data/experiments.db` (read-only snapshot)

## Build

```bash
cd tools/nmon
go build -o nmon ./cmd/nmon
```

## Run (local)

```bash
cd /workspace/nmoe/tools/nmon
go run ./cmd/nmon
```

## Run (opt-in local + k8s remote data)

Runs `nmon` locally, but reads `/data/metrics` and `/data/experiments.db` from a cluster pod via the Kubernetes API (exec).

```bash
cd /workspace/nmoe/tools/nmon

go run ./cmd/nmon --k8s \
  --k8s-namespace=default \
  --k8s-selector='app=nmoe,stage=debug' \
  --metrics-dir=/data/metrics \
  --experiments-db=/data/experiments.db
```

Notes:
- If neither `--k8s-pod` nor `--k8s-selector` is set, `nmon` picks the first Running pod in the namespace.
- The target container must have `python3` with the `duckdb` module available (for metrics queries). SQLite queries use Python's `sqlite3`.
