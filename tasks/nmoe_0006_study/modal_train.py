"""Modal launcher: generate the four 0006 blog_artifacts on 8xB200.

The public nmoe repo does not commit run outputs — blog_artifacts/ is the
output of running four specific torchrun commands on an 8-GPU Blackwell
node. This script reproduces those commands on Modal and lands the
experiments.db + step_*.parquet bundle on a shared Volume that we then
download into `cases/nmoe_0006_study/`.

Image: xjdr/nmoe_train:latest — prebuilt by the repo's docker/Dockerfile.*
chain, targets sm_100a (B200). Includes torch nightly + Triton + vendored
FlashAttention CuTe + FlashMLA + Quack + built CUDA kernels.

Source-of-truth code visible to the *agent* (in the Docker env) is a fresh
checkout at commit 970a146. This Modal run uses the training image's own
code to produce artifacts; a separate script pulls the 970a146 checkout
into the case bundle for the agent.

Usage:
  # 1. One-time dataset prep (tokenize karpathy/fineweb-edu-100b-shuffle)
  modal run modal_train.py::prep_data

  # 2. Smoke the pipeline: tiny step-count main run to validate end-to-end
  modal run modal_train.py --action smoke

  # 3. Full four-run campaign (serial, one GPU pod at a time)
  modal run modal_train.py --action all

  # 4. Single run by label
  modal run modal_train.py --action one --label 0006_super1024_r1

  # 5. Pack artifacts into a single tar.gz on the Volume
  modal run modal_train.py::pack_bundle

  # 6. Download locally
  modal volume get nmoe-0006-data /bundle.tar.gz ./cases/nmoe_0006_study/bundle.tar.gz

Cost guide (Modal on-demand, approximate):
  B200 @ ~$6.25/GPU-hr × 8 = ~$50/hr per pod
  - prep_data:              ~30-45 min  → ~$25-40
  - 0006_super4096_clean:   12k steps,  ~6-8 hr → ~$300-400
  - 0006_super4096_auxclean: 2k steps,  ~1.0 hr → ~$50
  - 0006_super1024_r1:      2k steps,   ~0.8 hr → ~$40
  - 0006_super2048_long_r1: 2k steps,   ~1.0 hr → ~$50
  Total ~$465-580 wall, one pod serial; parallel cuts wall ~4x but not $.
"""
from __future__ import annotations

import modal

APP_NAME = "nmoe-0006-training"
TRAIN_IMAGE = "xjdr/nmoe_train:latest"
VOLUME_NAME = "nmoe-0006-data"
NMOE_COMMIT = "970a146433f9c649d09ddab36f675974f53dd905"

# The nmoe_train image has its venv at /workspace/nmoe/.venv and expects
# PYTHONPATH to include vendored third_party/*. Replicate the repo's
# runtime env vars here.
image = (
    modal.Image.from_registry(TRAIN_IMAGE, add_python="3.12")
    # hf_transfer gives multi-threaded per-file downloads (5-10x on large
    # parquets). The base image does not include it, and without it
    # hf_hub_download is single-threaded and ~30s/shard on fineweb-edu.
    .run_commands(
        "/root/.local/bin/uv pip install --python /workspace/nmoe/.venv/bin/python hf_transfer"
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "HF_HOME": "/data/hf_cache",
        "PATH": "/workspace/nmoe/.venv/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin",
        "PYTHONPATH": (
            "/workspace/nmoe"
            ":/workspace/nmoe/nmoe/csrc"
            ":/workspace/nmoe/third_party/flash_attn"
            ":/workspace/nmoe/third_party/quack"
            ":/workspace/nmoe/triton/python"
        ),
        "CUDA_HOME": "/usr/local/cuda",
        "LD_LIBRARY_PATH": "/usr/local/cuda/lib64",
        "TORCH_CUDA_ARCH_LIST": "10.0a",
        "NCCL_DEBUG": "WARN",
        "NCCL_IB_DISABLE": "1",
        "NCCL_P2P_DISABLE": "0",
        "NCCL_P2P_LEVEL": "NVL",
        "NCCL_SHM_DISABLE": "0",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "DATA_DIR": "/data",
    })
)

data_vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

app = modal.App(APP_NAME, image=image)


# The four commands transcribed from repro/0006.receipts.json. Each entry
# is the extra-args suffix after the common torchrun + config prefix.
RUNS: list[dict] = [
    {
        "label": "0006_super4096_rerun_r2_clean",
        "args": ["--steps=12000"],
    },
    {
        "label": "0006_super4096_auxclean_pair",
        "args": ["--steps=2048",
                 "--router_bias_update_rate=0.0",
                 "--aux_loss_alpha=0.0001"],
    },
    {
        "label": "0006_super1024_r1",
        "args": ["--steps=2048", "--n_routed_experts=1024"],
    },
    {
        "label": "0006_super2048_long_r1",
        "args": ["--steps=2048", "--n_routed_experts=2048"],
    },
]

# Nearby runs the researcher has in the folder but that should NOT be in
# the r2_clean comparison set. Scope-selection gets meaningful only when
# the bundle includes real plausible distractors.
#
#  - `0006_super4096_baseline_r1`: short first-pass baseline that was
#    superseded by the r2_clean rerun (shorter, earlier, effectively the
#    same config but predates the "corrected stack" run).
#  - `0006_super4096_bias0_aux1e4_long_r1`: the aux-cleaned variant run
#    out to 8k steps. Referenced in the real 0006.receipts.json as
#    aux_long_root but NOT used by the final falsification-pair analysis,
#    which uses the matched 2k-step pair instead. Including it here
#    forces the scope-selection axis to decide between "matched 2k pair"
#    and this longer variant.
STALE_RUNS: list[dict] = [
    # 0006 receipts say the canonical bundle uses bf16 and was built
    # AFTER the researcher switched off an older fp8 baseline + its
    # probe-heavy stack. We stand in for that "numerical-stability
    # supersession" by running earlier baselines at --dtype=fp8 —
    # an agent reading config_json in experiments.db will see the
    # dtype mismatch and treat these as pre-correction runs.
    {
        "label": "0006_super4096_baseline_r0_jan",
        "args": ["--steps=1024", "--dtype=fp8"],
    },
    {
        "label": "0006_super4096_baseline_r0_mar03",
        "args": ["--steps=2048", "--dtype=fp8"],
    },
    # An aux-loss-too-strong attempt that got abandoned — aux=1e-3 is
    # 10x the canonical aux=1e-4, short probe before giving up.
    {
        "label": "0006_super4096_aux1e3_bias0",
        "args": ["--steps=1024",
                 "--router_bias_update_rate=0.0",
                 "--aux_loss_alpha=0.001"],
    },
    # Tried removing the shared-expert escape valve (n_shared_experts=0
    # instead of the canonical 1). Collapse got worse, got abandoned as
    # a dead-end exploration. Integer-valued — safe through the image's
    # CLI parser (floats like --lr_dense stay strings and crash AdamW
    # init, so use an int-shaped knob).
    {
        "label": "0006_super4096_no_shared_r1",
        "args": ["--steps=1024",
                 "--n_shared_experts=0"],
    },
    # Shorter bf16 first-pass baseline predating the r2_clean rerun.
    # Same dtype, same config, just short and early — stale by recency.
    {
        "label": "0006_super4096_baseline_r1",
        "args": ["--steps=1024"],
    },
    # The aux-cleaned variant run out to 8k steps. Referenced in
    # repro/0006.receipts.json as aux_long_root but NOT used by the
    # final falsification-pair analysis (which uses the matched
    # 2k-step pair). Forces scope-selection to prefer the matched
    # pair over this longer variant.
    {
        "label": "0006_super4096_bias0_aux1e4_long_r1",
        "args": ["--steps=8192",
                 "--router_bias_update_rate=0.0",
                 "--aux_loss_alpha=0.0001"],
    },
]


def _venv_python() -> str:
    return "/workspace/nmoe/.venv/bin/python"


@app.function(
    cpu=16.0,
    memory=32 * 1024,
    volumes={"/data": data_vol},
    secrets=[modal.Secret.from_name("hf-secret")],
    timeout=3 * 3600,
)
def prefetch_shards(num_train_shards: int = 210) -> dict:
    """Parallel-download the parquet shards we'll need into /data/hf_cache.

    The nmoe HfHubParquetSource iterates files serially in one process, which
    means serial hf_hub_download calls (~30s/shard). By pre-populating the HF
    cache with `max_workers=64` we turn those calls into cache hits.

    With --max-tokens-total=10B and ~54M tokens/shard, the train iterator
    reads roughly the first ~190 parquets before early-exit; prefetch 210 for
    safety margin. The val split uses just `shard_01822.parquet`.
    """
    import os, time
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from huggingface_hub import hf_hub_download, list_repo_files

    repo = "karpathy/fineweb-edu-100b-shuffle"
    all_parquets = sorted(
        f for f in list_repo_files(repo_id=repo, repo_type="dataset")
        if f.endswith(".parquet")
    )
    print(f"[prefetch] repo has {len(all_parquets)} parquets", flush=True)
    # Train: first N train shards (train = all but last per nanochat convention)
    train_files = all_parquets[:-1][:num_train_shards]
    # Val: the specific shard nmoe's ensure_speedrun_data uses
    val_files = ["shard_01822.parquet"]
    to_fetch = train_files + val_files
    print(f"[prefetch] fetching {len(to_fetch)} files (train={len(train_files)}, val={len(val_files)})",
          flush=True)

    t0 = time.time()
    bytes_fetched = 0

    def _one(fname: str) -> int:
        path = hf_hub_download(repo_id=repo, repo_type="dataset", filename=fname)
        return os.path.getsize(path)

    with ThreadPoolExecutor(max_workers=64) as ex:
        futures = {ex.submit(_one, f): f for f in to_fetch}
        for i, fut in enumerate(as_completed(futures)):
            fname = futures[fut]
            try:
                sz = fut.result()
                bytes_fetched += sz
                if (i + 1) % 25 == 0 or i == len(to_fetch) - 1:
                    dt = time.time() - t0
                    gb = bytes_fetched / (1024**3)
                    print(f"[prefetch] {i+1}/{len(to_fetch)} done "
                          f"({gb:.1f} GiB in {dt:.0f}s = {gb*1024/dt:.0f} MiB/s)",
                          flush=True)
            except Exception as e:
                print(f"[prefetch] ERROR on {fname}: {e}", flush=True)
                raise

    data_vol.commit()
    return {
        "elapsed_s": time.time() - t0,
        "files": len(to_fetch),
        "bytes": bytes_fetched,
        "gib": round(bytes_fetched / (1024**3), 2),
    }


HF_DATASET = "karpathy/fineweb-edu-100b-shuffle"
VAL_DATA_FILE = "shard_01822.parquet"
TRAIN_TOKEN_BUDGET = 10_000_000_000   # 10B
VAL_TOKEN_BUDGET   = 10_485_760        # 10M
# HF cache path shape: hub/datasets--<org>--<name>/snapshots/<commit>/...
CACHE_REPO_DIR = "hub/datasets--karpathy--fineweb-edu-100b-shuffle"


def _snapshot_root(hf_home: str) -> str:
    """Return the snapshots/<commit_hash>/ dir holding parquets in an HF cache.
    Assumes exactly one snapshot (the one prefetch_shards wrote)."""
    import os
    snaps_dir = os.path.join(hf_home, CACHE_REPO_DIR, "snapshots")
    ids = [d for d in os.listdir(snaps_dir) if os.path.isdir(os.path.join(snaps_dir, d))]
    if not ids:
        raise FileNotFoundError(f"no snapshot under {snaps_dir}")
    return os.path.join(snaps_dir, ids[0])


def _list_cached_parquets(hf_home: str = "/data/hf_cache") -> list[str]:
    import os
    root = _snapshot_root(hf_home)
    return sorted(os.path.join(root, f)
                  for f in os.listdir(root)
                  if f.endswith(".parquet"))


@app.function(
    cpu=16.0,
    memory=32 * 1024,
    volumes={"/data": data_vol},
    timeout=4 * 3600,
)
def tokenize_stripe(worker_index: int, num_workers: int, split: str) -> dict:
    """Tokenize one worker's stripe of prefetched parquets.

    Strategy:
      1. Parallel-copy just this worker's stripe of parquets from Volume to
         local NVMe (pyarrow + Modal Volume FUSE = errno 22 under load).
      2. Invoke `nmoe.data.cli prep --source parquet --paths ...` on locals.
         --source parquet uses ArrowSource, which reads pyarrow row-groups
         serially in the main process — that's fine because each container
         only owns 1/num_workers of the files, so K containers give K-fold
         pipeline parallelism across the campaign.
      3. Commit outputs under /data/speedrun/<split>_parts/worker_{i}/.
         The merge step writes a canonical manifest at /data/speedrun/<split>/.
    """
    import os, subprocess, time, shutil, json
    from concurrent.futures import ThreadPoolExecutor

    assert split in ("train", "val")
    os.chdir("/workspace/nmoe")
    t0 = time.time()

    vol_parquets = _list_cached_parquets("/data/hf_cache")
    val_file = next((p for p in vol_parquets
                     if os.path.basename(p) == VAL_DATA_FILE), None)

    if split == "train":
        # nanochat convention: train = all but last parquet
        train_pool = [p for p in vol_parquets if p != val_file]
        # Stripe by modular index (matches HfHubParquetSource's sharding rule)
        my_files = train_pool[worker_index::num_workers]
        # Trim to cover just this worker's share of the budget, with margin.
        # Each parquet shard ≈ 55M tokens (empirically on fineweb-edu).
        target = TRAIN_TOKEN_BUDGET // num_workers
        needed = target // 50_000_000 + 3
        my_files = my_files[:min(len(my_files), needed)]
        out_dir = f"/data/speedrun/train_parts/worker_{worker_index:02d}"
        my_budget = target
        name = "speedrun_train"
        num_out_shards = max(1, 64 // num_workers)
    else:  # val — tiny, one shard, only worker 0
        if worker_index != 0:
            return {"worker": worker_index, "skipped": "val handled by worker_0"}
        if val_file is None:
            raise FileNotFoundError(f"{VAL_DATA_FILE} not in cache")
        my_files = [val_file]
        out_dir = "/data/speedrun/val"
        my_budget = VAL_TOKEN_BUDGET
        name = "speedrun_val"
        num_out_shards = 8

    # Wipe any partial state from a prior failed run (manifest.json is
    # written only on success, so its absence means this dir is garbage).
    if os.path.isdir(out_dir) and not os.path.exists(os.path.join(out_dir, "manifest.json")):
        print(f"[w{worker_index}] wiping incomplete {out_dir}", flush=True)
        shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(os.path.join(out_dir, "manifest.json")):
        return {"worker": worker_index, "skipped": "manifest already present",
                "output": out_dir}

    # Parallel threaded copy Volume -> local NVMe (pyarrow is happy on a
    # real filesystem). shutil.copy2 follows symlinks → writes blob content.
    local_dir = f"/root/local_parquets_{split}_{worker_index:02d}"
    os.makedirs(local_dir, exist_ok=True)

    def _copy(src: str) -> str:
        dst = os.path.join(local_dir, os.path.basename(src))
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
        return dst
    t_copy = time.time()
    with ThreadPoolExecutor(max_workers=16) as ex:
        local_paths = list(ex.map(_copy, my_files))
    total_bytes = sum(os.path.getsize(p) for p in local_paths)
    print(f"[w{worker_index}] copied {len(local_paths)} files "
          f"({total_bytes/1e9:.1f} GB) in {time.time()-t_copy:.1f}s", flush=True)

    cmd = [
        _venv_python(), "-m", "nmoe.data.cli", "prep",
        "--source", "parquet",
        "--paths", *local_paths,
        "--output", out_dir,
        "--name", name,
        "--tokenizer", "gpt2",
        "--vocab-size", "50304",
        "--eos-token-id", "50256",
        "--max-tokens-total", str(my_budget),
        "--num-shards", str(num_out_shards),
        "--workers", "12",
        "--batch-size", "5000",
        "--parallel",
    ]
    t_tok = time.time()
    subprocess.check_call(cmd, cwd="/workspace/nmoe")
    print(f"[w{worker_index}] tokenize done in {time.time()-t_tok:.1f}s", flush=True)

    # Optionally free local disk for the next run
    shutil.rmtree(local_dir, ignore_errors=True)

    data_vol.commit()
    # Read back the per-worker manifest for the return payload
    mpath = os.path.join(out_dir, "manifest.json")
    m = json.loads(open(mpath).read()) if os.path.exists(mpath) else {}
    return {
        "worker": worker_index,
        "output": out_dir,
        "tokens": int(m.get("total_tokens", 0)),
        "documents": int(m.get("total_documents", 0)),
        "num_shards": int(m.get("num_shards", 0)),
        "elapsed_s": round(time.time() - t0, 1),
    }


@app.function(
    cpu=4.0,
    memory=8 * 1024,
    volumes={"/data": data_vol},
    timeout=30 * 60,
)
def merge_parts(num_workers: int, split: str = "train") -> dict:
    """Combine K per-worker manifests into one canonical manifest at
    /data/speedrun/<split>/manifest.json.

    We symlink each worker's subtree under the merged dir (loader globs
    '**/*.npy' recursively) and set source_info.source to the canonical
    HF dataset name so ensure_speedrun_data's _manifest_ok check passes.
    """
    import os, json, shutil
    from datetime import datetime, timezone
    from pathlib import Path

    assert split in ("train", "val")
    parts_dir = Path(f"/data/speedrun/{split}_parts")
    out_dir = Path(f"/data/speedrun/{split}")
    out_dir.mkdir(parents=True, exist_ok=True)

    total_tokens = 0
    total_docs = 0
    all_shards: list[dict] = []
    template: dict | None = None

    for i in range(num_workers):
        wdir = parts_dir / f"worker_{i:02d}"
        mpath = wdir / "manifest.json"
        if not mpath.exists():
            print(f"[merge] WARN missing {mpath}", flush=True)
            continue
        m = json.loads(mpath.read_text())
        template = template or m
        total_tokens += int(m["total_tokens"])
        total_docs += int(m["total_documents"])
        for s in m["shards"]:
            s_copy = dict(s)
            s_copy["path"] = f"worker_{i:02d}/{s['path']}"
            s_copy["index_path"] = f"worker_{i:02d}/{s['index_path']}"
            all_shards.append(s_copy)
        # Expose this worker's subtree under the merged dir so a
        # recursive '**/*.npy' glob finds its shards.
        link = out_dir / f"worker_{i:02d}"
        if link.is_symlink() or link.exists():
            if link.is_symlink():
                link.unlink()
            else:
                shutil.rmtree(link)
        link.symlink_to(wdir.resolve())

    if template is None:
        raise RuntimeError(f"no per-worker manifests under {parts_dir}")

    merged = dict(template)
    merged["total_tokens"] = total_tokens
    merged["total_documents"] = total_docs
    merged["num_shards"] = len(all_shards)
    merged["shards"] = all_shards
    merged["source_info"] = {"source": HF_DATASET}
    merged["created_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

    (out_dir / "manifest.json").write_text(json.dumps(merged, indent=2))
    data_vol.commit()
    return {
        "split": split,
        "workers": num_workers,
        "total_tokens": total_tokens,
        "total_documents": total_docs,
        "num_shards": len(all_shards),
        "output": str(out_dir),
    }


@app.function(
    gpu="B200:8",
    volumes={"/data": data_vol},
    timeout=10 * 3600,
)
def run_training(label: str, extra_args: list[str]) -> dict:
    """Run one 0006-family training command, then repackage its outputs
    under /data/blog_artifacts/<label>/ so the shape matches the receipt
    manifest the grader references."""
    import os, subprocess, shutil, time, json

    os.chdir("/workspace/nmoe")

    # Wipe any prior bundle dir for this label so we don't end up with
    # stale run_id subdirs alongside fresh ones from this re-run.
    bundle_pre = f"/data/blog_artifacts/{label}"
    if os.path.isdir(bundle_pre):
        print(f"[{label}] wiping prior {bundle_pre}", flush=True)
        shutil.rmtree(bundle_pre, ignore_errors=True)

    metrics_root = "/data/metrics"
    os.makedirs(metrics_root, exist_ok=True)
    before = set(os.listdir(metrics_root))

    # NOTE: --collect_update_stats=false is in the 0006 receipts but absent
    # from xjdr/nmoe_train:latest (image is 2026-03-08, receipts-commit 970a146
    # is 2026-03-15). That kwarg disables a later-added telemetry collector
    # that doesn't exist in the image, so dropping it is behaviorally a no-op.
    cmd = [
        "torchrun", "--nproc_per_node=8", "-m", "nmoe.train",
        "configs/speedrun/small_moe_super.toml",
        "--dtype=bf16",
    ] + list(extra_args)
    print(f"[{label}] launching: {' '.join(cmd)}", flush=True)
    t0 = time.time()
    subprocess.check_call(cmd, cwd="/workspace/nmoe")
    elapsed = time.time() - t0
    print(f"[{label}] training finished in {elapsed/3600:.2f} hr", flush=True)

    after = set(os.listdir(metrics_root))
    new_runs = sorted(after - before)
    if not new_runs:
        raise RuntimeError(f"[{label}] no new run_id directory under {metrics_root}")
    run_id = new_runs[-1]

    bundle = f"/data/blog_artifacts/{label}"
    os.makedirs(f"{bundle}/metrics", exist_ok=True)
    # Hard-copy the metrics — the Volume layer will dedupe.
    shutil.copytree(
        f"{metrics_root}/{run_id}",
        f"{bundle}/metrics/{run_id}",
        dirs_exist_ok=True,
    )
    # The speedrun config points experiments_db at a container-local path
    # (configs/speedrun/small_moe_super.toml: experiments_db="/tmp/experiments_super.db")
    # — NOT /data/experiments.db. The authors do this deliberately so
    # concurrent pods don't fight over one SQLite file on a shared mount.
    for candidate_db in ("/tmp/experiments_super.db", "/data/experiments.db"):
        if os.path.exists(candidate_db):
            shutil.copy(candidate_db, f"{bundle}/experiments.db")
            print(f"[{label}] experiments.db <- {candidate_db}", flush=True)
            break
    else:
        print(f"[{label}] WARN: no experiments.db found on any known path", flush=True)
    (os.path.join(bundle, "RUN_META.json"))
    with open(os.path.join(bundle, "RUN_META.json"), "w") as f:
        json.dump({
            "label": label,
            "run_id": run_id,
            "cmd": cmd,
            "elapsed_s": elapsed,
            "commit_used": "xjdr/nmoe_train:latest (image)",
            "nmoe_agent_visible_commit": NMOE_COMMIT,
        }, f, indent=2)

    data_vol.commit()
    return {"label": label, "run_id": run_id, "elapsed_s": elapsed}


@app.function(volumes={"/data": data_vol}, timeout=30 * 60)
def pack_bundle() -> dict:
    """Tar up /data/blog_artifacts into /data/bundle.tar.gz for download."""
    import subprocess, os
    out = "/data/bundle.tar.gz"
    src = "/data/blog_artifacts"
    if not os.path.isdir(src):
        raise RuntimeError(f"no {src} to pack")
    subprocess.check_call(["tar", "-czf", out, "-C", "/data", "blog_artifacts"])
    size_mb = os.path.getsize(out) / (1024 * 1024)
    data_vol.commit()
    return {"bundle": out, "size_mb": round(size_mb, 1)}


@app.function(gpu="B200:8", volumes={"/data": data_vol}, timeout=2 * 3600)
def smoke() -> dict:
    """Validate the full pipeline with a 64-step main-config run before
    committing to 12k steps. Writes to /data/blog_artifacts/_smoke/."""
    return run_training.local(
        label="_smoke",
        extra_args=["--steps=64"],
    )


@app.local_entrypoint()
def main(
    action: str = "all",
    label: str = "",
    serial: bool = True,
) -> None:
    """Local entrypoint.

    action:
      prep  - tokenize the speedrun dataset
      smoke - 64-step validation run
      one   - run a single entry from RUNS by --label
      all   - prep (if needed) then all four RUNS
    """
    if action == "prefetch":
        print(prefetch_shards.remote())
        return

    if action == "tokenize":
        # Fan out K parallel workers (train), 1 for val, then merge.
        K = 8
        print(f"=== tokenize train stripes (K={K}) ===")
        calls = [tokenize_stripe.spawn(i, K, "train") for i in range(K)]
        for fc in calls:
            print(fc.get())
        print("=== tokenize val ===")
        print(tokenize_stripe.remote(0, 1, "val"))
        print("=== merge train parts ===")
        print(merge_parts.remote(K, "train"))
        return

    if action == "prep":
        print("=== prefetch ===")
        print(prefetch_shards.remote())
        print("=== tokenize train stripes ===")
        K = 8
        calls = [tokenize_stripe.spawn(i, K, "train") for i in range(K)]
        for fc in calls:
            print(fc.get())
        print("=== tokenize val ===")
        print(tokenize_stripe.remote(0, 1, "val"))
        print("=== merge train parts ===")
        print(merge_parts.remote(K, "train"))
        return

    if action == "smoke":
        print(smoke.remote())
        return

    if action == "one":
        assert label, "--label required for action=one"
        all_runs = RUNS + STALE_RUNS
        rn = next((r for r in all_runs if r["label"] == label), None)
        if rn is None:
            raise SystemExit(f"unknown label {label!r}; known: "
                             + ", ".join(r["label"] for r in all_runs))
        print(run_training.remote(rn["label"], rn["args"]))
        return

    if action == "pack":
        print(pack_bundle.remote())
        return

    if action == "stale":
        # Launch just the stale distractor runs.
        calls = [run_training.spawn(r["label"], r["args"]) for r in STALE_RUNS]
        for fc in calls:
            print(fc.get())
        return

    if action == "all":
        if serial:
            for rn in RUNS + STALE_RUNS:
                print(f"=== launching {rn['label']} ===")
                print(run_training.remote(rn["label"], rn["args"]))
        else:
            calls = [run_training.spawn(r["label"], r["args"])
                     for r in RUNS + STALE_RUNS]
            for fc in calls:
                print(fc.get())
        print("=== packing bundle ===")
        print(pack_bundle.remote())
        return

    raise SystemExit(f"unknown action {action!r}")
