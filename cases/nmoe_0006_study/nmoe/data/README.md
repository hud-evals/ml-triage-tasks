# nmoe.data

`nmoe.data` provides two things:

1) **HF → shards**: deterministic preprocessing into `.npy` token shards + `.idx` boundaries + `manifest.json`  
2) **Loader**: deterministic, resumable training loader (direct shards or mixture plan)

This repo is container-first; run prep inside the training image so optional deps (HF datasets, pyarrow, etc.) are present.

## Golden paths

### 1) Prep a single HuggingFace dataset

```bash
python -m nmoe.data.cli prep \
  --source hf \
  --dataset HuggingFaceFW/fineweb-edu \
  --split train \
  --output /data/fineweb_edu \
  --name fineweb_edu \
  --max-tokens-total 100M
```

Outputs:
- `/data/fineweb_edu/manifest.json`
- `/data/fineweb_edu/shard_*/.../*.npy` + matching `.idx`

### 2) Prep all HuggingFace sources referenced by a mixture+flow

```bash
python -m nmoe.data.cli prep-mixture \
  --config configs/mixtures/olmo3_1025.toml \
  --flow-profiles configs/flow_profiles.toml \
  --flow dev \
  --stage pretrain
```

Default output layout:
`/data/flows/<flow>/<stage>/<source>/<split>/manifest.json`

### 3) Verify a dataset

```bash
python -m nmoe.data.cli verify /data/fineweb_edu/manifest.json --checksums
python -m nmoe.data.cli info /data/fineweb_edu/manifest.json
```

## Training configs

Two supported ways to point training at data:

- **Direct**: set `data_path=/data/fineweb_edu` and omit `flow_mode`.
- **Mixture plan**: set `flow_mode`, `mixture_toml`, `flow_profiles_toml` and ensure the corresponding flow outputs exist under `data_root/flows/<flow_mode>/...`.

## Contract (shards)

- Shards are `uint32` token arrays (1D) with EOS appended per document.
- `.idx` stores document boundaries (start,end offsets into the shard).
- `manifest.json` is the single source of truth for dataset metadata and shard inventory.

