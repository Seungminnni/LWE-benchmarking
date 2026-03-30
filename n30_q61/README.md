# n30_q61 Successful Pipeline

This document records the exact end-to-end pipeline that successfully solved the `n30_q61` LWE run inside this repository.

All commands below are intended to be run from the repository root:

`/home/yu_mcc/LWE-benchmarking`

## Directory layout

After the successful run, the outputs live under:

- `n30_q61/fresh`
- `n30_q61/preproc`
- `n30_q61/ab`
- `n30_q61/salsa`

Key files:

- `n30_q61/fresh/origA_n30_logq6.npy`
- `n30_q61/preproc/main/data_0.prefix`
- `n30_q61/preproc/main/data_1.prefix`
- `n30_q61/preproc/main/data_2.prefix`
- `n30_q61/preproc/main/data_3.prefix`
- `n30_q61/preproc/data.prefix`
- `n30_q61/ab/train_A.npy`
- `n30_q61/ab/test_A.npy`
- `n30_q61/ab/binary_secrets_h3_3/secret.npy`
- `n30_q61/salsa/main/train.log`

## Step 1: Generate fresh A

```bash
python3 src/generate/make_orig_A.py \
  --N 30 --Q 61 \
  --num_rows 40000 \
  --representation mod_q \
  --dump_path ./n30_q61/fresh
```

Expected output:

- `n30_q61/fresh/origA_n30_logq6.npy`

Observed shape from the successful run:

- `origA_n30_logq6.npy`: `(40000, 30)`

## Step 2: Run preprocessing

```bash
conda run -n lattice_env python src/generate/preprocess.py \
  --N 30 --Q 61 \
  --dump_path ./n30_q61 \
  --exp_name preproc \
  --exp_id main \
  --num_workers 4 \
  --reload_data ./n30_q61/fresh/origA_n30_logq6.npy \
  --thresholds '0.85,0.95,0.98' \
  --lll_penalty 10 \
  --lll_delta 0.97 \
  --bkz_block_size 10 \
  --max_prefix_lines 500000
```

Important outputs:

- Raw worker prefixes are written to `n30_q61/preproc/main/data_*.prefix`
- Progress is tracked in `n30_q61/preproc/main/prefix_progress.txt`

Observed final progress from the successful run:

```text
500006 19231
```

This means the preprocessing target was reached successfully.

## Step 3: Build A/b dataset

```bash
conda run -n lattice_env python src/generate/generate_A_b.py \
  --processed_dump_path ./n30_q61/preproc \
  --orig_A_path ./n30_q61/fresh/origA_n30_logq6.npy \
  --dump_path ./n30_q61/ab \
  --N 30 \
  --min_hamming 3 --max_hamming 3 \
  --secret_type binary \
  --num_secret_seeds 5 \
  --actions secrets
```

Important outputs:

- `n30_q61/ab/orig_A.npy`
- `n30_q61/ab/orig_b.npy`
- `n30_q61/ab/train_A.npy`
- `n30_q61/ab/test_A.npy`
- `n30_q61/ab/binary_secrets_h3_3/secret.npy`
- `n30_q61/ab/binary_secrets_h3_3/train_b_3_0.npy`
- `n30_q61/ab/binary_secrets_h3_3/test_b_3_0.npy`

Observed shapes from the successful run:

- `orig_A.npy`: `(40000, 30)`
- `train_A.npy`: `(56446, 30)`
- `test_A.npy`: `(10000, 30)`
- `secret.npy`: `(30, 5)`
- `train_b_3_0.npy`: `(56446,)`
- `test_b_3_0.npy`: `(10000,)`

## Step 4: Train SALSA and recover the secret

```bash
MPLCONFIGDIR=/tmp/matplotlib conda run -n lattice_env python src/salsa/train_and_recover.py \
  --data_path ./n30_q61/ab/binary_secrets_h3_3 \
  --task lwe \
  --hamming 3 \
  --secret_seed 0 \
  --dump_path ./n30_q61 \
  --exp_name salsa \
  --exp_id main \
  --train_batch_size 8 \
  --val_batch_size 100 \
  --enc_emb_dim 1024 \
  --n_enc_layers 1 \
  --n_enc_heads 4 \
  --distinguisher_size 128 \
  --workers 0 \
  --clip_grad_norm 5.0 \
  --optimizer 'adam_warmup,lr=0.00001,warmup_updates=8000,weight_decay=0.99' \
  --compile 0 \
  --dtype float32
```

Important outputs:

- `n30_q61/salsa/main/train.log`
- `n30_q61/salsa/main/checkpoint.pth`

Observed success signal from the successful run:

- Secret recovery succeeded at about `0:01:00`
- Log shows `recover/matched: true`
- Log shows `Recovered secret!`

## Sanity checks

If the run is healthy, the following should all be true:

- `n30_q61/fresh/origA_n30_logq6.npy` exists
- `n30_q61/preproc/main/data_0.prefix` through `data_3.prefix` exist
- `n30_q61/preproc/main/prefix_progress.txt` is near the target
- `n30_q61/preproc/data.prefix` exists after `generate_A_b.py`
- `n30_q61/ab/train_A.npy` is non-empty
- `n30_q61/salsa/main/train.log` eventually contains `Recovered secret!`

## Important gotchas

### 1. Use the parent preprocess directory in `generate_A_b.py`

Use:

```text
--processed_dump_path ./n30_q61/preproc
```

Do not use:

```text
--processed_dump_path ./n30_q61/preproc/main
```

The current code looks for `data_*.prefix` under subdirectories when it auto-merges prefix files. Passing the parent directory avoids empty or misleading `data.prefix` states.

### 2. Keep `orig_A` consistent across preprocessing and A/b generation

These two must refer to the same file:

- `preprocess.py --reload_data`
- `generate_A_b.py --orig_A_path`

For this successful run, both pointed to:

```text
./n30_q61/fresh/origA_n30_logq6.npy
```

### 3. Avoid `--actions describe` for now

In the current codebase, `generate_A_b.py --actions describe` can hang after printing `Cruel bits:` because the brute-force recommendation loop does not terminate for some values of `n_cruel_bits`.

Use:

```text
--actions secrets
```

instead of:

```text
--actions secrets describe
```

until that bug is fixed.

### 4. `--epochs` is not the main stopping condition in SALSA

`train_and_recover.py` currently stops based on recovery success or time logic, not simply because the nominal epoch count was reached. If training appears to continue longer than expected, check the recovery condition in the log.

## Minimal rerun checklist

1. Create `orig_A` in `n30_q61/fresh`.
2. Run preprocess into `n30_q61/preproc/main`.
3. Build the dataset into `n30_q61/ab` using `--processed_dump_path ./n30_q61/preproc`.
4. Train SALSA from `n30_q61/ab/binary_secrets_h3_3`.
5. Confirm `Recovered secret!` appears in `n30_q61/salsa/main/train.log`.

## Successful run summary

- Problem: `N=30`, `Q=61`
- `orig_A` rows: `40000`
- Preprocess workers: `4`
- Preprocess `m`: `26`
- Prefix target: `500000` lines
- Final train samples: `56446`
- Final test samples: `10000`
- Secret type: `binary`
- Hamming weight: `3`
- SALSA result: recovered successfully

