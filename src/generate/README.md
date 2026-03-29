# Data Generation Guide

This document covers the data-generation side of the repository:

- building and wiring `flatter`
- generating `orig_A`
- preprocessing into `data_*.prefix`
- turning preprocessed data into `(A, b)` datasets

It is written to be usable by both humans and coding agents. The commands below
match the current repository layout and the locally tested toy setup
`n=10, q=61`.

## Pipeline Overview

The generation pipeline is:

1. Build `flatter`
2. Create `orig_A.npy`
3. Run `preprocess.py`
4. Run `generate_A_b.py`
5. Feed the resulting `.npy` files into SALSA / CC

The most important invariant is:

- `preprocess.py --reload_data <orig_A>`
- `generate_A_b.py --orig_A_path <orig_A>`

must point to the same file.

Why this matters:

- `preprocess.py` does not store the full original `A` rows inside
  `data_*.prefix`
- it stores indices into the original `orig_A` pool plus reduction data
- later, `generate_A_b.py` reconstructs rows by loading that same `orig_A`
  again

If the files differ, the generated data is invalid.

## What `orig_A` Is

`orig_A` is the original pool of LWE rows.

If `N=10` and you create `4 * N` rows, then:

- shape is `(40, 10)`
- there are `40` original LWE equations
- each row has `10` coordinates

For the default `mod_q` representation, entries are sampled uniformly in
`[0, Q)`.

For example, with `Q=61`:

- minimum possible entry is `0`
- maximum possible entry is `60`

This is why a sanity check after generation might print:

```text
shape=(40, 10)
min=0
max=60
```

That is about the values inside `orig_A`, not the noise distribution.

## Choosing `Q` for Toy Runs

For toy experiments there is no single cryptographically optimal `Q`.
The practical choice depends on:

- `N`
- noise scale `sigma`
- relative noise `alpha = sigma / Q`

For the local toy path documented here:

- `N = 10`
- `Q = 61`
- `sigma = 3`

This is a reasonable small setup because:

- `Q=61` is large enough that the noise is not dominated by modular wraparound
- with fixed `sigma=3`, it is a tighter toy problem than `Q=113`
- it is still small enough to iterate quickly

## Building `flatter` Locally

This repository now supports a repo-local `flatter` build.
`src/generate/genSamples.py` first looks for:

```text
vendor/flatter/install/bin/flatter
```

and also supports overriding the path with:

```bash
export FLATTER_BIN=/path/to/flatter
```

### Environment

Use the non-sage environment:

```bash
conda activate lattice_env
```

### Install build dependencies into `lattice_env`

```bash
mamba install -n lattice_env -y -c conda-forge cmake gmp mpfr eigen=3.4.0 openblas fplll pkg-config
```

Notes:

- `eigen=3.4.0` matters here because newer `Eigen` versions can fail the
  `flatter` CMake version check
- these packages are installed inside `lattice_env`, so root access is not
  required

### Clone and build

From the repository root:

```bash
git clone https://github.com/keeganryan/flatter.git ./vendor/flatter
conda run -n lattice_env cmake -S ./vendor/flatter -B ./vendor/flatter/build -DCMAKE_INSTALL_PREFIX=$(pwd)/vendor/flatter/install -DCMAKE_PREFIX_PATH=$CONDA_PREFIX
conda run -n lattice_env env CPATH=$CONDA_PREFIX/include LIBRARY_PATH=$CONDA_PREFIX/lib LD_LIBRARY_PATH=$CONDA_PREFIX/lib cmake --build ./vendor/flatter/build --parallel 4
conda run -n lattice_env env LD_LIBRARY_PATH=$CONDA_PREFIX/lib cmake --install ./vendor/flatter/build
```

### Sanity check

```bash
env LD_LIBRARY_PATH=$(pwd)/vendor/flatter/install/lib:$CONDA_PREFIX/lib ./vendor/flatter/install/bin/flatter -h
```

Expected result:

- `flatter` prints its help text
- no missing library error appears

## Creating `orig_A`

Use:

```bash
python3 src/generate/make_orig_A.py --N 10 --Q 61 --dump_path ./data/toy_n10_q61
```

This creates:

```text
./data/toy_n10_q61/origA_n10_logq6.npy
```

### Why `logq6` for `Q=61`

The naming convention uses:

```text
logq = ceil(log2(Q))
```

For `Q=61`:

- `log2(61) ~= 5.93`
- `ceil(log2(61)) = 6`

So the file name becomes `origA_n10_logq6.npy`.

### Default row count

If `--num_rows` is omitted, `make_orig_A.py` uses:

```text
num_rows = 4 * N
```

So for `N=10`, the output shape is:

```text
(40, 10)
```

You can override this explicitly, for example:

```bash
python3 src/generate/make_orig_A.py --N 10 --Q 61 --num_rows 80 --dump_path ./data/toy_n10_q61
```

### Supported representations

Default:

```bash
--representation mod_q
```

This stores entries in `[0, Q)`.

Optional:

```bash
--representation centered
```

This stores entries in a centered interval around `0`.

For this repository, `mod_q` is the safest default and matches the provided
toy examples more closely.

## Running `preprocess.py`

Once `orig_A` exists, run preprocessing:

```bash
conda activate lattice_env
python3 src/generate/preprocess.py --N 10 --Q 61 --dump_path ./data --exp_name toy_n10_q61 --num_workers 4 --reload_data ./data/toy_n10_q61/origA_n10_logq6.npy --thresholds "0.55,0.56,0.57" --lll_penalty 10 --max_prefix_lines 250000
```

If you want preprocessing to stop automatically after producing enough prefix
data, add one of these limits:

```bash
--max_prefix_lines 250000
--max_prefix_blocks 31250
```

For this toy setting, `m=8`, so one logical prefix block is typically `8`
lines.

### Sample count targets for attacks

Repository-level targets from the main README are:

- CC: at least about `500,000` reduced samples
- SALSA: at least about `2,000,000` reduced samples

For this specific toy setting (`N=10`, `Q=61`, `m=8`) we observed about `18`
reduced rows per prefix block, so the following stopping rules are a practical
fit:

- CC-sized toy run:

```bash
--max_prefix_lines 250000
# or
--max_prefix_blocks 31250
```

This is about `31,250` blocks and about `560K` reduced samples.

- SALSA-sized toy run:

```bash
--max_prefix_lines 900000
# or
--max_prefix_blocks 112500
```

This is about `112,500` blocks and about `2.0M` reduced samples.

### What this does

- loads `orig_A`
- repeatedly samples subsets of rows from it
- constructs q-ary lattices
- runs `flatter` / BKZ-style reduction
- writes reduced information to `data_*.prefix`
- writes experiment metadata to `params.pkl`

### Important parameters

- `--N 10`
  - LWE dimension
- `--Q 61`
  - modulus
- `--reload_data`
  - path to the `orig_A` file
- `--thresholds "0.55,0.56,0.57"`
  - relaxed thresholds that work reliably for this tiny toy setting
- `--lll_penalty 10`
  - penalty used in the q-ary basis construction
- `--num_workers 4`
  - parallel toy setting that reaches the target prefix size faster
- `--max_prefix_lines 250000`
  - stops automatically after building a large enough toy prefix
- `--max_prefix_lines 900000`
  - good SALSA-sized target for this toy setting

### About `m`

If `--m` is not provided, `preprocess.py` uses:

```text
m = floor(7N / 8)
```

So for `N=10`, this becomes:

```text
m = 8
```

This `m` is the number of rows used in one reduction instance.
It is not the same as the total number of original rows in `orig_A`.

## Running `generate_A_b.py`

After preprocessing produces a directory with `params.pkl` and `data_*.prefix`,
run:

```bash
python3 src/generate/generate_A_b.py --processed_dump_path ./data/toy_n10_q61/<exp_id> --orig_A_path ./data/toy_n10_q61/origA_n10_logq6.npy --dump_path ./data/toy_n10_q61_ab --secret_type binary --min_hamming 2 --max_hamming 3 --num_secret_seeds 4 --sigma 3 --actions secrets
```
# 해밍무게 고정이 필요할 경우 민맥스 모두 고정하면 됨
Replace `<exp_id>` with the actual experiment subdirectory created by
`preprocess.py`.

### What this step does

- loads `params.pkl` from the preprocessing output
- reconstructs reduced `RA` from `data.prefix` plus `orig_A`
- generates secrets
- samples Gaussian error with standard deviation `sigma`
- forms `b = A s + e mod q`
- writes `train_A.npy`, `test_A.npy`, `orig_A.npy`, and per-secret `b` files

### About `sigma=3`

`sigma=3` means the Gaussian error has standard deviation `3`.

It does not mean:

- "values within `+-3` are accepted as correct"

It means:

- noise values are sampled around `0`
- typical values are near `0`
- larger values still occur with smaller probability

## Output Structure

After `generate_A_b.py`, expect output like:

```text
dump_path/
  orig_A.npy
  orig_b.npy
  train_A.npy
  test_A.npy
  binary_secrets_h2_3/
    params.pkl
    params.json
    secret.npy
    secret_2_0.npy
    train_b_2_0.npy
    test_b_2_0.npy
    ...
```

## Troubleshooting

### `flatter` not found

Check:

```bash
ls ./vendor/flatter/install/bin/flatter
```

Or set:

```bash
export FLATTER_BIN=/absolute/path/to/flatter
```

### `flatter` fails with missing shared libraries

Run with:

```bash
export LD_LIBRARY_PATH=$(pwd)/vendor/flatter/install/lib:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

### `generate_A_b.py` seems to ignore CLI `--N`

That is expected in the normal `secrets` flow.
It reloads `N` and `Q` from the preprocessing directory's `params.pkl`.

### `orig_A` naming looks odd for non-power-of-two `Q`

That is also expected.
The naming uses `ceil(log2(Q))`, not exact `log2(Q)`.

## Recommended Toy Flow

From the repository root:

```bash
conda activate lattice_env
python3 src/generate/make_orig_A.py --N 10 --Q 61 --dump_path ./data/toy_n10_q61
python3 src/generate/preprocess.py --N 10 --Q 61 --dump_path ./data --exp_name toy_n10_q61 --num_workers 4 --reload_data ./data/toy_n10_q61/origA_n10_logq6.npy --thresholds "0.55,0.56,0.57" --lll_penalty 10 --max_prefix_lines 250000
```

For a SALSA-sized toy preprocessing run, use:

```bash
conda activate lattice_env
python3 src/generate/make_orig_A.py --N 10 --Q 61 --dump_path ./data/toy_n10_q61
python3 src/generate/preprocess.py --N 10 --Q 61 --dump_path ./data --exp_name toy_n10_q61_preproc_salsa --num_workers 4 --reload_data ./data/toy_n10_q61/origA_n10_logq6.npy --thresholds "0.55,0.56,0.57" --lll_penalty 10 --max_prefix_lines 900000
```

Then inspect the generated preprocessing experiment folder under:

```text
./data/toy_n10_q61/
```

or more generally under the `dump_path/exp_name/exp_id` structure created by the
repository utilities, and use that path as `--processed_dump_path` in
`generate_A_b.py`.
