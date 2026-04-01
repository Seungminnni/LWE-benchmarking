# n50_q251 Experiments

이 폴더는 `N=50`, `Q=251` 기준으로 지금까지 돌린 실험을 한곳에 모아둔 정리 문서다.

## Directory Map

- `fresh/`
  - reduced 파이프라인용 원본 `A`
- `preproc/`
  - `LLL/BKZ/flatter` 전처리 결과
- `ab/`
  - reduced `A,b` 데이터셋
- `salsa/`
  - reduced 데이터셋으로 돌린 SALSA 런
- `ab_orig_only/`
  - 기존 `40k orig_A`를 reduction 없이 그대로 split 한 control 데이터셋
- `salsa_orig_only/`
  - `ab_orig_only` 기준 SALSA control 런
- `orig200k/`
  - `200k orig_A`를 새로 만들어 reduction 없이 처음부터 만든 control 실험

## Reduced Baseline

사용한 핵심 설정:

```bash
python3 src/generate/make_orig_A.py \
  --N 50 --Q 251 \
  --num_rows 40000 \
  --representation mod_q \
  --dump_path ./n50_q251/fresh
```

전처리는 `thresholds="0.85,0.95,0.98"`, `lll_delta=0.97`, `bkz_block_size=10`으로 수행했고 결과는 아래 경로에 있다.

- 원본 `A`: [origA_n50_logq8.npy](/home/yu_mcc/LWE-benchmarking/n50_q251/fresh/origA_n50_logq8.npy)
- 전처리 로그: [train.log](/home/yu_mcc/LWE-benchmarking/n50_q251/preproc/main/train.log)
- prefix 진행도: [prefix_progress.txt](/home/yu_mcc/LWE-benchmarking/n50_q251/preproc/main/prefix_progress.txt)
- merged prefix: [data.prefix](/home/yu_mcc/LWE-benchmarking/n50_q251/preproc/data.prefix)
- reduced `ab`: [train_A.npy](/home/yu_mcc/LWE-benchmarking/n50_q251/ab/train_A.npy), [test_A.npy](/home/yu_mcc/LWE-benchmarking/n50_q251/ab/test_A.npy)

shape:

- `orig_A`: `(40000, 50)`
- `train_A`: `(241883, 50)`
- `test_A`: `(10000, 50)`

SALSA 결과 (`binary`, `hamming=3`):

- `seed 0`: [train.log](/home/yu_mcc/LWE-benchmarking/n50_q251/salsa/main/train.log) 에서 `0:00:13`, `recover/matched=true`
- `seed 1`: [train.log](/home/yu_mcc/LWE-benchmarking/n50_q251/salsa/seed1/train.log) 에서 `0:00:25`, `recover/matched=true`
- `seed 2`: [train.log](/home/yu_mcc/LWE-benchmarking/n50_q251/salsa/seed2/train.log) 에서 `0:00:25`, `recover/matched=true`
- `seed 3`: [train.log](/home/yu_mcc/LWE-benchmarking/n50_q251/salsa/seed3/train.log) 에서 `0:00:38`, `recover/matched=true`
- `seed 4`: [train.log](/home/yu_mcc/LWE-benchmarking/n50_q251/salsa/seed4/train.log) 에서 `0:00:14`, `recover/matched=true`

요약하면 reduced 데이터셋에서는 모든 secret seed가 `epoch 0` 안에서 바로 복구됐다.

## Control 1: Orig-Only With 40k A

이 control은 reduction 결과를 전혀 쓰지 않고, `orig_A`를 그대로 `test=10000`, `train=30000`으로 split 해서 만든 데이터셋이다.

- 데이터셋: [train_A.npy](/home/yu_mcc/LWE-benchmarking/n50_q251/ab_orig_only/train_A.npy), [test_A.npy](/home/yu_mcc/LWE-benchmarking/n50_q251/ab_orig_only/test_A.npy)
- 로그: [train.log](/home/yu_mcc/LWE-benchmarking/n50_q251/salsa_orig_only/main/train.log)

shape:

- `orig_A`: `(40000, 50)`
- `train_A`: `(30000, 50)`
- `test_A`: `(10000, 50)`

결과:

- `recover/matched=true` 없음
- recovery 실패 `159`회
- 마지막 실패 시점: `epoch 55`, `0:24:51`

즉 같은 `N=50`, `Q=251`, `binary h=3`라도 reduction을 빼면 빠른 복구가 재현되지 않았다.

## Control 2: Orig-Only With 200k A From Scratch

같은 차원과 같은 `Q`에서 샘플 수만 크게 늘린 control이다. 이 실험은 기존 결과와 섞이지 않게 `orig200k/` 아래에 따로 두었다.

원본 `A` 생성:

```bash
python3 src/generate/make_orig_A.py \
  --N 50 --Q 251 \
  --num_rows 200000 \
  --representation mod_q \
  --dump_path ./n50_q251/orig200k/fresh
```

이후 reduction 없이 `orig_A`를 직접 사용해 `test=10000`, `train=190000` split을 만들고, `binary h=3` secret 5개와 대응하는 `orig_b/train_b/test_b`를 생성했다.

- 원본 `A`: [origA_n50_logq8.npy](/home/yu_mcc/LWE-benchmarking/n50_q251/orig200k/fresh/origA_n50_logq8.npy)
- orig-only `ab`: [train_A.npy](/home/yu_mcc/LWE-benchmarking/n50_q251/orig200k/ab/train_A.npy), [test_A.npy](/home/yu_mcc/LWE-benchmarking/n50_q251/orig200k/ab/test_A.npy)
- SALSA 로그: [train.log](/home/yu_mcc/LWE-benchmarking/n50_q251/orig200k/salsa/main/train.log)

shape:

- `orig_A`: `(200000, 50)`
- `train_A`: `(190000, 50)`
- `test_A`: `(10000, 50)`

실행한 SALSA 설정은 reduced baseline과 동일하게 유지했다. 다만 데이터만 reduction 없이 바꿨다.

결과:

- `recover/matched=true` 없음
- recovery 실패 `22`회
- 마지막 실패 시점: `epoch 1`, `0:08:42`
- 마지막 기록 step: `train/step = 43551`

이 실험은 비교 목적상 충분한 차이를 확인한 뒤 중단했다. 결론적으로 `orig-only`는 샘플 수를 `200k`로 늘려도 reduced baseline처럼 즉시 복구되지 않았다.

## Takeaway

- `n50_q251`에서 빠른 복구는 단순히 `N`이 작아서가 아니라, reduction을 거친 데이터셋의 영향이 매우 크다.
- 같은 `N=50`, `Q=251`에서 reduction을 빼면 `40k orig_A`도 실패했고, `200k orig_A`로 늘려도 즉시 성공이 나오지 않았다.
- 현재 low-d 실험에서는 `BKZ`보다 `flatter`가 먼저 threshold를 강하게 통과하는 쪽이 더 큰 요인으로 보인다. 자세한 로그는 [train.log](/home/yu_mcc/LWE-benchmarking/n50_q251/preproc/main/train.log) 를 보면 된다.
