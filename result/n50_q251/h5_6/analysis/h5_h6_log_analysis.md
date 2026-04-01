# n50_q251 h5/h6 SALSA Log Analysis

## Summary

- Preprocess progress: `1000008 23256`
- Dataset: `train=508158`, `test=10000`, `orig=40000`
- h=5 first success: `0:00:55` at epoch `0`
- h=6 first success: `0:09:36` at epoch `1`
- h=6 failed recoveries before first success: `43`
- h=6 best failed recover acc1: `0.230469`
- h=6 success recover acc1: `0.156250`

## Files

- Curves: [h6_seed0_curves.png](/home/yu_mcc/LWE-benchmarking/n50_q251/h5_6/analysis/h6_seed0_curves.png)
- Comparison: [h5_vs_h6_comparison.png](/home/yu_mcc/LWE-benchmarking/n50_q251/h5_6/analysis/h5_vs_h6_comparison.png)
- Summary CSV: [h5_h6_summary.csv](/home/yu_mcc/LWE-benchmarking/n50_q251/h5_6/analysis/h5_h6_summary.csv)
- h6 recover attempts: [h6_seed0_recover_attempts.csv](/home/yu_mcc/LWE-benchmarking/n50_q251/h5_6/analysis/h6_seed0_recover_attempts.csv)
- h6 selected train checkpoints: [h6_seed0_selected_train.csv](/home/yu_mcc/LWE-benchmarking/n50_q251/h5_6/analysis/h6_seed0_selected_train.csv)

## Key Observations

- `h=6` did not recover in epoch 0 despite repeated recovery attempts every 2000 steps.
- `h=6` first success happened only in epoch 1 after a long plateau in `train/acc1` around `0.169`.
- `h=5` recovered much earlier even though the underlying reduced dataset and model configuration were the same.
- For `h=6`, a higher failed `recover/acc1` did not guarantee success; the best failed attempt had higher `recover/acc1` than the first successful attempt.
