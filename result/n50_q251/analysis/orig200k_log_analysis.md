# n50_q251 orig200k Log Analysis

## Summary

- Log: [n50_q251/orig200k/salsa/main/train.log](/home/yu_mcc/LWE-benchmarking/n50_q251/orig200k/salsa/main/train.log)
- Train points: 24514
- Recovery attempts: 1325
- Recovery successes: 0
- Last epoch: 102
- Last train step: 2446201
- Last elapsed time: 8.36 h
- Final train acc1: 0.175623
- Final train loss: 2.765573
- Best recover acc1: 0.214844 at epoch 5, line 1668
- Best recover loss: 2.792875 at epoch 83, line 24759

## Selected Epochs

| epoch | train_acc1_mean | train_acc1_last | train_loss_mean | recover_acc1_best | recover_acc1_mean | recover_loss_best | matched |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| 0 | 0.162379 | 0.167567 | 2.838281 | 0.187500 | 0.161133 | 2.798071 | N |
| 1 | 0.167935 | 0.168432 | 2.803669 | 0.187500 | 0.170974 | 2.798805 | N |
| 2 | 0.168755 | 0.168981 | 2.804093 | 0.203125 | 0.164663 | 2.796611 | N |
| 5 | 0.170715 | 0.170958 | 2.797899 | 0.214844 | 0.175781 | 2.795366 | N |
| 10 | 0.172998 | 0.173063 | 2.801972 | 0.195312 | 0.171875 | 2.794265 | N |
| 20 | 0.173951 | 0.173954 | 2.791359 | 0.203125 | 0.181190 | 2.796046 | N |
| 50 | 0.175041 | 0.175048 | 2.795157 | 0.207031 | 0.179988 | 2.795563 | N |
| 100 | 0.175612 | 0.175614 | 2.790974 | 0.195312 | 0.179087 | 2.795239 | N |
| 102 | 0.175626 | 0.175623 | 2.792129 | 0.187500 | 0.177409 | 2.796077 | N |

## Reduced Baseline Comparison

- Reduced log: [n50_q251/salsa/main/train.log](/home/yu_mcc/LWE-benchmarking/n50_q251/salsa/main/train.log)
- Reduced first recovery: acc1=0.226562, matched=True, elapsed=12s
- Reduced first success line: 103
