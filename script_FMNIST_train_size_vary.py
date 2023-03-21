import os

NS = [40 * (2**i) for i in range(6)]
LRS = [0.145, 0.145, 0.145, 0.145, 0.145, 0.145]
EPOCHS = [20, 30, 30, 50, 50, 50]

SINGLE_UPDATE_EXPERIMENTS = [
    "run_mnist_single_update_train_size_vary.py",
]

SU_SGD_FMT_STR = "python experiments/{} --n={} --num_trials=20 --epochs_pre={}"

for single_update in SINGLE_UPDATE_EXPERIMENTS:
    for N, epoch in zip(NS, EPOCHS):
        cmd = SU_SGD_FMT_STR.format(single_update, N, epoch)
        print(cmd)
        os.system(cmd)
