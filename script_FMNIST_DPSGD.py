import os
import numpy as np

base = 0.01
target_epsilons = [base * (2 ** (i + 1)) for i in range(20)]

fmt_str = "python experiments/run_mnist_single_update.py --n=1000 --nup=100 --epochs_pre=50 --epochs_re=10 --dpsgd --l2_norm_clip=0.5 --target_epsilon={} --num_trials=20 --retrain={}"

for epsilon in target_epsilons:
    cmd = fmt_str.format(epsilon, "sgd")
    print(cmd)
    os.system(cmd)

    cmd = fmt_str.format(epsilon, "sgd_only")
    print(cmd)
    os.system(cmd)
