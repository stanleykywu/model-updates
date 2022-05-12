import argparse
import os
import numpy as np

rnft_fmt_str = "python experiments/run_cifar_subpop.py --retrain={} --epochs_re={} --nup={} --ratio={} --subpop={}"

all_subpops = [1]  # range(2)
all_ratios = [0.2 * i for i in range(6)]
nups = [50, 100, 200]
for nup in nups:
    for ratio in all_ratios:
        for subpop in all_subpops:
            full_cmd = rnft_fmt_str.format("sgd", 2, nup, ratio, subpop, 0)
            print(full_cmd)
            os.system(full_cmd)

            only_cmd = rnft_fmt_str.format("sgd_only", 4, nup, ratio, subpop, 0)
            print(only_cmd)
            os.system(only_cmd)
