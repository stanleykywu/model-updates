import os

NUPS = [10 * (2 ** i) for i in range(6)]
SINGLE_UPDATE_EXPERIMENTS = [
    "run_mnist_single_update.py",
    "run_mnist_single_update_LIRA.py",
]

SU_SGD_FMT_STR = (
    "python experiments/{} --nup={} --num_trials=20 --retrain=sgd --epochs_re=10"
)
SU_SGD_ONLY_FMT_STR = (
    "python experiments/{} --nup={} --num_trials=20 --retrain=sgd_only --epochs_re=10"
)

for single_update in SINGLE_UPDATE_EXPERIMENTS:
    for nup in NUPS:
        cmd = SU_SGD_FMT_STR.format(single_update, nup)
        print(cmd)
        os.system(cmd)

        cmd = SU_SGD_ONLY_FMT_STR.format(single_update, nup)
        print(cmd)
        os.system(cmd)

MU_FMT_STR = "python experiments/run_mnist_updates.py --nup=10 --num_trials=20 --retrain={} --epochs_re=10"

cmd = MU_FMT_STR.format("sgd_only")
print(cmd)
os.system(cmd)

cmd = MU_FMT_STR.format("sgd")
print(cmd)
os.system(cmd)
