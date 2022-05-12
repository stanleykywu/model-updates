import os
import shutil

if os.path.exists("experiments/purchase/saved_model"):
    shutil.rmtree("experiments/purchase/saved_model")
os.system("python experiments/train_purchase_models.py")

NUPS = [250 * (2 ** i) for i in range(6)]
SINGLE_UPDATE_EXPERIMENTS = [
    "run_purchase_single_update_LIRA.py",
    "run_purchase_single_update_preload.py",
]

SU_SGD_FMT_STR = (
    "python experiments/{} --nup={} --num_trials=20 --retrain=sgd --epochs_re=10"
)
SU_SGD_ONLY_FMT_STR = (
    "python experiments/{} --nup={} --num_trials=20 --retrain=sgd_only --epochs_re=5"
)

for single_update in SINGLE_UPDATE_EXPERIMENTS:
    for nup in NUPS:
        cmd = SU_SGD_FMT_STR.format(single_update, nup)
        print(cmd)
        os.system(cmd)

        cmd = SU_SGD_ONLY_FMT_STR.format(single_update, nup)
        print(cmd)
        os.system(cmd)

MU_FMT_STR = "python experiments/run_purchase_updates.py --nup=250 --num_trials=20 --retrain={} --epochs_re={}"

cmd = MU_FMT_STR.format("sgd_only", 5)
print(cmd)
os.system(cmd)

cmd = MU_FMT_STR.format("sgd", 10)
print(cmd)
os.system(cmd)
