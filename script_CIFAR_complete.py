import os
import shutil

if os.path.exists("experiments/cifar/saved_model"):
    shutil.rmtree("experiments/cifar/saved_model")
os.system("python experiments/train_cifar_models.py")

NUPS = [250 * (2 ** i) for i in range(6)]
SINGLE_UPDATE_EXPERIMENTS = ["run_cifar_single_update_LIRA.py", "run_cifar_single_update_preload.py"]

SU_SGD_FMT_STR = (
    "python experiments/{} --nup={} --num_trials=20 --retrain=sgd --epochs_re=2"
)
SU_SGD_ONLY_FMT_STR = (
    "python experiments/{} --nup={} --num_trials=20 --retrain=sgd_only --epochs_re=4"
)

for single_update in SINGLE_UPDATE_EXPERIMENTS:
    for nup in NUPS:
        cmd = SU_SGD_FMT_STR.format(single_update, nup)
        print(cmd)
        os.system(cmd)

        cmd = SU_SGD_ONLY_FMT_STR.format(single_update, nup)
        print(cmd)
        os.system(cmd)

MU_FMT_STR = "python experiments/run_cifar_updates.py --nup=250 --num_trials=20 --retrain={} --epochs_re={}"

cmd = MU_FMT_STR.format("sgd_only", 4)
print(cmd)
os.system(cmd)

cmd = MU_FMT_STR.format("sgd", 2)
print(cmd)
os.system(cmd)
