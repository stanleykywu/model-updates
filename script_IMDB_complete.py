import os
import shutil

if os.path.exists("experiments/imdb/saved_model"):
    shutil.rmtree("experiments/imdb/saved_model")
if os.path.exists("experiments/imdb/saved_BERT"):
    shutil.rmtree("experiments/imdb/saved_BERT")
os.system("python experiments/train_imdb_models.py")
os.system("python experiments/train_imdb_shadow_models.py")

NUPS = [20 * (2 ** i) for i in range(5)]
SINGLE_UPDATE_EXPERIMENTS = ["run_imdb_single_update_preload.py"]

SU_SGD_FMT_STR = (
    "python experiments/{} --nup={} --num_trials=20 --retrain=sgd --epochs_re=3"
)
SU_SGD_ONLY_FMT_STR = (
    "python experiments/{} --nup={} --num_trials=20 --retrain=sgd_only --epochs_re=6"
)

for single_update in SINGLE_UPDATE_EXPERIMENTS:
    for nup in NUPS:
        cmd = SU_SGD_FMT_STR.format(single_update, nup)
        print(cmd)
        os.system(cmd)

        cmd = SU_SGD_ONLY_FMT_STR.format(single_update, nup)
        print(cmd)
        os.system(cmd)

MU_FMT_STR = "python experiments/run_imdb_updates.py --nup=20 --num_trials=20 --retrain={} --epochs_re={}"

cmd = MU_FMT_STR.format("sgd_only", 6)
print(cmd)
os.system(cmd)

cmd = MU_FMT_STR.format("sgd", 3)
print(cmd)
os.system(cmd)
