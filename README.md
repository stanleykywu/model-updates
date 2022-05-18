# model-updates

Experiments that explore privacy of model updates. Our attacks are shown effective using update information to give the adversary a significant advantage over attacks on standalone models, but also compared to a prior MI attack that takes advantage of model updates in a related machine-unlearning setting. The full paper can be found here: https://arxiv.org/abs/2205.06369

## How to Reproduce Paper Figures and Results

All results can be reproduced by running pre-written scripts from the root directory. Make sure to `pip install -r requirements.txt` before attempting to do so. Using a GPU is highly recommended for CIFAR-10 and IMDb experiments.

### FMNIST

1. `python script_FMNIST_complete.py`
   Generates results for both single update and multi-update for the FMNIST dataset. Head to the the respective notebooks to plot all figures after script completes.

2. `python script_FMNIST_DPSGD.py`
   Generates results for single update training with differential privacy on the FMNIST dataset. Head to the `experiments/dpsgd/fashion/noise_experiments` notebook to generate DPSGD plots after script completes.

### CIFAR-10

1. `python script_CIFAR_complete.py`
   Generates results for both single update and multi-update for the CIFAR-10 dataset. Head to the the respective notebooks to plot all figures after script completes.

2. `python script_CIFAR_subpop.py`
   Generates results for single update training with drift on the CIFAR-10 dataset. Head to the same directory as above for results.

### Purchase-100

1. `python script_PURCHASE_complete.py`
   Generates results for both single update and multi-update for the Purchase-100 dataset. Head to the the respective notebooks to plot all figures after script completes.

### IMDb

1. `python script_IMDB_complete.py`
   Generates results for both single update and multi-update for the IMDb dataset. Head to the the respective notebooks to plot all figures after script completes.
