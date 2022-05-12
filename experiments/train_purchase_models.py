import argparse
import os
import numpy as np
import pickle

from utils import data_utils, model_utils


def run_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_models", nargs="?", const=3, type=int, default=3)
    parser.add_argument(
        "--training_size", nargs="?", const=25000, type=int, default=25000
    )
    parser.add_argument(
        "--max_update_size", nargs="?", const=8000, type=int, default=8000
    )
    parser.add_argument("--epochs", nargs="?", const=95, type=int, default=95)
    parser.add_argument("--lr", nargs="?", const=0.1, type=float, default=0.1)
    return parser.parse_args()


def main(args):
    if not os.path.exists("experiments/purchase/saved_model"):
        os.mkdir("experiments/purchase/saved_model")

    training_scores = []
    testing_scores = []
    for i in range(args.num_models):
        all_dses = data_utils.purchase(
            1, args.training_size, args.max_update_size, data_seed=i
        )
        train_ds, _, _, test_ds, _ = all_dses
        x_trn, y_trn = train_ds
        x_tst, y_tst = test_ds

        model = model_utils.ImageNNet(
            "2f", x_trn.shape[1:], 100, np.arange(100), nup=None, lr=args.lr
        )

        model.fit_sgd(x_trn, np.eye(100)[y_trn], epochs=args.epochs)
        training_scores.append(model.score(x_trn, np.eye(100)[y_trn]))
        testing_scores.append(model.score(x_tst, np.eye(100)[y_tst]))
        print(f"Training Score: {training_scores[-1]}")
        print(f"Testing Score: {testing_scores[-1]}")

        model.keras_model.save(f"experiments/purchase/saved_model/model_{i}")
        pickle.dump(
            all_dses,
            open(f"experiments/purchase/saved_model/dataset_{i}", "wb"),
        )

    np.save("experiments/purchase/saved_model/training_scores", training_scores)
    np.save("experiments/purchase/saved_model/testing_scores", testing_scores)


if __name__ == "__main__":
    args = run_argparse()
    main(args)
