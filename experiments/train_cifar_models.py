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
    parser.add_argument("--epochs", nargs="?", const=12, type=int, default=12)
    parser.add_argument("--lr", nargs="?", const=1e-4, type=float, default=1e-4)
    return parser.parse_args()


def main(args):
    if not os.path.exists("experiments/cifar/saved_model"):
        os.mkdir("experiments/cifar/saved_model")

    training_scores = []
    testing_scores = []
    for i in range(args.num_models):
        all_dses = data_utils.mnist(
            1,
            args.training_size,
            args.max_update_size,
            data_seed=i,
            flatten=False,
            fashion="cifar",
            withold=45000,
        )
        train_ds, _, _, test_ds, _ = all_dses
        x_trn, y_trn = train_ds
        x_tst, y_tst = test_ds

        model = model_utils.ImageNNet(
            "rnft", (32, 32, 3), 10, np.arange(10), nup=128, lr=args.lr
        )

        model.fit_sgd(
            x_trn,
            np.eye(10)[y_trn],
            epochs=args.epochs,
        )
        training_scores.append(model.score(x_trn, np.eye(10)[y_trn]))
        testing_scores.append(model.score(x_tst, np.eye(10)[y_tst]))
        print(f"Training Score: {training_scores[-1]}")
        print(f"Testing Score: {testing_scores[-1]}")

        model.keras_model.save(f"experiments/cifar/saved_model/model_{i}")
        pickle.dump(
            all_dses,
            open(f"experiments/cifar/saved_model/dataset_{i}", "wb"),
        )

    np.save("experiments/purchase/saved_model/training_scores", training_scores)
    np.save("experiments/purchase/saved_model/testing_scores", testing_scores)


if __name__ == "__main__":
    args = run_argparse()
    main(args)
