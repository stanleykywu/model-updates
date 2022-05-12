import argparse
import os
import torch
import pickle
import numpy as np

from utils import data_utils, model_utils


def run_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_models", nargs="?", const=8, type=int, default=8)
    parser.add_argument(
        "--training_size", nargs="?", const=25000, type=int, default=25000
    )
    parser.add_argument(
        "--max_update_size", nargs="?", const=5000, type=int, default=5000
    )
    parser.add_argument("--epochs", nargs="?", const=4, type=int, default=4)
    parser.add_argument("--lr", nargs="?", const=1e-5, type=float, default=1e-5)
    return parser.parse_args()


def main(args):
    if not os.path.exists("experiments/imdb/saved_BERT"):
        os.mkdir("experiments/imdb/saved_BERT")
    if not os.path.exists("experiments/imdb/saved_data"):
        os.mkdir("experiments/imdb/saved_data")

    training_scores = []
    testing_scores = []
    for i in range(args.num_models):
        all_dses = data_utils.imdb(
            1, args.training_size, args.max_update_size, data_seed=i
        )
        (train_ds, _, _, test_ds, _, tokenizer, data_collator) = all_dses

        model = model_utils.BERTSentiment(
            "bert",
            (300,),
            2,
            100,
            output_dir="experiments/imdb/saved_data",
            tokenizer=tokenizer,
            data_collator=data_collator,
            lr=args.lr,
        )
        model.fit_sgd(train_ds, epochs=args.epochs)
        training_scores.append(model.score(train_ds))
        testing_scores.append(model.score(test_ds))
        print(f"Training Score: {training_scores[-1]}")
        print(f"Testing Score: {testing_scores[-1]}")

        torch.save(model.model, "experiments/imdb/saved_BERT/model_{}".format(i))
        pickle.dump(
            all_dses,
            open("experiments/imdb/saved_BERT/dataset_{}".format(i), "wb"),
        )

    np.save("experiments/imdb/saved_data/training_scores", training_scores)
    np.save("experiments/imdb/saved_data/testing_scores", testing_scores)


if __name__ == "__main__":
    args = run_argparse()
    main(args)
