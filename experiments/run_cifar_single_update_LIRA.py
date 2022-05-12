import pickle
from utils import data_utils, model_utils
from utils.attack_utils import *

import os
import argparse
import numpy as np


def run_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_models", nargs="?", const=3, type=int, default=3)
    parser.add_argument("--nup", type=int)
    parser.add_argument("--num_trials", nargs="?", const=20, type=int, default=20)
    parser.add_argument("--retrain", choices=["sgd", "sgd_only"])
    parser.add_argument("--epochs_re", type=int)
    parser.add_argument("--lr", nargs="?", const=1e-4, type=float, default=1e-4)
    parser.add_argument(
        "--max_update_size", nargs="?", const=8000, type=int, default=8000
    )
    # ------------------ Line for LiRA shadow arguments
    parser.add_argument(
        "--lira_batch_size", nargs="?", const=128, type=int, default=128
    )
    parser.add_argument("--lira_epochs", nargs="?", const=30, type=int, default=30)
    parser.add_argument(
        "--lira_num_training", nargs="?", const=5000, type=int, default=5000
    )
    parser.add_argument("--lira_num_shadow", nargs="?", const=25, type=int, default=25)
    parser.add_argument("--lira_lr", nargs="?", const=1e-4, type=float, default=1e-4)
    return parser.parse_args()


def main(args):
    if not os.path.exists("experiments/CarliniShadowModels/cifar"):
        SAVE_FLAG = True
        LOAD_FLAG = False
    else:
        SAVE_FLAG = False
        LOAD_FLAG = True

    all_lira = []
    all_lira_ratio = []
    all_lira_precision = []
    all_lira_ratio_precision = []

    no_update_lira = []
    no_update_lira_precision = []

    all_initial_trn_acc = []
    all_initial_tst_acc = []
    all_post_trn_acc = []
    all_post_tst_acc = []

    # Experiment begins
    for data_seed in range(args.num_trials):
        for i in range(args.num_models):
            model = model_utils.ImageNNet(
                "rnft", (32, 32, 3), 10, np.arange(10), 128, lr=args.lr
            )
            load_model = tf.keras.models.load_model(
                f"experiments/cifar/saved_model/model_{i}"
            )
            model.keras_model = load_model
            model.lr = args.lr

            all_dses = pickle.load(
                open(f"experiments/cifar/saved_model/dataset_{i}", "rb")
            )
            (train_ds, update_dses, update_tests, test_ds, withold_ds) = all_dses
            x_trn, y_trn = train_ds
            x_tst, y_tst = test_ds
            x_up, y_up = (update_dses[0][0], update_dses[0][1])
            x_upt, y_upt = (update_tests[0][0], update_tests[0][1])

            x_up = x_up[: args.nup]
            y_up = y_up[: args.nup]

            rand_indices = np.random.choice(
                args.max_update_size, args.nup, replace=False
            )
            x_upt, y_upt = x_upt[rand_indices], y_upt[rand_indices]

            new_x = np.concatenate((x_up, x_upt))
            new_y = np.concatenate((y_up, y_upt))

            all_initial_trn_acc.append(model.score(x_trn, np.eye(10)[y_trn]))
            print("Initial Training Accuracy: {}".format(all_initial_trn_acc[-1]))
            all_initial_tst_acc.append(model.score(x_tst, np.eye(10)[y_tst]))
            print("Initial Testing Accuracy: {}".format(all_initial_tst_acc[-1]))

            lira_no_update_scores = LIRA_TF_scores(
                model=model.keras_model,
                dataset_x=withold_ds[0],
                dataset_y=np.eye(10)[withold_ds[1]],
                attack_x=new_x,
                attack_y=np.eye(10)[new_y],
                batch_size=args.lira_batch_size,
                epochs=args.lira_epochs,
                num_training=args.lira_num_training,
                num_shadow=args.lira_num_shadow,
                learning_rate=args.lira_lr,
                save=SAVE_FLAG,
                use_saved_models=LOAD_FLAG,
                model_name="cifar",
            )

            SAVE_FLAG = False
            LOAD_FLAG = True

            x_full, y_full = np.concatenate((x_trn, x_up)), np.concatenate(
                (y_trn, y_up)
            )
            if args.retrain == "sgd":
                model.fit_sgd(x_full, np.eye(10)[y_full], epochs=args.epochs_re)
            elif args.retrain == "sgd_only":
                model.fit_sgd(
                    x_up, np.eye(10)[y_up], epochs=args.epochs_re, sgd_only=True
                )
            else:
                raise NotImplementedError

            all_post_trn_acc.append(model.score(x_trn, np.eye(10)[y_trn]))
            print("Post Update Training Accuracy: {}".format(all_post_trn_acc[-1]))
            all_post_tst_acc.append(model.score(x_tst, np.eye(10)[y_tst]))
            print("Post Update Testing Accuracy: {}".format(all_post_tst_acc[-1]))

            lira_updated_scores = LIRA_TF_scores(
                model=model.keras_model,
                dataset_x=withold_ds[0],
                dataset_y=np.eye(10)[withold_ds[1]],
                attack_x=new_x,
                attack_y=np.eye(10)[new_y],
                batch_size=args.lira_batch_size,
                epochs=args.lira_epochs,
                num_training=args.lira_num_training,
                num_shadow=args.lira_num_shadow,
                learning_rate=args.lira_lr,
                save=SAVE_FLAG,
                use_saved_models=LOAD_FLAG,
                model_name="cifar",
            )

            # LIRA
            lira_score_diff = lira_updated_scores - lira_no_update_scores
            lira_acc = threshold(
                lira_score_diff[: args.nup],
                lira_score_diff[args.nup :],
                np.median(lira_score_diff),
            )

            lira_ratio = np.array(
                [
                    (0.0005 + new) / (0.0005 + old)
                    for new, old in zip(lira_updated_scores, lira_no_update_scores)
                ]
            )
            lira_ratio_acc = threshold(
                lira_ratio[: args.nup],
                lira_ratio[args.nup :],
                np.median(lira_ratio),
            )
            lira_precision = high_percententile(
                lira_score_diff[: args.nup], lira_score_diff[args.nup :], args.nup
            )
            lira_ratio_precision = high_percententile(
                lira_ratio[: args.nup], lira_ratio[args.nup :], args.nup
            )

            # LIRA no update
            lira_no_update_acc = threshold(
                lira_updated_scores[: args.nup],
                lira_updated_scores[args.nup :],
                np.median(lira_updated_scores),
            )
            lira_no_update_precision = high_percententile(
                lira_updated_scores[: args.nup],
                lira_updated_scores[args.nup :],
                args.nup,
            )

            print("=====================================================")
            print("=====================================================")
            print(
                "seed: {}, LIRA: {}, LIRA ratio: {}".format(
                    data_seed,
                    lira_acc,
                    lira_ratio_acc,
                )
            )
            all_lira.append(lira_acc)
            all_lira_ratio.append(lira_ratio_acc)
            print("=====================================================")
            print(
                "Precision: LIRA: {}, LIRA ratio: {}".format(
                    lira_precision,
                    lira_ratio_precision,
                )
            )
            all_lira_precision.append(lira_precision)
            all_lira_ratio_precision.append(lira_ratio_precision)
            print("=====================================================")
            print("(No update) lira: {}".format(lira_no_update_acc))
            no_update_lira.append(lira_no_update_acc)
            print("(No update Precision) lira: {}".format(lira_no_update_precision))
            no_update_lira_precision.append(lira_no_update_precision)

    print(
        "Overall: overall LIRA: {}, overall LIRA ratio: {}".format(
            np.mean(all_lira),
            np.mean(all_lira_ratio),
        )
    )
    print(
        "Overall Precision: overall LIRA: {}, overall LIRA ratio: {}".format(
            all_lira_precision,
            all_lira_ratio_precision,
        )
    )
    print("(No update) overall LIRA: {}".format(np.mean(no_update_lira)))
    print("(No update Precision) overall LIRA: {}".format(no_update_lira_precision))

    save_data = [
        all_lira,
        all_lira_ratio,
        all_lira_precision,
        all_lira_ratio_precision,
        no_update_lira,
        no_update_lira_precision,
    ]
    save_tag = f"experiments/cifar/single_update/{args.nup}_{args.retrain}_rnft_LIRA"

    np.save(save_tag, save_data, allow_pickle=True)


if __name__ == "__main__":
    args = run_argparse()
    main(args)
