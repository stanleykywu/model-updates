from utils import model_utils
from utils.attack_utils import *

import tensorflow as tf
import argparse
import numpy as np
import pickle

targ_clses = [((0, 1), (3, 2)), ((1, 9), (3, 5))]


def run_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_models", nargs="?", const=3, type=int, default=3)
    parser.add_argument("--nup", type=int)
    parser.add_argument("--num_trials", nargs="?", const=20, type=int, default=20)
    parser.add_argument("--retrain", choices=["sgd", "sgd_only"])
    parser.add_argument("--epochs_re", type=int)
    parser.add_argument("--subpop", type=int)
    parser.add_argument("--ratio", type=float)
    parser.add_argument("--lr", nargs="?", const=1e-4, type=float, default=1e-4)
    parser.add_argument(
        "--max_update_size", nargs="?", const=8000, type=int, default=8000
    )
    return parser.parse_args()


def main(args):
    CUT_SIZE = 500
    all_losses = []
    all_gap = []
    all_ratio = []
    all_loss_precision = []
    all_loss_precision_calibration = []
    all_ratio_precision = []
    all_ratio_precision_calibration = []

    no_update_losses = []
    no_update_loss_precision = []
    no_update_loss_precision_calibration = []
    no_update_gap = []

    all_initial_trn_acc = []
    all_initial_tst_acc = []
    all_post_trn_acc = []
    all_post_tst_acc = []

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
                args.max_update_siz, args.nup, replace=False
            )
            x_upt, y_upt = x_tst[rand_indices], y_tst[rand_indices]

            new_x = np.concatenate((x_up, x_upt))
            new_y = np.concatenate((y_up, y_upt))

            all_initial_trn_acc.append(model.score(x_trn, np.eye(10)[y_trn]))
            print("Initial Training Accuracy: {}".format(all_initial_trn_acc[-1]))
            all_initial_tst_acc.append(model.score(x_tst, np.eye(10)[y_tst]))
            print("Initial Testing Accuracy: {}".format(all_initial_tst_acc[-1]))

            orig_losses = np.array(
                [
                    model.get_loss(np.array([x]), np.array([np.eye(2)[y]]))
                    for (x, y) in zip(new_x, new_y)
                ]
            )
            gap_no_update = gap_attack(
                model.keras_model.predict(new_x), new_y, args.nup
            )
            test_orig_losses = np.array(
                [
                    model.get_loss(np.array([x]), np.array([np.eye(2)[y]]))
                    for (x, y) in zip(test_ds[0][:CUT_SIZE], test_ds[1][:CUT_SIZE])
                ]
            )

            x_full, y_full = np.concatenate((x_trn, x_up)), np.concatenate(
                (y_trn, y_up)
            )
            if args.retrain == "sgd":
                model.fit_sgd(x_full, np.eye(2)[y_full], epochs=args.epochs_re)
            elif args.retrain == "sgd_only":
                model.fit_sgd(
                    x_up, np.eye(2)[y_up], epochs=args.epochs_re, sgd_only=True
                )
            else:
                raise NotImplementedError

            all_post_trn_acc.append(model.score(x_trn, np.eye(100)[y_trn]))
            print("Post Update Training Accuracy: {}".format(all_post_trn_acc[-1]))
            all_post_tst_acc.append(model.score(x_tst, np.eye(100)[y_tst]))
            print("Post Update Testing Accuracy: {}".format(all_post_tst_acc[-1]))

            # finding fixed ratios
            new_x = np.concatenate((x_up, x_upt))
            new_y = np.concatenate((y_up, y_upt))
            losses = np.array(
                [
                    model.get_loss(np.array([x]), np.array([np.eye(2)[y]]))
                    for (x, y) in zip(new_x, new_y)
                ]
            )
            test_losses = np.array(
                [
                    model.get_loss(np.array([x]), np.array([np.eye(2)[y]]))
                    for (x, y) in zip(test_ds[0][:500], test_ds[1][:500])
                ]
            )

            test_loss_diffs = test_losses - test_orig_losses
            loss_diffs = losses - orig_losses
            test_ratio = np.array(
                [
                    (0.0005 + new) / (0.0005 + old)
                    for new, old in zip(test_losses, test_orig_losses)
                ]
            )

            # update attacks
            loss_acc = threshold(
                loss_diffs[: args.nup], loss_diffs[args.nup :], np.median(loss_diffs)
            )
            loss_precision = high_percententile(
                loss_diffs[: args.nup], loss_diffs[args.nup :], args.nup
            )
            loss_calibration_precision = [
                (
                    precision(loss_diffs, args.nup, np.percentile(test_loss_diffs, i)),
                    recall(loss_diffs, args.nup, np.percentile(test_loss_diffs, i)),
                )
                for i in range(0, 110, 10)
            ]

            gap_acc = gap_attack(model.keras_model.predict(new_x), new_y, args.nup)

            ratio = np.array(
                [
                    (0.0005 + new) / (0.0005 + old)
                    for new, old in zip(losses, orig_losses)
                ]
            )
            ratio_acc = threshold(
                ratio[: args.nup], ratio[args.nup :], np.median(ratio)
            )
            ratio_precision = high_percententile(
                ratio[: args.nup], ratio[args.nup :], args.nup
            )
            ratio_calibration_precision = [
                (
                    precision(ratio, args.nup, np.percentile(test_ratio, i)),
                    recall(ratio, args.nup, np.percentile(test_ratio, i)),
                )
                for i in range(10, 100, 10)
            ]

            # no update attacks
            loss_no_update = threshold(
                losses[: args.nup], losses[args.nup :], np.median(losses)
            )
            loss_no_update_precision = high_percententile(
                losses[: args.nup], losses[args.nup :], args.nup
            )
            loss_no_update_precision_calibration = [
                (
                    precision(losses, args.nup, np.percentile(test_orig_losses, i)),
                    recall(losses, args.nup, np.percentile(test_orig_losses, i)),
                )
                for i in range(0, 110, 10)
            ]

            print("=====================================================")
            print("=====================================================")
            print(
                "seed/model: {}/{}, loss: {}, gap: {}, ratio: {}".format(
                    data_seed, i, loss_acc, gap_acc, ratio_acc
                )
            )
            all_losses.append(loss_acc)
            all_gap.append(gap_acc)
            all_ratio.append(ratio_acc)
            print("=====================================================")
            print(
                "Precision: loss: {}, ratio: {}".format(loss_precision, ratio_precision)
            )
            all_loss_precision.append(loss_precision)
            all_ratio_precision.append(ratio_precision)
            print(
                "Test Calibration Precision: loss: {}, ratio: {}".format(
                    loss_calibration_precision, ratio_calibration_precision
                )
            )
            all_loss_precision_calibration.append(loss_calibration_precision)
            all_ratio_precision_calibration.append(ratio_calibration_precision)
            print("=====================================================")
            print("(No update) loss: {}, gap: {}".format(loss_no_update, gap_no_update))
            no_update_losses.append(loss_no_update)
            no_update_gap.append(gap_no_update)
            print("(No update Precision) loss: {}".format(loss_no_update_precision))
            no_update_loss_precision.append(loss_no_update_precision)
            print(
                "(No update) Test Calibration Precision: loss: {}".format(
                    loss_no_update_precision_calibration
                )
            )
            no_update_loss_precision_calibration.append(
                loss_no_update_precision_calibration
            )
            print("=====================================================")

    print(
        "Overall loss: {}, overall gap: {}, overall ratio: {}".format(
            np.mean(all_losses), np.mean(all_gap), np.mean(all_ratio)
        )
    )
    print(
        "Overall Precision: overall loss: {}, overall ratio: {}".format(
            all_loss_precision, all_ratio_precision
        )
    )
    print(
        "Overall Test Calibration Precision: overall loss: {}, overall ratio: {}".format(
            all_loss_precision_calibration, all_ratio_precision_calibration
        )
    )
    print(
        "(No update) overall loss: {}, overall gap: {}".format(
            np.mean(no_update_losses), np.mean(no_update_gap)
        )
    )
    print("(No update Precision) overall loss: {}".format(no_update_loss_precision))
    print(
        "(No update Precision) Test Calibration overall loss: {}".format(
            no_update_loss_precision_calibration
        )
    )

    save_data = [
        all_losses,
        all_gap,
        all_ratio,
        all_loss_precision,
        all_ratio_precision,
        all_loss_precision_calibration,
        all_ratio_precision_calibration,
        no_update_losses,
        no_update_gap,
        no_update_loss_precision,
        no_update_loss_precision_calibration,
        all_initial_trn_acc,
        all_initial_tst_acc,
        all_post_trn_acc,
        all_post_tst_acc,
    ]

    save_tag = f"experiments/cifar/single_update/{args.nup}_{args.retrain}_{args.ratio}_{args.subpop}_drift"

    np.save(save_tag, save_data, allow_pickle=True)


if __name__ == "__main__":
    args = run_argparse()
    main(args)
