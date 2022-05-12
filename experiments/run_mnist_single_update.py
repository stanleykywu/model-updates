from utils import data_utils, model_utils
from utils.attack_utils import *

import argparse
import numpy as np
from tensorflow_privacy.privacy.analysis.compute_noise_from_budget_lib import (
    compute_noise,
)


def run_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", nargs="?", const=1000, type=int, default=1000)
    parser.add_argument("--nup", type=int)
    parser.add_argument("--num_trials", nargs="?", const=20, type=int, default=20)
    parser.add_argument("--retrain", choices=["sgd", "sgd_only"])
    parser.add_argument("--epochs_pre", nargs="?", const=50, type=int, default=50)
    parser.add_argument("--epochs_re", type=int)
    parser.add_argument("--lr", nargs="?", const=0.0145, type=float, default=0.0145)
    # ---------------------------------------------------------
    # Divider for dpsgd required values
    parser.add_argument("--dpsgd", dest="dpsgd", action="store_true")
    parser.set_defaults(feature=False)
    parser.add_argument("--l2_norm_clip", type=float)
    parser.add_argument("--target_epsilon", type=float)
    return parser.parse_args()


def main(args):
    CUT_SIZE = 500
    all_losses = []
    all_losses_shadow = []
    all_gap = []
    all_ratio = []
    all_ratio_shadow = []
    all_loss_precision = []
    all_loss_precision_shadow = []
    all_loss_precision_calibration = []
    all_ratio_precision = []
    all_ratio_precision_shadow = []
    all_ratio_precision_calibration = []

    no_update_losses = []
    no_update_losses_shadow = []
    no_update_loss_precision = []
    no_update_loss_precision_shadow = []
    no_update_loss_precision_calibration = []
    no_update_gap = []

    all_initial_trn_acc = []
    all_initial_tst_acc = []
    all_post_trn_acc = []
    all_post_tst_acc = []

    # Declare victim model
    if args.dpsgd:
        noise_multiplier = compute_noise(
            args.n, args.nup, args.target_epsilon, args.epochs_re, 1e-5, 1e-5
        )
        print("The current noise multiplier is: {}".format(noise_multiplier))

        eps = compute_epsilon(
            noise_multiplier, args.nup, args.epochs_re * args.n // args.nup, args.n
        )
        print("The current epsilon is: {}".format(eps))

        model = model_utils.DPLogisticRegression(
            model_type="logreg",
            x_shape=784,
            y_shape=10,
            output_classes=np.arange(10),
            nup=args.nup,
            noise_multiplier=noise_multiplier,
            l2_norm_clip=args.l2_norm_clip,
            lr=args.lr,
        )
    else:
        model = model_utils.LogisticRegression(
            "logreg", 784, 10, np.arange(10), nup=args.nup, lr=args.lr
        )

    # Curate correct mnist dataset for shadow setting
    all_dses = data_utils.mnist(
        1,
        args.n,
        args.nup,
        data_seed=3000,
        flatten=True,
        fashion="True",
    )
    train_ds, update_dses, update_tests, test_ds, new_test_ds = all_dses

    x_trn, y_trn = train_ds
    x_tst, y_tst = test_ds[0], test_ds[1]

    # Train victim model
    model.fit_sgd(x_trn, np.eye(10)[y_trn], epochs=args.epochs_pre)

    # Calculate original loss benchmarks for victim model
    for update_ds, update_test in zip(update_dses, update_tests):
        x_up, y_up = update_ds
        x_upt, y_upt = update_test
        new_x = np.concatenate((x_up, x_upt))
        new_y = np.concatenate((y_up, y_upt))

        orig_losses = model.get_loss(new_x, new_y)
        orig_losses_shadow_thresh = best_acc_thresh(orig_losses, args.nup)
        orig_losses_shadow_precision_thresh = best_precision_thresh(
            orig_losses, args.nup
        )

    # Shadow models threshold training and setting
    x_full, y_full = x_trn, y_trn
    for update_ds, update_test in zip(update_dses, update_tests):
        x_up, y_up = update_ds
        x_upt, y_upt = update_test
        x_full, y_full = np.concatenate((x_full, x_up)), np.concatenate((y_full, y_up))

        if args.retrain == "sgd":
            model.fit_sgd(x_full, np.eye(10)[y_full], epochs=args.epochs_re)
        elif args.retrain == "sgd_only":
            model.fit_sgd(x_up, np.eye(10)[y_up], epochs=args.epochs_re, sgd_only=True)
        else:
            raise NotImplementedError

        new_x = np.concatenate((x_up, x_upt))
        new_y = np.concatenate((y_up, y_upt))
        losses = model.get_loss(new_x, new_y)

        loss_diffs = losses - orig_losses
        loss_diffs_shadow_thresh = best_acc_thresh(loss_diffs, args.nup)
        loss_diffs_shadow_precision_thresh = best_precision_thresh(loss_diffs, args.nup)
        ratio_shadow_thresh = best_acc_thresh(
            np.array(
                [
                    (0.0005 + new) / (0.0005 + old)
                    for new, old in zip(losses, orig_losses)
                ]
            ),
            args.nup,
        )
        ratio_shadow_precision_thresh = best_precision_thresh(
            np.array(
                [
                    (0.0005 + new) / (0.0005 + old)
                    for new, old in zip(losses, orig_losses)
                ]
            ),
            args.nup,
        )

    # Experiment begins
    for data_seed in range(args.num_trials):
        all_dses = data_utils.mnist(
            1,
            args.n,
            args.nup,
            data_seed=data_seed,
            flatten=True,
            fashion="True",
        )

        train_ds, update_dses, update_tests, test_ds, _ = all_dses

        x_trn, y_trn = train_ds
        x_tst, y_tst = test_ds[0], test_ds[1]

        if args.dpsgd:
            model = model_utils.DPLogisticRegression(
                "logreg",
                784,
                10,
                np.arange(10),
                nup=args.nup,
                noise_multiplier=noise_multiplier,
                l2_norm_clip=args.l2_norm_clip,
            )
        else:
            model = model_utils.LogisticRegression(
                "logreg", 784, 10, np.arange(10), nup=args.nup, lr=args.lr
            )

        model.fit_sgd(x_trn, np.eye(10)[y_trn], epochs=args.epochs_pre)

        all_initial_trn_acc.append(model.score(x_trn, y_trn))
        print("Initial Training Accuracy: {}".format(all_initial_trn_acc[-1]))
        all_initial_tst_acc.append(model.score(x_tst, y_tst))
        print("Initial Testing Accuracy: {}".format(all_initial_tst_acc[-1]))

        for update_ds, update_test in zip(update_dses, update_tests):
            x_up, y_up = update_ds
            x_upt, y_upt = update_test

            new_x = np.concatenate((x_up, x_upt))
            new_y = np.concatenate((y_up, y_upt))

            orig_losses = model.get_loss(new_x, new_y)
            test_orig_losses = model.get_loss(
                test_ds[0][:CUT_SIZE], test_ds[1][:CUT_SIZE]
            )

            gap_no_update = gap_attack(
                model.sklearn_model.predict(new_x), new_y, args.nup
            )

        x_full, y_full = x_trn, y_trn
        for update_ds, update_test in zip(update_dses, update_tests):
            x_up, y_up = update_ds
            x_upt, y_upt = update_test
            x_full, y_full = np.concatenate((x_full, x_up)), np.concatenate(
                (y_full, y_up)
            )

            if args.retrain == "sgd":
                model.fit_sgd(x_full, np.eye(10)[y_full], epochs=args.epochs_re)
            elif args.retrain == "sgd_only":
                model.fit_sgd(
                    x_up, np.eye(10)[y_up], epochs=args.epochs_re, sgd_only=True
                )
            else:
                raise NotImplementedError

            new_x = np.concatenate((x_up, x_upt))
            new_y = np.concatenate((y_up, y_upt))
            losses = model.get_loss(new_x, new_y)
            test_losses = model.get_loss(test_ds[0][:500], test_ds[1][:500])

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
            loss_shadow_acc = threshold(
                loss_diffs[: args.nup], loss_diffs[args.nup :], loss_diffs_shadow_thresh
            )
            loss_shadow_precision = precision(
                loss_diffs, args.nup, loss_diffs_shadow_precision_thresh
            ), recall(loss_diffs, args.nup, loss_diffs_shadow_precision_thresh)
            loss_calibration_precision = [
                (
                    precision(loss_diffs, args.nup, np.percentile(test_loss_diffs, i)),
                    recall(loss_diffs, args.nup, np.percentile(test_loss_diffs, i)),
                )
                for i in range(0, 110, 10)
            ]

            gap_acc = gap_attack(model.sklearn_model.predict(new_x), new_y, args.nup)

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
            ratio_shadow_acc = threshold(
                ratio[: args.nup], ratio[args.nup :], ratio_shadow_thresh
            )
            ratio_shadow_precision = precision(
                ratio, args.nup, ratio_shadow_precision_thresh
            ), recall(ratio, args.nup, ratio_shadow_precision_thresh)
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
            loss_no_update_shadow = threshold(
                losses[: args.nup], losses[args.nup :], orig_losses_shadow_thresh
            )
            loss_no_update_precision = high_percententile(
                losses[: args.nup], losses[args.nup :], args.nup
            )

            loss_no_update_precision_shadow = precision(
                losses, args.nup, orig_losses_shadow_precision_thresh
            ), recall(losses, args.nup, orig_losses_shadow_precision_thresh)
            loss_no_update_precision_calibration = [
                (
                    precision(losses, args.nup, np.percentile(test_orig_losses, i)),
                    recall(losses, args.nup, np.percentile(test_orig_losses, i)),
                )
                for i in range(0, 110, 10)
            ]

        all_post_trn_acc.append(model.score(x_trn, y_trn))
        print("Post Update Training Accuracy: {}".format(all_post_trn_acc[-1]))
        all_post_tst_acc.append(model.score(x_tst, y_tst))
        print("Post Update Testing Accuracy: {}".format(all_post_tst_acc[-1]))

        print("=====================================================")
        print("=====================================================")
        print(
            "seed: {}, loss: {}, gap: {}, ratio: {}".format(
                data_seed, loss_acc, gap_acc, ratio_acc
            )
        )
        all_losses.append(loss_acc)
        all_gap.append(gap_acc)
        all_ratio.append(ratio_acc)
        print("Shadow: loss: {}, ratio: {}".format(loss_shadow_acc, ratio_shadow_acc))
        all_losses_shadow.append(loss_shadow_acc)
        all_ratio_shadow.append(ratio_shadow_acc)
        print("=====================================================")
        print("Precision: loss: {}, ratio: {}".format(loss_precision, ratio_precision))
        all_loss_precision.append(loss_precision)
        all_ratio_precision.append(ratio_precision)
        print(
            "Shadow Precision: loss: {}, ratio: {}".format(
                loss_shadow_precision, ratio_shadow_precision
            )
        )
        all_loss_precision_shadow.append(loss_shadow_precision)
        all_ratio_precision_shadow.append(ratio_shadow_precision)
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
        print("(No update) Shadow: loss: {}".format(loss_no_update_shadow))
        no_update_losses_shadow.append(loss_no_update_shadow)
        print("(No update Precision) loss: {}".format(loss_no_update_precision))
        no_update_loss_precision.append(loss_no_update_precision)
        print(
            "(No update) Shadow Precision: loss: {}".format(
                loss_no_update_precision_shadow
            )
        )
        no_update_loss_precision_shadow.append(loss_no_update_precision_shadow)
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
        "Overall Shadow loss: {}, overall ratio: {}".format(
            np.mean(all_losses_shadow), np.mean(all_ratio_shadow)
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
        "Overall Precision Shadow: overall loss: {}, overall ratio: {}".format(
            all_loss_precision_shadow, all_ratio_precision_shadow
        )
    )
    print(
        "(No update) overall loss: {}, overall gap: {}".format(
            np.mean(no_update_losses), np.mean(no_update_gap)
        )
    )
    print(
        "(No update) Shadow overall loss: {}".format(np.mean(no_update_losses_shadow))
    )
    print("(No update Precision) overall loss: {}".format(no_update_loss_precision))
    print(
        "(No update Precision) Shadow overall loss: {}".format(
            no_update_loss_precision_shadow
        )
    )
    print(
        "(No update Precision) Test Calibration overall loss: {}".format(
            no_update_loss_precision_calibration
        )
    )

    save_data = [
        all_losses,
        all_gap,
        all_ratio,
        all_losses_shadow,
        all_ratio_shadow,
        all_loss_precision,
        all_ratio_precision,
        all_loss_precision_calibration,
        all_ratio_precision_calibration,
        all_loss_precision_shadow,
        all_ratio_precision_shadow,
        no_update_losses,
        no_update_gap,
        no_update_losses_shadow,
        no_update_loss_precision,
        no_update_loss_precision_shadow,
        no_update_loss_precision_calibration,
        all_initial_trn_acc,
        all_initial_tst_acc,
        all_post_trn_acc,
        all_post_tst_acc,
    ]

    if args.dpsgd:
        save_tag = f"experiments/dpsgd/{eps}_{noise_multiplier}_{args.l2_norm_clip}_{args.nup}_{args.n}_{args.retrain}"
        save_data.insert(0, eps)
    else:
        save_tag = f"experiments/mnist/fashion/single_update/{args.nup}_{args.n}_{args.retrain}"

    np.save(save_tag, save_data, allow_pickle=True)


if __name__ == "__main__":
    args = run_argparse()
    main(args)
