from utils import data_utils, model_utils
from utils.attack_utils import *

import argparse
import numpy as np

# from tensorflow_privacy.privacy.analysis.compute_noise_from_budget_lib import (
#     compute_noise,
# )


def run_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", nargs="?", const=1000, type=int, default=1000)
    parser.add_argument("--num_trials", nargs="?", const=1, type=int, default=1)
    parser.add_argument("--epochs_pre", nargs="?", const=50, type=int, default=50)
    parser.add_argument("--lr", nargs="?", const=0.145, type=float, default=0.145)
    # ---------------------------------------------------------
    # Divider for dpsgd required values
    parser.add_argument("--dpsgd", dest="dpsgd", action="store_true")
    parser.set_defaults(feature=False)
    parser.add_argument("--l2_norm_clip", type=float)
    parser.add_argument("--target_epsilon", type=float)
    # ------------------ Line for LiRA shadow arguments
    parser.add_argument("--lira_num_shadow", nargs="?", const=50, type=int, default=50)
    return parser.parse_args()


def main(args):
    SAVE_FLAG = True
    LOAD_FLAG = False
    CUT_SIZE = 500
    no_update_losses = []
    no_update_losses_shadow = []
    no_update_loss_precision = []
    no_update_loss_precision_shadow = []
    no_update_loss_precision_calibration = []
    no_update_gap = []
    no_update_lira = []
    no_update_lira_precision = []

    all_trn_acc = []
    all_tst_acc = []

    # Declare victim model
    if args.dpsgd:
        noise_multiplier = compute_noise(
            args.n, args.n, args.target_epsilon, args.epochs_pre, 1e-5, 1e-5
        )
        print("The current noise multiplier is: {}".format(noise_multiplier))

        eps = compute_epsilon(
            noise_multiplier, args.n, args.epochs_pre * args.n // args.n, args.n
        )
        print("The current epsilon is: {}".format(eps))

        model = model_utils.DPLogisticRegression(
            model_type="logreg",
            x_shape=784,
            y_shape=10,
            output_classes=np.arange(10),
            nup=args.n,
            noise_multiplier=noise_multiplier,
            l2_norm_clip=args.l2_norm_clip,
            lr=args.lr,
        )
    else:
        model = model_utils.LogisticRegression(
            "logreg", 784, 10, np.arange(10), nup=args.n, lr=args.lr
        )

    # Curate correct mnist dataset for shadow setting
    all_dses = data_utils.mnist(
        1,
        args.n,
        args.n,
        data_seed=3000,
        flatten=True,
        fashion="True",
    )
    train_ds, update_dses, update_tests, _, _ = all_dses

    x_trn, y_trn = train_ds

    # Train victim model
    model.fit_sgd(x_trn, np.eye(10)[y_trn], epochs=args.epochs_pre)

    # Calculate original loss benchmarks for victim model
    for update_ds in update_dses:
        x_up, y_up = update_ds
        new_x = np.concatenate((x_trn, x_up))
        new_y = np.concatenate((y_trn, y_up))

        orig_losses = model.get_loss(new_x, new_y)
        orig_losses_shadow_thresh = best_acc_thresh(orig_losses, args.n)
        orig_losses_shadow_precision_thresh = best_precision_thresh(orig_losses, args.n)

    # Experiment begins
    for data_seed in range(args.num_trials):
        all_dses = data_utils.mnist(
            1,
            args.n,
            args.n,
            data_seed=data_seed,
            flatten=True,
            fashion="True",
        )

        train_ds, update_dses, _, test_ds, withold_ds = all_dses

        x_trn, y_trn = train_ds
        x_tst, y_tst = test_ds[0], test_ds[1]

        if args.dpsgd:
            model = model_utils.DPLogisticRegression(
                "logreg",
                784,
                10,
                np.arange(10),
                nup=args.n,
                noise_multiplier=noise_multiplier,
                l2_norm_clip=args.l2_norm_clip,
            )
        else:
            model = model_utils.LogisticRegression(
                "logreg", 784, 10, np.arange(10), nup=args.n, lr=args.lr
            )

        model.fit_sgd(x_trn, np.eye(10)[y_trn], epochs=args.epochs_pre)

        all_trn_acc.append(model.score(x_trn, y_trn))
        print("Training Accuracy: {}".format(all_trn_acc[-1]))
        all_tst_acc.append(model.score(x_tst, y_tst))
        print("Testing Accuracy: {}".format(all_tst_acc[-1]))

        for update_ds in update_dses:
            x_up, y_up = update_ds

            new_x = np.concatenate((x_trn, x_up))
            new_y = np.concatenate((y_trn, y_up))

            lira_scores = LIRA_TF_scores(
                model=model.keras_model,
                dataset_x=withold_ds[0],
                dataset_y=np.eye(10)[withold_ds[1]],
                attack_x=new_x,
                attack_y=np.eye(10)[new_y],
                batch_size=args.n,
                epochs=args.epochs_pre,
                num_training=args.n,
                num_shadow=args.lira_num_shadow,
                learning_rate=args.lr,
                save=SAVE_FLAG,
                use_saved_models=LOAD_FLAG,
                model_name="FMNIST",
            )

            SAVE_FLAG = False
            LOAD_FLAG = True

            losses = model.get_loss(new_x, new_y)
            test_orig_losses = model.get_loss(
                test_ds[0][:CUT_SIZE], test_ds[1][:CUT_SIZE]
            )

            gap_no_update = gap_attack(
                model.sklearn_model.predict(new_x), new_y, args.n, logits=False
            )

            # no update attacks
            loss_no_update = threshold(
                losses[: args.n], losses[args.n :], np.median(losses)
            )
            loss_no_update_shadow = threshold(
                losses[: args.n], losses[args.n :], orig_losses_shadow_thresh
            )
            loss_no_update_precision = high_percententile(
                losses[: args.n], losses[args.n :], args.n
            )

            loss_no_update_precision_shadow = precision(
                losses, args.n, orig_losses_shadow_precision_thresh
            ), recall(losses, args.n, orig_losses_shadow_precision_thresh)
            loss_no_update_precision_calibration = [
                (
                    precision(losses, args.n, np.percentile(test_orig_losses, i)),
                    recall(losses, args.n, np.percentile(test_orig_losses, i)),
                )
                for i in range(0, 110, 10)
            ]

            lira_no_update_acc = threshold(
                lira_scores[: args.n],
                lira_scores[args.n :],
                np.median(lira_scores),
            )
            lira_no_update_precision = high_percententile(
                lira_scores[: args.n],
                lira_scores[args.n :],
                args.n,
            )

        print("=====================================================")
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
        print("(No update) lira: {}".format(lira_no_update_acc))
        no_update_lira.append(lira_no_update_acc)
        print("(No update Precision) lira: {}".format(lira_no_update_precision))
        no_update_lira_precision.append(lira_no_update_precision)

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
        no_update_losses,
        no_update_gap,
        no_update_losses_shadow,
        no_update_loss_precision,
        no_update_loss_precision_shadow,
        no_update_loss_precision_calibration,
        no_update_lira,
        no_update_lira_precision,
        all_trn_acc,
        all_tst_acc,
    ]

    if args.dpsgd:
        save_tag = f"experiments/dpsgd/train_size_vary/{eps}_{noise_multiplier}_{args.l2_norm_clip}_{args.n}"
        save_data.insert(0, eps)
    else:
        save_tag = f"experiments/mnist/fashion/single_update/train_size_vary/{args.n}"

    np.save(save_tag, save_data, allow_pickle=True)


if __name__ == "__main__":
    args = run_argparse()
    main(args)
