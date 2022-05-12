import argparse
import numpy as np

from utils import model_utils, data_utils
from utils.attack_utils import *


def run_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", nargs="?", const=1000, type=int, default=1000)
    parser.add_argument("--nup", type=int)
    parser.add_argument("--num_trials", nargs="?", const=20, type=int, default=20)
    parser.add_argument("--retrain", choices=["sgd", "sgd_only"])
    parser.add_argument("--epochs_pre", nargs="?", const=50, type=int, default=50)
    parser.add_argument("--epochs_re", type=int)
    parser.add_argument("--num_updates", nargs="?", const=10, type=int, default=10)
    parser.add_argument("--lr", nargs="?", const=0.0145, type=int, default=0.0145)
    return parser.parse_args()


def main(args):
    accuracies = {
        update: {"training": [], "testing": []}
        for update in range(args.num_updates + 1)
    }
    total_data = {
        update: {
            "bf_loss": [],
            "dg_loss": [],
            "ds_loss": [],
            "bf_ratio": [],
            "dg_ratio": [],
            "ds_ratio": [],
            "nup_loss": [],
            "gap": [],
            "avg_dist": [],
        }
        for update in range(args.num_updates)
    }

    for data_seed in range(args.num_trials):
        print("Seed: ", data_seed)
        all_dses = data_utils.mnist(
            args.num_updates,
            args.n,
            args.nup,
            data_seed=data_seed,
            flatten=True,
            fashion="True",
        )

        train_ds, update_dses, update_tests, test_ds, _ = all_dses
        x_trn, y_trn = train_ds
        x_tst, y_tst = train_ds

        model = model_utils.LogisticRegression(
            "logreg", 784, 10, np.arange(10), args.nup, lr=args.lr
        )
        model.fit_sgd(x_trn, np.eye(10)[y_trn], epochs=args.epochs_pre)

        accuracies[0]["training"].append(model.sklearn_model.score(x_trn, y_trn))
        accuracies[0]["testing"].append(model.sklearn_model.score(x_tst, y_tst))

        data = [[] for _ in range(args.nup * args.num_updates * 2)]

        for index, (update_ds, update_test) in enumerate(
            zip(update_dses, update_tests)
        ):
            x_up, y_up = update_ds
            x_upt, y_upt = update_test

            for pt_id, loss in enumerate(model.get_loss(x_up, y_up)):
                data[index * args.nup + pt_id].append(loss)

            for pt_id, loss in enumerate(model.get_loss(x_upt, y_upt)):
                data[args.num_updates * args.nup + index * args.nup + pt_id].append(
                    loss
                )

        x_full, y_full = x_trn, y_trn
        for index, (update_ds, update_test) in enumerate(
            zip(update_dses, update_tests)
        ):
            print("update ", index + 1)

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

            x_temp = np.concatenate([xup for xup, _ in update_dses])
            y_temp = np.concatenate([yup for _, yup in update_dses])

            for pt_id, loss in enumerate(model.get_loss(x_temp, y_temp)):
                data[pt_id].append(loss)

            x_temp = np.concatenate([xupt for xupt, _ in update_tests])
            y_temp = np.concatenate([yupt for _, yupt in update_tests])

            for pt_id, loss in enumerate(model.get_loss(x_temp, y_temp)):
                data[args.num_updates * args.nup + pt_id].append(loss)

            accuracies[index + 1]["training"].append(
                model.sklearn_model.score(x_trn, y_trn)
            )
            accuracies[index + 1]["testing"].append(
                model.sklearn_model.score(x_tst, y_tst)
            )

            temp_data = np.concatenate(
                (
                    data[: (index + 1) * args.nup],
                    data[
                        args.num_updates * args.nup : args.num_updates * args.nup
                        + (index + 1) * args.nup
                    ],
                )
            )
            (
                delta_general_accuracy_loss,
                delta_specific_accuracy_loss,
                avg_dist_loss,
            ) = delta_attack(temp_data, index + 1, args.nup)
            back_front_accuracy_loss = back_front_attack(temp_data, index + 1, args.nup)
            (
                delta_general_accuracy_ratio,
                delta_specific_accuracy_ratio,
                avg_dist_ratio,
            ) = delta_attack(temp_data, index + 1, args.nup, type="ratio")
            back_front_accuracy_ratio = back_front_attack(
                temp_data, index + 1, args.nup, type="ratio"
            )
            print(
                "   Loss Back Front Attack general accuracy: ",
                back_front_accuracy_loss,
            )
            print(
                "   Loss Delta attack general attack accuracy: ",
                delta_general_accuracy_loss,
            )
            print(
                "   Loss Delta attack specific accuracy: ",
                delta_specific_accuracy_loss,
            )
            print(
                "   Ratio Back Front Attack general accuracy: ",
                back_front_accuracy_ratio,
            )
            print(
                "   Ratio Delta attack general attack accuracy: ",
                delta_general_accuracy_ratio,
            )
            print(
                "   Ratio Delta attack specific accuracy: ",
                delta_specific_accuracy_ratio,
            )
            total_data[index]["bf_loss"].append(back_front_accuracy_loss)
            total_data[index]["dg_loss"].append(delta_general_accuracy_loss)
            total_data[index]["ds_loss"].append(delta_specific_accuracy_loss)
            total_data[index]["bf_ratio"].append(back_front_accuracy_ratio)
            total_data[index]["dg_ratio"].append(delta_general_accuracy_ratio)
            total_data[index]["ds_ratio"].append(delta_specific_accuracy_ratio)
            total_data[index]["nup_loss"].append(
                threshold(
                    temp_data[: args.nup * (index + 1)],
                    temp_data[args.nup * (index + 1) :],
                    np.median(temp_data),
                )
            )
            acc_x, acc_y = [], []
            for x, y in update_dses[: index + 1]:
                for x_data in x:
                    acc_x.append(x_data)
                for y_data in y:
                    acc_y.append(y_data)

            for x, y in update_tests[: index + 1]:
                for x_data in x:
                    acc_x.append(x_data)
                for y_data in y:
                    acc_y.append(y_data)
            total_data[index]["gap"].append(
                gap_attack(
                    model.keras_model.predict(np.array(acc_x)),
                    np.array(acc_y),
                    args.nup * (index + 1),
                )
            )
            total_data[index]["avg_dist"].append((avg_dist_loss, avg_dist_ratio))
            print("----------------------------------------------------------------")

    np.save(
        f"experiments/mnist/fashion/multiple_update/{args.num_updates}_{args.nup}_{args.retrain}",
        (total_data, accuracies),
    )


if __name__ == "__main__":
    args = run_argparse()
    main(args)
