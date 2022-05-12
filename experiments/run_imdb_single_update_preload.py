from utils import model_utils
from utils.attack_utils import *

import os
import pickle

import argparse
import numpy as np
import datasets


def run_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_models", nargs="?", const=3, type=int, default=3)
    parser.add_argument("--nup", type=int)
    parser.add_argument("--num_trials", nargs="?", const=20, type=int, default=20)
    parser.add_argument("--retrain", choices=["sgd", "sgd_only"])
    parser.add_argument("--epochs_re", type=int)
    parser.add_argument("--num_shadow", nargs="?", const=6, type=int, default=6)
    parser.add_argument("--lr", nargs="?", const=1e-5, type=float, default=1e-5)
    parser.add_argument(
        "--max_update_size", nargs="?", const=5000, type=int, default=5000
    )
    return parser.parse_args()


def main(args):
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

    if not os.path.exists("experiments/imdb/saved_BERT"):
        os.mkdir("experiments/imdb/saved_BERT")
    if not os.path.exists("experiments/imdb/saved_data"):
        os.mkdir("experiments/imdb/saved_data")

    # Experiment begins with pre-loaded models
    for data_seed in range(args.num_trials):
        np.random.seed(data_seed)
        shadow_model_id = np.random.randint(3, args.num_shadow + 2)

        orig_losses = np.load(
            f'experiments/imdb/saved_BERT/model_{shadow_model_id}_{args.nup}_{True if args.retrain == "sgd_only" else False}_orig_losses.npy'
        )
        losses = np.load(
            f'experiments/imdb/saved_BERT/model_{shadow_model_id}_{args.nup}_{True if args.retrain == "sgd_only" else False}_losses.npy'
        )
        loss_diffs = losses - orig_losses

        orig_losses_shadow_thresh = best_acc_thresh(orig_losses, args.nup)
        orig_losses_shadow_precision_thresh = best_precision_thresh(
            orig_losses, args.nup
        )

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

        for i in range(args.num_models):
            loaded_model = torch.load("experiments/imdb/saved_BERT/model_{}".format(i))
            all_dses = pickle.load(
                open("experiments/imdb/saved_BERT/dataset_{}".format(i), "rb")
            )
            (
                train_ds,
                update_dses,
                update_tests,
                test_ds,
                _,
                tokenizer,
                data_collator,
            ) = all_dses

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
            model.model = loaded_model

            all_initial_trn_acc.append(model.score(train_ds))
            print("Initial Training Accuracy: {}".format(all_initial_trn_acc[-1]))
            all_initial_tst_acc.append(model.score(test_ds))
            print("Initial Testing Accuracy: {}".format(all_initial_tst_acc[-1]))

            full_update_set_text = np.array(update_dses["text"])
            full_update_set_label = np.array(update_dses["label"])
            full_update_set_input_ids = np.array(update_dses["input_ids"])
            full_update_set_token_type_ids = np.array(update_dses["token_type_ids"])
            full_update_set_attention_mask = np.array(update_dses["attention_mask"])
            update_set_text = full_update_set_text[: args.nup]
            update_set_label = full_update_set_label[: args.nup]
            update_set_input_ids = full_update_set_input_ids[: args.nup]
            update_set_token_type_ids = full_update_set_token_type_ids[: args.nup]
            update_set_attention_mask = full_update_set_attention_mask[: args.nup]
            update_ds = datasets.Dataset.from_dict(
                {
                    "text": update_set_text,
                    "label": update_set_label,
                    "input_ids": update_set_input_ids,
                    "token_type_ids": update_set_token_type_ids,
                    "attention_mask": update_set_attention_mask,
                }
            )
            full_update = datasets.concatenate_datasets([train_ds, update_ds])

            rand_indices = np.random.choice(
                args.max_update_size, args.nup, replace=False
            )
            full_update_set_text_test = np.array(update_tests["text"])
            full_update_set_label_test = np.array(update_tests["label"])
            full_update_set_input_ids_test = np.array(update_tests["input_ids"])
            full_update_set_token_type_ids_test = np.array(
                update_tests["token_type_ids"]
            )
            full_update_set_attention_mask_test = np.array(
                update_tests["attention_mask"]
            )
            update_set_text_test = full_update_set_text_test[rand_indices]
            update_set_label_test = full_update_set_label_test[rand_indices]
            update_set_input_ids_test = full_update_set_input_ids_test[rand_indices]
            update_set_token_type_ids_test = full_update_set_token_type_ids_test[
                rand_indices
            ]
            update_set_attention_mask_test = full_update_set_attention_mask_test[
                rand_indices
            ]
            update_test_ds = datasets.Dataset.from_dict(
                {
                    "text": update_set_text_test,
                    "label": update_set_label_test,
                    "input_ids": update_set_input_ids_test,
                    "token_type_ids": update_set_token_type_ids_test,
                    "attention_mask": update_set_attention_mask_test,
                }
            )
            update_attack_ds = datasets.concatenate_datasets(
                [update_ds, update_test_ds]
            )

            orig_losses = model.get_loss(update_attack_ds)
            gap_no_update = gap_attack(
                model.predict(update_attack_ds), update_attack_ds["label"], args.nup
            )
            test_orig_losses = model.get_loss(test_ds)

            print(
                "\nUpdating Victim Model\n===================================================================\n"
            )
            if args.retrain == "sgd":
                model.fit_sgd(full_update, epochs=args.epochs_re)
            elif args.retrain == "sgd_only":
                model.fit_sgd(update_ds, epochs=args.epochs_re, sgd_only=True)
            else:
                raise NotImplementedError
            print(
                "===================================================================\n"
            )

            # finding fixed ratios
            losses = model.get_loss(update_attack_ds)

            test_losses = model.get_loss(test_ds)

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

            gap_acc = gap_attack(
                model.predict(update_attack_ds), update_attack_ds["label"], args.nup
            )

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

            all_post_trn_acc.append(model.score(train_ds))
            print("Post Update Training Accuracy: {}".format(all_post_trn_acc[-1]))
            all_post_tst_acc.append(model.score(test_ds))
            print("Post Update Testing Accuracy: {}".format(all_post_tst_acc[-1]))

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
            print(
                "Shadow: loss: {}, ratio: {}".format(loss_shadow_acc, ratio_shadow_acc)
            )
            all_losses_shadow.append(loss_shadow_acc)
            all_ratio_shadow.append(ratio_shadow_acc)
            print("=====================================================")
            print(
                "Precision: loss: {}, ratio: {}".format(loss_precision, ratio_precision)
            )
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

    save_tag = f"experiments/imdb/single_update/{args.nup}_25000_{args.retrain}_BERT"

    np.save(save_tag, save_data, allow_pickle=True)


if __name__ == "__main__":
    args = run_argparse()
    main(args)
