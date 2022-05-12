import torch
from utils import model_utils
import argparse
import numpy as np
import pickle
import datasets

from utils.attack_utils import *


def run_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nup", nargs="?", const=100, type=int, default=100)
    parser.add_argument("--num_trials", nargs="?", const=20, type=int, default=20)
    parser.add_argument("--retrain", choices=["sgd", "sgd_only"])
    parser.add_argument("--epochs_re", type=int)
    parser.add_argument("--num_updates", nargs="?", const=10, type=int, default=10)
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

    if not os.path.exists("experiments/imdb/saved_BERT"):
        os.mkdir("experiments/imdb/saved_BERT")
    if not os.path.exists("experiments/imdb/saved_data"):
        os.mkdir("experiments/imdb/saved_data")

    for data_seed in range(args.num_trials):
        for i in range(3):
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
            )
            model.model = loaded_model

            total_text = np.array(update_dses["text"])
            total_label = np.array(update_dses["label"])
            total_input_ids = np.array(update_dses["input_ids"])
            total_token_type_ids = np.array(update_dses["token_type_ids"])
            total_attention_mask = np.array(update_dses["attention_mask"])
            rand_indices = np.random.choice(
                len(total_text), args.num_updates * args.nup * 2, replace=False
            )
            update_set_text = total_text[rand_indices]
            update_set_label = total_label[rand_indices]
            update_set_input_ids = total_input_ids[rand_indices]
            update_set_token_type_ids = total_token_type_ids[rand_indices]
            update_set_attention_mask = total_attention_mask[rand_indices]
            total = [
                {
                    "text": text,
                    "label": label,
                    "input_id": input_id,
                    "token_type_id": token_type_id,
                    "attention_mask": attention_mask,
                }
                for (text, label, input_id, token_type_id, attention_mask) in zip(
                    update_set_text,
                    update_set_label,
                    update_set_input_ids,
                    update_set_token_type_ids,
                    update_set_attention_mask,
                )
            ]

            print("Seed/model: {}/{}".format(data_seed, i))

            update_dses = [
                total[args.nup * u : args.nup * (u + 1)]
                for u in range(args.num_updates)
            ]
            offset = args.num_updates * args.nup
            update_tests = [
                total[offset + args.nup * u : offset + args.nup * (u + 1)]
                for u in range(args.num_updates)
            ]

            accuracies[0]["training"].append(model.score(train_ds))
            accuracies[0]["testing"].append(model.score(test_ds))

            res = [[] for _ in range(args.nup * args.num_updates * 2)]

            for index, (update_ds, update_test) in enumerate(
                zip(update_dses, update_tests)
            ):
                update_ds = datasets.Dataset.from_dict(
                    {
                        "text": [data["text"] for data in update_ds],
                        "label": [data["label"] for data in update_ds],
                        "input_ids": [data["input_id"] for data in update_ds],
                        "token_type_ids": [data["token_type_id"] for data in update_ds],
                        "attention_mask": [
                            data["attention_mask"] for data in update_ds
                        ],
                    }
                )
                update_test_ds = datasets.Dataset.from_dict(
                    {
                        "text": [data["text"] for data in update_test],
                        "label": [data["label"] for data in update_test],
                        "input_ids": [data["input_id"] for data in update_test],
                        "token_type_ids": [
                            data["token_type_id"] for data in update_test
                        ],
                        "attention_mask": [
                            data["attention_mask"] for data in update_test
                        ],
                    }
                )

                for pt_id, loss in enumerate(model.get_loss(update_ds)):
                    res[index * args.nup + pt_id].append(loss)

                for pt_id, loss in enumerate(model.get_loss(update_test_ds)):
                    res[args.num_updates * args.nup + index * args.nup + pt_id].append(
                        loss
                    )

            full_update = train_ds
            for index, (update_ds, update_test) in enumerate(
                zip(update_dses, update_tests)
            ):
                print("update ", index + 1)
                update_ds = datasets.Dataset.from_dict(
                    {
                        "text": [data["text"] for data in update_ds],
                        "label": [data["label"] for data in update_ds],
                        "input_ids": [data["input_id"] for data in update_ds],
                        "token_type_ids": [data["token_type_id"] for data in update_ds],
                        "attention_mask": [
                            data["attention_mask"] for data in update_ds
                        ],
                    }
                )
                update_test_ds = datasets.Dataset.from_dict(
                    {
                        "text": [data["text"] for data in update_test],
                        "label": [data["label"] for data in update_test],
                        "input_ids": [data["input_id"] for data in update_test],
                        "token_type_ids": [
                            data["token_type_id"] for data in update_test
                        ],
                        "attention_mask": [
                            data["attention_mask"] for data in update_test
                        ],
                    }
                )

                full_update = datasets.concatenate_datasets([full_update, update_ds])

                if args.retrain == "sgd":
                    model.fit_sgd(full_update, epochs=args.epochs_re)
                elif args.retrain == "sgd_only":
                    model.fit_sgd(update_ds, epochs=args.epochs_re, sgd_only=True)
                else:
                    raise NotImplementedError

                temp_ds = np.concatenate([data for data in update_dses])
                temp_ds = datasets.Dataset.from_dict(
                    {
                        "text": [data["text"] for data in temp_ds],
                        "label": [data["label"] for data in temp_ds],
                        "input_ids": [data["input_id"] for data in temp_ds],
                        "token_type_ids": [data["token_type_id"] for data in temp_ds],
                        "attention_mask": [data["attention_mask"] for data in temp_ds],
                    }
                )
                for pt_id, loss in enumerate(model.get_loss(temp_ds)):
                    res[pt_id].append(loss)

                temp_ds = np.concatenate([data for data in update_tests])
                temp_ds = datasets.Dataset.from_dict(
                    {
                        "text": [data["text"] for data in temp_ds],
                        "label": [data["label"] for data in temp_ds],
                        "input_ids": [data["input_id"] for data in temp_ds],
                        "token_type_ids": [data["token_type_id"] for data in temp_ds],
                        "attention_mask": [data["attention_mask"] for data in temp_ds],
                    }
                )
                for pt_id, loss in enumerate(model.get_loss(temp_ds)):
                    res[args.num_updates * args.nup + pt_id].append(loss)

                accuracies[index + 1]["training"].append(model.score(train_ds))
                accuracies[index + 1]["testing"].append(model.score(test_ds))

                temp_data = np.concatenate(
                    (
                        res[: (index + 1) * args.nup],
                        res[
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
                back_front_accuracy_loss = back_front_attack(
                    temp_data, index + 1, args.nup
                )
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
                acc = []
                for subset in update_dses[: index + 1]:
                    for dt in subset:
                        acc.append(dt)

                for subset in update_tests[: index + 1]:
                    for dt in subset:
                        acc.append(dt)

                aux_ds = datasets.Dataset.from_dict(
                    {
                        "text": [data["text"] for data in acc],
                        "label": [data["label"] for data in acc],
                        "input_ids": [data["input_id"] for data in acc],
                        "token_type_ids": [data["token_type_id"] for data in acc],
                        "attention_mask": [data["attention_mask"] for data in acc],
                    }
                )
                total_data[index]["gap"].append(
                    gap_attack(
                        model.predict(aux_ds),
                        aux_ds["label"],
                        args.nup * (index + 1),
                    )
                )
                total_data[index]["avg_dist"].append((avg_dist_loss, avg_dist_ratio))
                print(
                    "----------------------------------------------------------------"
                )

    np.save(
        f"experiments/imdb/multiple_update/{args.num_updates}_{args.nup}_{args.retrain}",
        (total_data, accuracies),
    )


if __name__ == "__main__":
    args = run_argparse()
    main(args)
