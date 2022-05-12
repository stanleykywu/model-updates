# epochs = 5 for sgd only, 2 for sgd full

import argparse
import os
import numpy as np
from utils import model_utils

import torch
import pickle
from datasets import concatenate_datasets, Dataset


def run_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--update_sizes", "--names-list", nargs="+", default=[20, 40, 80, 160, 320]
    )
    parser.add_argument(
        "--max_update_size", nargs="?", const=5000, type=int, default=5000
    )
    parser.add_argument("--num_shadow", nargs="?", const=6, type=int, default=6)
    parser.add_argument("--epochs_sgd_full", nargs="?", const=3, type=int, default=3)
    parser.add_argument("--epochs_sgd_new", nargs="?", const=6, type=int, default=6)
    return parser.parse_args()


def main(args):
    if not os.path.exists("experiments/imdb/saved_BERT"):
        os.mkdir("experiments/imdb/saved_BERT")
    if not os.path.exists("experiments/imdb/saved_data"):
        os.mkdir("experiments/imdb/saved_data")

    for i in range(3, args.num_shadow + 2):
        for update_size in args.update_sizes:
            for (sgd_only, epochs) in [
                (True, args.epochs_sgd_new),
                (False, args.epochs_sgd_full),
            ]:
                loaded_model = torch.load(
                    "experiments/imdb/saved_BERT/model_{}".format(i)
                )
                all_dses = pickle.load(
                    open("experiments/imdb/saved_BERT/dataset_{}".format(i), "rb")
                )
                (
                    train_ds,
                    update_dses,
                    update_tests,
                    _,
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

                full_update_set_text = np.array(update_dses["text"])
                full_update_set_label = np.array(update_dses["label"])
                full_update_set_input_ids = np.array(update_dses["input_ids"])
                full_update_set_token_type_ids = np.array(update_dses["token_type_ids"])
                full_update_set_attention_mask = np.array(update_dses["attention_mask"])
                rand_indices = np.random.choice(
                    args.max_update_size, update_size, replace=False
                )
                update_set_text = full_update_set_text[rand_indices]
                update_set_label = full_update_set_label[rand_indices]
                update_set_input_ids = full_update_set_input_ids[rand_indices]
                update_set_token_type_ids = full_update_set_token_type_ids[rand_indices]
                update_set_attention_mask = full_update_set_attention_mask[rand_indices]
                update_ds = Dataset.from_dict(
                    {
                        "text": update_set_text,
                        "label": update_set_label,
                        "input_ids": update_set_input_ids,
                        "token_type_ids": update_set_token_type_ids,
                        "attention_mask": update_set_attention_mask,
                    }
                )
                rand_indices = np.random.choice(
                    args.max_update_size, update_size, replace=False
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
                update_test_ds = Dataset.from_dict(
                    {
                        "text": update_set_text_test,
                        "label": update_set_label_test,
                        "input_ids": update_set_input_ids_test,
                        "token_type_ids": update_set_token_type_ids_test,
                        "attention_mask": update_set_attention_mask_test,
                    }
                )
                update_attack_ds = concatenate_datasets([update_ds, update_test_ds])
                full_update = concatenate_datasets([train_ds, update_ds])

                orig_losses = model.get_loss(update_attack_ds)

                if sgd_only:
                    model.fit_sgd(update_ds, epochs=epochs, sgd_only=sgd_only)
                else:
                    model.fit_sgd(full_update, epochs=epochs, sgd_only=sgd_only)

                losses = model.get_loss(update_attack_ds)

                np.save(
                    f"experiments/imdb/saved_BERT/model_{i}_{update_size}_{sgd_only}_orig_losses",
                    orig_losses,
                )
                np.save(
                    f"experiments/imdb/saved_BERT/model_{i}_{update_size}_{sgd_only}_losses",
                    losses,
                )


if __name__ == "__main__":
    args = run_argparse()
    main(args)
