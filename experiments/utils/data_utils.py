import numpy as np
import tensorflow.keras.datasets as ds
import random
from transformers import AutoTokenizer, DataCollatorWithPadding
import datasets as hf


def dataset_pop(dataset, ct):
    pop = dataset[:ct]
    popped = dataset[ct:]
    return pop, popped


def synthetic_regression(x_shape, noise, model_seed=0, data_seed=0):
    if data_seed != -1:
        np.random.seed(model_seed)
    true_w = np.random.normal(size=(x_shape[1],))

    if model_seed != -1:
        np.random.seed(data_seed)
    x = np.random.normal(size=x_shape)

    y = np.dot(x, true_w) + np.random.normal(0, noise, size=(x_shape[0],))
    return x, y, true_w


def mnist(
    num_updates,
    trn_size,
    up_size,
    data_seed=0,
    flatten=True,
    fashion="False",
    withold=50000,
    intensity=0,
):
    np.random.seed(data_seed)
    random.seed(data_seed)
    if fashion == "True":
        (trn_x, trn_y), (tst_x, tst_y) = ds.fashion_mnist.load_data()
    elif fashion == "cifar":
        (trn_x, trn_y), (tst_x, tst_y) = ds.cifar10.load_data()
        trn_y, tst_y = trn_y.ravel(), tst_y.ravel()
    else:
        (trn_x, trn_y), (tst_x, tst_y) = ds.mnist.load_data()

    complete_x = np.concatenate((trn_x, tst_x))
    complete_y = np.concatenate((trn_y, tst_y))
    complete_x = complete_x / 255.0 - 0.5
    if flatten:
        complete_x = complete_x.reshape((complete_x.shape[0], -1))
    elif len(complete_x.shape) == 3:
        complete_x = complete_x[:, :, :, None]

    # Withold for faster LIRA computation
    withold_x = complete_x[withold:]
    withold_y = complete_y[withold:]

    total_x = complete_x[:withold]
    total_y = complete_y[:withold]

    shffl_inds = np.random.choice(total_x.shape[0], total_x.shape[0], replace=False)
    train_inds = shffl_inds[:trn_size]
    train_ds = (total_x[train_inds], total_y[train_inds])

    update_dses = []
    update_tests = []
    for update in range(2 * num_updates):
        start_ind, end_ind = (
            trn_size + update * up_size,
            trn_size + (update + 1) * up_size,
        )
        update_inds = shffl_inds[start_ind:end_ind]

        if update >= num_updates:
            update_tests.append((total_x[update_inds], total_y[update_inds]))
        else:
            y = total_y[update_inds]
            update_dses.append((total_x[update_inds], y))

    test_inds = shffl_inds[end_ind:]  # woo hanging on to state
    test_ds = (total_x[test_inds], total_y[test_inds])

    return (train_ds, update_dses, update_tests, test_ds, (withold_x, withold_y))


def sample_biased(x_split, up_size, frac_per_class):
    num_cls = len(frac_per_class)
    new_x, new_y = [], []
    new_cls = np.random.choice(num_cls, up_size, p=frac_per_class)
    new_x_split = []

    for cls in range(num_cls):
        num_to_add = (new_cls == cls).sum()
        new_x.append(x_split[cls][0:num_to_add])
        new_y.append(cls * np.ones(num_to_add, dtype=np.int))
        new_x_split.append(x_split[cls][num_to_add:])

    new_x = np.concatenate(new_x)
    new_y = np.concatenate(new_y)

    new_inds = np.random.choice(new_x.shape[0], new_x.shape[0], replace=False)

    return (new_x[new_inds], new_y[new_inds]), new_x_split


def fashion_label_shift(
    num_updates,
    trn_size,
    up_size,
    ratios,
    targ_cls=1,
    same_distr=True,
    data_seed=0,
    flatten=True,
):
    train_ds, test_ds, update_tests, rest_x, rest_y = get_fashion_orig_distr(
        trn_size, up_size, num_updates, same_distr, data_seed, flatten
    )

    update_dses = []

    rest_x_split = [rest_x[rest_y == i] for i in range(10)]

    for update, ratio in zip(range(2 * num_updates), 2 * ratios):
        frac_per_class = [(1 - ratio) / 9 for _ in range(10)]
        frac_per_class[targ_cls] = ratio
        update_ds, rest_x_split = sample_biased(rest_x_split, up_size, frac_per_class)

        if update >= num_updates and same_distr:
            update_tests.append(update_ds)
        else:
            update_dses.append(update_ds)

    new_test_ds_x = np.concatenate([v[0] for v in update_tests])
    new_test_ds_y = np.concatenate([v[1] for v in update_tests])
    new_test_ds = (new_test_ds_x, new_test_ds_y)

    return train_ds, update_dses, update_tests, test_ds, new_test_ds


def fashion_covariate_shift(
    num_updates,
    trn_size,
    up_size,
    ratios,
    targ_cls=((0, 1), (8, 9)),
    same_distr=True,
    flatten=True,
):
    (trn_x, trn_y), (tst_x, tst_y) = ds.fashion_mnist.load_data()
    all_x = np.concatenate((trn_x, tst_x))
    all_y = np.concatenate((trn_y, tst_y))
    all_x = all_x / 255.0 - 0.5
    if flatten:
        all_x = all_x.reshape((all_x.shape[0], -1))
    elif len(all_x.shape) == 3:
        all_x = all_x[:, :, :, None]
    else:
        pass

    clses = targ_cls  # in each pair, drift from first to second

    x_split = []

    for cls_pair in clses:
        x_split_li = []
        for cls in cls_pair:
            cls_x = all_x[all_y == cls]
            cls_x_inds = np.random.choice(cls_x.shape[0], cls_x.shape[0], replace=False)
            x_split_li.append(cls_x[cls_x_inds])
        x_split.append(x_split_li)

    train_0, x_split[0][0] = dataset_pop(x_split[0][0], trn_size // 2)
    train_1, x_split[1][0] = dataset_pop(x_split[1][0], trn_size // 2)
    train_x = np.concatenate((train_0, train_1))
    train_y = np.concatenate((np.zeros(trn_size // 2), np.ones(trn_size // 2))).astype(
        np.int
    )

    test_0, x_split[0][0] = dataset_pop(x_split[0][0], trn_size // 2)
    test_1, x_split[1][0] = dataset_pop(x_split[1][0], trn_size // 2)
    test_x = np.concatenate((test_0, test_1))
    test_y = np.concatenate((np.zeros(trn_size // 2), np.ones(trn_size // 2))).astype(
        np.int
    )

    update_tests, update_dses = [], []

    if not same_distr:
        for update in range(num_updates):
            up_x_0, x_split[0][0] = dataset_pop(x_split[0][0], up_size // 2)
            up_x_1, x_split[1][0] = dataset_pop(x_split[1][0], up_size // 2)

            up_x = np.concatenate((up_x_0, up_x_1))
            up_y = np.concatenate(
                (
                    np.zeros(up_size // 2, dtype=np.int),
                    np.ones(up_size // 2, dtype=np.int),
                )
            )
            update_tests.append((up_x, up_y))

    for update, ratio in zip(range(2 * num_updates), 2 * ratios):
        up_x, up_y = [], []
        for i, cls_pair in enumerate(clses):
            num_to_add = np.random.binomial(up_size // 2, 1 - ratio)
            cur_up_x, x_split[i][0] = dataset_pop(x_split[i][0], num_to_add)
            up_x.append(cur_up_x)
            cur_up_x, x_split[i][1] = dataset_pop(
                x_split[i][1], up_size // 2 - num_to_add
            )
            up_x.append(cur_up_x)
            up_y.append(np.zeros(up_size // 2, dtype=np.int) + i)

        update_ds = np.concatenate(up_x), np.concatenate(up_y)

        if update >= num_updates:
            update_tests.append(update_ds)
        else:
            update_dses.append(update_ds)

    new_test_ds_x = np.concatenate([v[0] for v in update_tests])
    new_test_ds_y = np.concatenate([v[1] for v in update_tests])
    new_test_ds = new_test_ds_x, new_test_ds_y
    train_ds = train_x, train_y
    test_ds = test_x, test_y
    return train_ds, update_dses, update_tests, test_ds, new_test_ds


def get_fashion_orig_distr(
    trn_size, up_size, num_updates, same_distr=True, data_seed=0, flatten=True
):
    (trn_x, trn_y), (tst_x, tst_y) = ds.fashion_mnist.load_data()
    all_x = np.concatenate([trn_x, tst_x])
    all_y = np.concatenate([trn_y, tst_y])
    all_x = all_x / 255.0 - 0.5
    if flatten:
        all_x = all_x.reshape((all_x.shape[0], -1))
    elif len(all_x.shape) == 3:
        all_x = all_x[:, :, :, None]
    else:
        pass

    np.random.seed(data_seed)
    shffl_inds = np.random.choice(all_x.shape[0], all_x.shape[0], replace=False)
    train_inds = shffl_inds[:trn_size]
    test_inds = shffl_inds[trn_size : trn_size * 2]

    train_ds = (all_x[train_inds], all_y[train_inds])
    test_ds = (all_x[test_inds], all_y[test_inds])

    end_ind = trn_size * 2

    update_tests = []

    if not same_distr:  # sample test updates from original distr
        for update in range(num_updates):
            start_ind, end_ind = end_ind, end_ind + up_size
            update_inds = shffl_inds[start_ind:end_ind]
            update_tests.append((all_x[update_inds], all_y[update_inds]))

    rest_inds = shffl_inds[end_ind:]
    rest_x = all_x[rest_inds]
    rest_y = all_y[rest_inds]
    return train_ds, test_ds, update_tests, rest_x, rest_y


def purchase(
    num_updates,
    trn_size,
    up_size,
    data_seed=0,
    withold=50000,
):
    np.random.seed(data_seed)
    random.seed(data_seed)
    fea_path = "/net/data/pois_priv/data/p100/p100_fea.npy"
    lab_path = "/net/data/pois_priv/data/p100/p100_lab.npy"
    x = np.load(fea_path)
    y = np.load(lab_path)

    withold_x = x[withold:]
    withold_y = y[withold:]

    x = x[:withold]
    y = y[:withold]

    shffl_inds = np.random.choice(x.shape[0], x.shape[0], replace=False)
    train_inds = shffl_inds[:trn_size]
    train_ds = (x[train_inds], y[train_inds])

    update_dses = []
    update_tests = []
    for update in range(2 * num_updates):
        start_ind, end_ind = (
            trn_size + update * up_size,
            trn_size + (update + 1) * up_size,
        )
        update_inds = shffl_inds[start_ind:end_ind]

        if update >= num_updates:
            update_tests.append((x[update_inds], y[update_inds]))
        else:
            cur_y = y[update_inds]
            update_dses.append((x[update_inds], cur_y))

    test_inds = shffl_inds[end_ind:]  # woo hanging on to state
    test_ds = (x[test_inds], y[test_inds])
    return (train_ds, update_dses, update_tests, test_ds, (withold_x, withold_y))


def imdb(
    num_updates,
    trn_size,
    up_size,
    withold=40000,
    data_seed=0,
):
    np.random.seed(data_seed)
    random.seed(data_seed)
    imdb = hf.load_dataset("imdb")
    trn, tst = imdb["train"], imdb["test"]
    complete_text = trn["text"] + tst["text"]
    complete_label = trn["label"] + tst["label"]

    withold_x = complete_text[withold:]
    withold_y = complete_label[withold:]
    withold_ds = hf.Dataset.from_dict({"text": withold_x, "label": withold_y})

    all_text = complete_text[:withold]
    all_label = complete_label[:withold]

    shffl_inds = np.random.choice(len(all_text), len(all_text), replace=False)
    train_inds = shffl_inds[:trn_size]
    train_ds = ([all_text[i] for i in train_inds], [all_label[i] for i in train_inds])
    train_ds = hf.Dataset.from_dict({"text": train_ds[0], "label": train_ds[1]})

    update_dses = []
    update_tests = []
    for update in range(2 * num_updates):
        start_ind, end_ind = (
            trn_size + update * up_size,
            trn_size + (update + 1) * up_size,
        )
        update_inds = shffl_inds[start_ind:end_ind]

        if update >= num_updates:
            update_tests.append(
                hf.Dataset.from_dict(
                    {
                        "text": [all_text[i] for i in update_inds],
                        "label": [all_label[i] for i in update_inds],
                    }
                )
            )
        else:
            y = [all_label[i] for i in update_inds]
            update_dses.append(
                hf.Dataset.from_dict(
                    {
                        "text": [all_text[i] for i in update_inds],
                        "label": y,
                    }
                )
            )

    test_inds = shffl_inds[end_ind:]  # woo hanging on to state
    test_ds = ([all_text[i] for i in test_inds], [all_label[i] for i in test_inds])
    test_ds = hf.Dataset.from_dict({"text": test_ds[0], "label": test_ds[1]})

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    dataset_dict = hf.DatasetDict()
    dataset_dict["train"] = train_ds
    dataset_dict["update"] = update_dses[0]
    dataset_dict["update_test"] = update_tests[0]
    dataset_dict["test"] = test_ds
    dataset_dict["withold"] = withold_ds

    tokenized_dataset = dataset_dict.map(preprocess_function, batched=True)

    return (
        tokenized_dataset["train"],
        tokenized_dataset["update"],
        tokenized_dataset["update_test"],
        tokenized_dataset["test"],
        tokenized_dataset["withold"],
        tokenizer,
        data_collator,
    )


def sample_biased(x_split, up_size, frac_per_class):
    num_cls = len(frac_per_class)
    new_x, new_y = [], []
    new_cls = np.random.choice(num_cls, up_size, p=frac_per_class)
    new_x_split = []

    for cls in range(num_cls):
        num_to_add = (new_cls == cls).sum()
        new_x.append(x_split[cls][0:num_to_add])
        new_y.append(cls * np.ones(num_to_add, dtype=np.int))
        new_x_split.append(x_split[cls][num_to_add:])

    new_x = np.concatenate(new_x)
    new_y = np.concatenate(new_y)

    new_inds = np.random.choice(new_x.shape[0], new_x.shape[0], replace=False)

    return (new_x[new_inds], new_y[new_inds]), new_x_split
