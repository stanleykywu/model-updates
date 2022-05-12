import numpy as np
from tensorflow_privacy.privacy.analysis.rdp_accountant import compute_rdp
from tensorflow_privacy.privacy.analysis.rdp_accountant import get_privacy_spent
from utils.LIRA import *


def compute_epsilon(noise_multiplier, batch_size, steps, training_size):
    """Computes epsilon value for given hyperparameters."""
    if noise_multiplier == 0.0:
        return float("inf")
    orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
    sampling_probability = batch_size / training_size
    rdp = compute_rdp(
        q=sampling_probability,
        noise_multiplier=noise_multiplier,
        steps=steps,
        orders=orders,
    )

    return get_privacy_spent(orders, rdp, target_delta=1e-4)[0]


def gap_attack(predicted, actual, nup):
    preds = np.array([np.argmax(values) for values in predicted])
    correct_predictions = np.array(
        [True if x == y else False for (x, y) in zip(preds, actual)]
    )

    return (sum(correct_predictions[:nup]) + (nup - sum(correct_predictions[nup:]))) / (
        2 * nup
    )



def threshold(x_up, x_upt, thresh):
    up_acc = (x_up <= thresh).mean()
    upt_acc = (x_upt >= thresh).mean()

    acc = 0.5 * (up_acc + upt_acc)
    return acc



def high_precision(x_up, x_upt, nup):
    x_total = np.concatenate((x_up, x_upt))
    thresh = np.percentile(x_total, 10)

    predicted_positives = 0
    true_positives = 0
    for i, x in enumerate(x_total):
        if x < thresh:
            predicted_positives += 1
            if i < nup:
                true_positives += 1

    if predicted_positives == 0: return np.nan

    return true_positives / predicted_positives


def precision(x_total, nup, thresh):
    predicted_positives = 0
    true_positives = 0
    for i, x in enumerate(x_total):
        if x < thresh:
            predicted_positives += 1
            if i < nup:
                true_positives += 1

    if predicted_positives == 0: return np.nan

    return true_positives / predicted_positives


def recall(x_total, nup, thresh):
    predicted_positives = 0
    true_positives = 0
    for i, x in enumerate(x_total):
        if x < thresh:
            predicted_positives += 1
            if i < nup:
                true_positives += 1

    return true_positives / nup


def high_percententile(x_up, x_upt, nup):
    x_total = np.concatenate((x_up, x_upt))
    thresh = np.percentile(x_total, 10)

    return precision(x_total, nup, thresh), recall(x_total, nup, thresh)


def best_acc_thresh(values, nup):
    best_acc = 0
    best_thresh = None

    for v in values:
        if threshold(values[:nup], values[nup:], v) > best_acc:
            best_acc = threshold(values[:nup], values[nup:], v)
            best_thresh = v

    return best_thresh


def best_precision_thresh(values, nup, bound=0.1):
    best_precision = 0

    for v in values:
        prec, rc = precision(values, nup, v), recall(values, nup, v)
        
        if np.isnan(prec) or np.isnan(rc): continue

        if rc > bound:
            if prec > best_precision:
                best_precision = prec
                best_thresh = v

    return best_thresh


def LIRA_TF_scores(
    model,
    dataset_x,
    dataset_y,
    attack_x,
    attack_y,
    batch_size,
    num_training=5000,
    learning_rate=1e-4,
    epochs=5,
    num_shadow=25,
    save=True,
    use_saved_models=False,
    model_name=None,
):
    LIRA_attack = CarliniAttackUtilityKeras(
        model=model,
        data_x=dataset_x,
        data_y=dataset_y,
        num_shadow_models=num_shadow,
        save=save,
        use_saved_models=use_saved_models,
        model_name=model_name,
    )

    scores = LIRA_attack.offline_attack(
        attack_ds=(attack_x, attack_y),
        batch_size=batch_size,
        num_samples=num_training,
        training_epochs=epochs,
        lr=learning_rate
    )

    return np.array([score[0] for score in scores])





def back_front_attack(values, num_scs, nup, type="loss"):
    if type == "loss":
        back_front_indices = np.array(
            [
                (index, losses[num_scs] - losses[0])
                for index, losses in enumerate(values)
            ]
        )
    else:
        back_front_indices = np.array(
            [
                (index, (0.0005 + losses[num_scs]) / (0.0005 + losses[0]))
                for index, losses in enumerate(values)
            ]
        )

    total_correct = 0

    thresh = np.median([x[1] for x in back_front_indices])
    for i in range(2 * nup * num_scs):
        if (
            back_front_indices[i][1] <= thresh
            and back_front_indices[i][0] < nup * num_scs
        ):
            total_correct += 1
        elif (
            back_front_indices[i][1] >= thresh
            and back_front_indices[i][0] >= nup * num_scs
        ):
            total_correct += 1

    return total_correct / (2 * nup * num_scs)


def delta_attack(values, num_scs, nup, type="loss"):
    btwn_sc_losses = []
    sc_values = np.array([np.array(xi) for xi in values]).T
    for index, pt_losses in enumerate(sc_values):
        if index == 0:
            continue

        if type == "loss":
            btwn_sc_losses.append(np.subtract(sc_values[index - 1], pt_losses))
        else:
            btwn_sc_losses.append(
                [
                    (0.0005 + old) / (0.0005 + new)
                    for old, new in zip(sc_values[index - 1], pt_losses)
                ]
            )

    num_correct_in_general = 0
    num_correct_specific = 0
    total = num_scs * nup
    total_dist, num = 0, 0
    seen = set()

    for sc, sc_diffs in enumerate(btwn_sc_losses):
        aggr_pt_ids = [(pt_id, diff) for pt_id, diff in enumerate(sc_diffs)]
        points = sorted(aggr_pt_ids, key=lambda x: x[1], reverse=True)

        counter = 0

        if len(points) < nup:
            chosen = points
        else:
            chosen = []
            for pt_index, delta in points[:nup]:
                if pt_index in seen:
                    counter += 1
                else:
                    chosen.append((pt_index, delta))
            chosen += points[nup : nup + counter]

        for pt_index, delta in chosen:
            if pt_index < nup * num_scs:
                total_dist += abs(pt_index // nup - sc)
                num += 1
                if pt_index // nup == sc:
                    num_correct_specific += 1
                num_correct_in_general += 1
            seen.add(pt_index)

    return (
        num_correct_in_general / total,
        num_correct_specific / total,
        total_dist / num,
    )



        