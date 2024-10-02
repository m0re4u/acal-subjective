import copy
import json
import logging
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, f1_score
from transformers import TrainerCallback


def torch_kl(output, target):
    """
    Compute KL Divergence between model output and target distribution.
    """
    kldiv = F.kl_div(output.log(), target, None, None, "sum")
    return kldiv


def jsd(output, target, reduction="batchmean"):
    """
    Compute Jensen-Shannon Divergence between model output and target distribution.
    """
    kl = torch.nn.KLDivLoss(reduction=reduction, log_target=True)
    output = output.view(-1, output.size(-1))
    target = target.view(-1, target.size(-1))
    m = (0.5 * (output + target)).log()
    loss = 0.5 * (kl(m, output.log()) + kl(m, target.log()))
    return loss


def compute_hard_metrics(labels, logits):
    # Evaluate normally with majority votes
    labels = torch.argmax(labels, dim=1)
    logits = torch.argmax(logits, dim=1)

    return classification_report(
        labels.numpy(), logits.numpy(), output_dict=True, zero_division=0.0
    )


def process_logits(logits):
    # Applying softmax to both labels and logits is the way to learn with soft labels
    # https://www.jair.org/index.php/jair/article/view/12752
    if isinstance(logits, tuple):
        logits = logits[0]

    outputs = torch.from_numpy(logits)
    outputs = F.softmax(outputs, dim=1)

    return outputs


def normalize_labels(labels, label_normalize_strategy):
    # Applying softmax to the labels seems emperically to be a bad idea. Rather, we can
    # alternatively stabilize the labels by adding a small epsilon and then normalizing.
    if label_normalize_strategy == "softmax":
        labels = torch.from_numpy(labels)
        labels = F.softmax(labels, dim=1)
    elif label_normalize_strategy == "epsilon":
        labels = torch.from_numpy(labels)
        labels = labels + 1e-12
        labels = labels / torch.sum(labels, dim=1).unsqueeze(1)
    else:
        labels = torch.from_numpy(labels)

    return labels


def make_one_hot_targets(item, label_normalize_strategy):
    one_hot_targets = np.eye(item.num_labels)[np.array(item.raw_labels, dtype=int)]
    one_hot_targets = normalize_labels(one_hot_targets, label_normalize_strategy)
    return one_hot_targets


def compute_metrics(eval_pred, label_normalize_strategy):
    logits, labels = eval_pred

    outputs = process_logits(logits)
    labels = normalize_labels(labels, label_normalize_strategy)

    kldiv = torch_kl(outputs, labels)

    jsdiv = jsd(outputs, labels)

    hard_metrics = compute_hard_metrics(labels, outputs)

    return {"kldiv": kldiv, "jsdiv": jsdiv, **hard_metrics}


def compute_metrics_passive(eval_pred, eval_data, label_normalize_strategy):
    logits, labels = eval_pred

    outputs = process_logits(logits)
    labels = normalize_labels(labels, label_normalize_strategy)

    kldiv = torch_kl(outputs, labels)

    jsdiv = jsd(outputs, labels)

    hard_metrics = compute_hard_metrics(labels, outputs)

    additional_eval = additional_evaluations(
        logits,
        eval_data.items,
        label_normalize_strategy,
    )
    prefix = "eval"
    results = {"kldiv": kldiv, "jsdiv": jsdiv, **hard_metrics}
    # add all annot-centric results
    results[f"{prefix}_worst_jsdiv_per_item"] = additional_eval[
        "worst_jsdiv_per_item"
    ]["mean"]
    results[f"{prefix}_worst_jsdiv_per_annotator"] = additional_eval[
        "worst_jsdiv_per_annotator"
    ]["worst_jsdiv"]
    results[f"{prefix}_worst_mean_jsdiv_per_annotator"] = additional_eval[
        "worst_jsdiv_per_annotator"
    ]["mean"]
    results[f"{prefix}_worst_f1_macro_per_annotator"] = additional_eval[
        "f1_per_annotator"
    ]["worst_f1"]
    results[f"{prefix}_mean_f1_macro_per_annotator"] = additional_eval[
        "f1_per_annotator"
    ]["mean"]
    return results


def worst_jsdiv_per_item(outputs, labeled_items, label_normalize_strategy):
    # Loop through each item and compute the JS Divergence between the model's output
    # and the one-hot encoded annotations. Return the highest JS divergence and the
    # annotators who have the highest JS divergence.
    results = torch.empty(size=(len(labeled_items), 1))
    if isinstance(outputs, torch.Tensor):
        outputs = outputs.clone().detach()
    else:
        outputs = torch.from_numpy(outputs)
    worst_annotators = []
    for i, (output, item) in enumerate(zip(outputs, labeled_items)):
        one_hot_targets = make_one_hot_targets(item, label_normalize_strategy)
        output = output.unsqueeze(0)
        zero_dim = one_hot_targets.size(0)
        output = output.repeat(*(zero_dim, 1))

        # compute jsdiv for all annotations in the item
        jsdiv = jsd(output, one_hot_targets, reduction="none").sum(dim=(-1))

        # return the worst jsdiv
        results[i] = torch.max(jsdiv)

        # select all annotators whose jsdiv is the highest
        worst_annotators.append(
            [item.annotators[i] for i in torch.where(jsdiv == torch.max(jsdiv))[0]]
        )

    worst_jsdivs = [result[0] for result in results.tolist()]  # flatten the tensor

    return {
        "worst_annotators": worst_annotators,
        "worst_jsdivs": worst_jsdivs,
        "mean": np.mean(worst_jsdivs),
    }


def worst_jsdiv_per_annotator(outputs, labeled_items, label_normalize_strategy):
    annotator2labels = defaultdict(list)
    annotator2predictions = defaultdict(list)

    for predicted_label, item in zip(outputs, labeled_items):
        annotators = item.annotators
        one_hot_targets = make_one_hot_targets(item, label_normalize_strategy)
        for annotator, one_hot_target in zip(annotators, one_hot_targets):
            annotator2labels[annotator].append(one_hot_target)
            annotator2predictions[annotator].append(predicted_label)
    results = torch.empty(size=(len(annotator2labels), 1))
    for i, annotator in enumerate(annotator2labels):
        labels = torch.vstack(annotator2labels[annotator])
        predictions = torch.vstack(annotator2predictions[annotator])
        results[i] = jsd(predictions, labels)

    all_jsdivs = [result[0] for result in results.tolist()]  # flatten the tensor

    worst_jsdiv = max(all_jsdivs)
    worst_annotators = [
        annotator
        for annotator, jsdiv in zip(annotator2labels, all_jsdivs)
        if jsdiv == worst_jsdiv
    ]

    return {
        "worst_annotators": worst_annotators,
        "worst_jsdiv": worst_jsdiv,
        "mean": np.mean(all_jsdivs),
    }


def annotator_based_f1(outputs, labeled_items):
    predicted_labels = torch.argmax(outputs, dim=1).numpy()
    annotator2labels = defaultdict(list)
    annotator2predictions = defaultdict(list)

    for predicted_label, item in zip(predicted_labels, labeled_items):
        annotator_labels = item.raw_labels
        annotators = item.annotators
        for annotator, label in zip(annotators, annotator_labels):
            annotator2labels[annotator].append(label)
            annotator2predictions[annotator].append(predicted_label)
    all_f1_scores = []
    for annotator in annotator2labels:
        f1_score_annotator = f1_score(
            y_true=annotator2labels[annotator],
            y_pred=annotator2predictions[annotator],
            average="macro",
        )
        all_f1_scores.append(f1_score_annotator)

    worst_f1 = min(all_f1_scores)
    worst_annotators = [
        annotator
        for annotator, f1 in zip(annotator2labels, all_f1_scores)
        if f1 == worst_f1
    ]

    return {
        "worst_annotators": worst_annotators,
        "worst_f1": worst_f1,
        "mean": np.mean(all_f1_scores),
    }


def additional_evaluations(logits, labeled_items, label_normalize_strategy):
    assert len(logits) == len(labeled_items), (
        f"Number of predictions ({len(logits)}) does not match number of items in "
        f"evaluation dataset ({len(labeled_items)})"
    )
    outputs = process_logits(logits)
    results = {}
    results["worst_jsdiv_per_item"] = worst_jsdiv_per_item(
        outputs, labeled_items, label_normalize_strategy
    )
    results["worst_jsdiv_per_annotator"] = worst_jsdiv_per_annotator(
        outputs, labeled_items, label_normalize_strategy
    )
    results["f1_per_annotator"] = annotator_based_f1(outputs, labeled_items)

    return results


class CustomPassiveLoggingCallback(TrainerCallback):
    def __init__(self, annotations_per_epoch, samples_per_epoch, log_filename=None):
        self.annotations_per_epoch = annotations_per_epoch
        self.samples_per_epoch = samples_per_epoch
        self.num_unique_annotations = 0
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(logging.StreamHandler(sys.stdout))
        if log_filename is not None:
            self.logger.addHandler(logging.FileHandler(log_filename))
        else:
            self.logger.addHandler(logging.FileHandler("passive_learning.log"))

    def on_evaluate(self, args, state, control, logs=None, **kwargs):
        # assume metric_key_prefix based on the first metric key
        metric_key_prefix = list(kwargs["metrics"].keys())[0].split("_")[0]
        last_log = state.log_history[-1]
        num_annotations = last_log[f"{metric_key_prefix}_total_annotations_seen"]
        num_samples = state.global_step * args.per_device_train_batch_size
        if state.epoch is None or state.epoch <= 1 and num_annotations > 0:
            self.num_unique_annotations = copy.copy(num_annotations)
        else:
            self.num_unique_annotations = self.annotations_per_epoch

        log_dict = {
            "results": {**last_log},
            "counters": {
                "total_annotations_seen": num_annotations,
                "total_samples_seen": num_samples,
                "total_epochs_trained": state.epoch,
                "total_unique_annotations_seen": self.num_unique_annotations,
            },
        }
        self.logger.info(json.dumps(log_dict))
