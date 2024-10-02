import random

import torch
from transformers import Trainer


def randomly_select_datapoints(unlabelled_dataset, n):
    """
    Queries n random data points from the unlabelled dataset and returns them.

    Args:
        n (int): The number of data points to query.

    Returns:
        list: A list of n UnlabelledItem instances, randomly sampled or k items if k < n and k is the number of items
        that is left in the dataset.
        @param unlabelled_dataset: an UnlabelledDataset object that contains 'items' attribute for each instance.
    """
    if n > len(unlabelled_dataset.items):
        # return all items that are available
        print(f"Only {len(unlabelled_dataset.items)} items are available")
        return unlabelled_dataset.items

    return random.sample(unlabelled_dataset.items, n)


def select_datapoints_by_uncertainty(
        unlabelled_to_predict, unlabelled_dataset, trainer, n
):
    """
    Queries n data points from the unlabelled dataset based on model uncertainty and returns them.

    Args:
        unlabelled_dataset (UnlabelledDataset): An unlabelled dataset object that contains 'items' attribute for each instance.
        model (Model): A task-specific model that can be used to query uncertainty.
        n (int): The number of data points to query.

    Returns:
        list: A list of n UnlabelledItem instances, selected based on uncertainty, or k items if k < n and k is the number of items left in the dataset.
    """
    if n > len(unlabelled_dataset.items):
        print(f"Only {len(unlabelled_dataset.items)} items are available")
        return unlabelled_dataset.items

    logits = trainer.predict(unlabelled_to_predict).label_ids
    # convert logits to tensor
    logits = torch.tensor(logits)
    # compute entropy for each element in logits
    entropy = torch.distributions.Categorical(logits=logits).entropy()
    # get the indices with the highest entropy
    _, indices = torch.topk(entropy, n)
    # get the items with the highest entropy
    selected_items_unlablled_to_predict = [unlabelled_to_predict[i] for i in indices]
    selected_items_indices = [
        item["index"] for item in selected_items_unlablled_to_predict
    ]
    # get the items at the indices from unlabeled dataset
    selected_items = [
        unlabelled_dataset.get_item_by_index(index) for index in selected_items_indices
    ]

    return selected_items
