import random
import copy

import numpy as np
import torch
from torch.nn import CosineSimilarity

from tqdm import tqdm


def randomly_select_annotations(sampled_items):
    """
    Randomly selects an annotation and annotator for each item in the sampled_items list and returns them.
    @param sampled_items: a list of UnlabelledItem instances that are selected from the unlabelled dataset.
    @return: a dictionary with keys "annotations", "annotators", and "annotation_indices". The values are lists of
    annotations, annotators, and annotation indices that correspond to the selected items. The annotation indices
    correspond to the current indices of the annotations that are left in the unlabelled dataset.
    """
    indices, annotations, annotators, annotation_identifier = [], [], [], []

    for item in sampled_items:
        # Skip items with no annotations
        if not item.annotations:
            print(f"Item with index {item.data_id} has no annotations")
            continue

        # Randomly select an annotation index from the list of annotations that are still available for that
        # specific item.
        random_index = random.randint(0, len(item.annotations) - 1)
        indices.append(random_index)

        # Retrieve and store the corresponding annotator and annotation
        annotator = item.annotators[random_index]
        annotation = item.annotations[random_index]
        annotation_id = item.annotation_ids[random_index]
        annotations.append(annotation)
        annotators.append(annotator)
        annotation_identifier.append(annotation_id)

    return {
        "annotators": annotators,
        "annotations": annotations,
        "annotation_indices": indices,
        "annotation_ids": annotation_identifier,
    }


def select_n_or_all(sampled_items, n=None):
    """
    Selects several (or all) available annotations for each item in the sampled_items list and returns them.
    If n is specified, it will select n random annotations for each item. If n is not specified, it will select all
    available annotations for each item.
    :param sampled_items: a list of UnlabelledItem instances that are selected from the unlabelled dataset.
    :param n: the number of annotations to select for each item. If n is None, all available annotations are selected.
    """

    indices, annotations, annotators, annotation_identifier = [], [], [], []

    for item in sampled_items:
        # Skip items with no annotations
        if not item.annotations:
            print(f"Item with index {item.data_id} has no annotations")
            continue

        number_current_annotations = len(item.annotations)
        if n > number_current_annotations or n < 0 or n is None:
            n_to_select = number_current_annotations
        else:
            n_to_select = copy.copy(n)

        # Randomly select n annotation indices from the list of annotations that are still available for that
        # specific item.
        random_indices = random.sample(range(number_current_annotations), n_to_select)
        indices.append(random_indices)
        annotators.append([item.annotators[i] for i in random_indices])
        annotations.append([item.annotations[i] for i in random_indices])
        annotation_identifier.append([item.annotation_ids[i] for i in random_indices])

    return {
        "annotators": annotators,
        "annotations": annotations,
        "annotation_indices": indices,
        "annotation_ids": annotation_identifier,
    }


def get_annotators_with_lowest_cosim_fast(
    annotators_with_history,
    annotator2history_embeddings,
    item2embedding,
    item,
    cos_sim_scorer,
):
    average_semantic_similarity_scores = []

    # Get the embedding of the current item to be annotated
    item_embedding = item2embedding[item.data_id].reshape(1, -1)

    for annot in annotators_with_history:
        # Get all embeddings of items that annotator has annotated
        all_annotated_items_embeddings = annotator2history_embeddings[annot]

        # Compute the cosine similarity between the current item and all items the
        # annotator has annotated
        semantic_similarities = cos_sim_scorer(
            torch.from_numpy(all_annotated_items_embeddings),
            torch.from_numpy(item_embedding),
        )
        average_semantic_similarity_scores.append((annot, semantic_similarities.mean()))

    # Select the annotator that has annotated on average the least similar texts
    selected_annotator, _ = min(average_semantic_similarity_scores, key=lambda x: x[1])
    annotator_index = item.annotators.index(selected_annotator)
    annotation = item.annotations[annotator_index]
    annotation_id = item.annotation_ids[annotator_index]
    return annotator_index, annotation, selected_annotator, annotation_id


def select_based_on_semantic_similarity(
    sampled_items, labelled_train_dataset, unlabelled_dataset, device
):
    """
    This method picks the annotator based on semantic similarity. If our annotators have already annotated something, we
    can check for each new item, for which annotator the semantics / this type of text is comparably new. Then we choose
    the annotator that has annotated on average the least similar items.
    @param sampled_items: the items that are sampled from the unlabelled dataset
    @param labelled_train_dataset: a LabelledDataset instance that contains the labelled items
    @return: a dictionary with keys "annotations", "annotators", and "annotation_indices". The values are lists of
    annotations, annotators, and annotation indices that correspond to the selected items. The annotation indices
    correspond to the current indices of the annotations that are left in the unlabelled dataset.
    """
    # iterate over items in sampled items, check if there are annotators that have already seen someting, if yes, select
    # the annotator that has annotated the least similar item, if not, select randomly.
    indices, annotations, annotators, annotation_identifier = [], [], [], []

    cos_sim = CosineSimilarity().to(device)

    annotators_already_used = [item.annotators for item in labelled_train_dataset.items]
    annotators_already_used = set(
        [item for sublist in annotators_already_used for item in sublist]
    )
    data_id2embedding = unlabelled_dataset.text2semantic_representation
    annotato2items = labelled_train_dataset.get_annotator2items()
    if annotators_already_used:
        annotator2history_embeddings = {
            annotator: np.stack(
                [data_id2embedding[item.index] for item in annotato2items[annotator]]
            )
            for annotator in annotators_already_used
        }

    for item in tqdm(sampled_items, desc="Computing semantic similarity"):
        # Get the annotators that have already annotated something
        item_annotators = item.annotators
        item_annotator_set = set(item_annotators)
        annotators_with_history = item_annotator_set.intersection(
            annotators_already_used
        )

        # If there is one (or more) annotator(s) that have not yet annotated anything, select one
        # of them randomly
        if len(annotators_with_history) != len(item_annotator_set):
            annotators_without_history = item_annotator_set.difference(
                annotators_with_history
            )
            selected_annotator = random.choice(list(annotators_without_history))
            annotator_index = item_annotators.index(selected_annotator)
            annotation = item.annotations[annotator_index]
            annotation_id = item.annotation_ids[annotator_index]

        # If all annotators have already annotated, select the annotator that has annotated the
        # least similar item
        else:
            (
                annotator_index,
                annotation,
                selected_annotator,
                annotation_id,
            ) = get_annotators_with_lowest_cosim_fast(
                annotators_with_history=annotators_with_history,
                annotator2history_embeddings=annotator2history_embeddings,
                item2embedding=data_id2embedding,
                item=item,
                cos_sim_scorer=cos_sim,
            )

        indices.append(annotator_index)
        annotations.append(annotation)
        annotators.append(selected_annotator)
        annotation_identifier.append(annotation_id)

    return {
        "annotators": annotators,
        "annotations": annotations,
        "annotation_indices": indices,
        "annotation_ids": annotation_identifier,
    }


def select_based_on_minority_label(sampled_items, labelled_train_dataset):
    """
    This method takes the labels into account that annotators provided. For a new item, pick one of the annotators that
    has a label bias towards the minority label.
    @param sampled_items: the items that are sampled from the unlabelled dataset
    @param labelled_train_dataset: the labelled items
    @return:
    """
    indices, annotations, annotators, annotation_identifier = [], [], [], []
    annotators_already_used = [item.annotators for item in labelled_train_dataset.items]
    annotators_already_used = set(
        [item for sublist in annotators_already_used for item in sublist]
    )
    # get the current minority label. get all annotations from the labelled_train_dataset.
    all_annotations = [item.raw_labels for item in labelled_train_dataset.items]
    all_annotations = [item for sublist in all_annotations for item in sublist]
    # get the minority label
    minority_label = min(all_annotations, key=all_annotations.count)

    for item in sampled_items:
        item_annotators = item.annotators
        annotators_with_history = set(item_annotators).intersection(
            annotators_already_used
        )

        # If there is one (or more) annotator(s) that have not yet annotated anything, select one of them randomly
        if annotators_with_history != set(item_annotators):
            annotators_without_history = set(item_annotators).difference(
                annotators_with_history
            )
            selected_annotator = random.choice(list(annotators_without_history))
            annotator_index = item_annotators.index(selected_annotator)
            annotation = item.annotations[annotator_index]
            annotation_id = item.annotation_ids[annotator_index]

        # If all annotators have already annotated, select the annotator that has annotated the least similar item
        else:
            # Pre-set max similarity to 0, as we want to maximize it
            max_minority_bias = 0.0

            for annotator in annotators_with_history:
                # for each item get the annotation at the index of the annotator index in the list
                annotation_history = [
                    item.raw_labels[item.annotators.index(annotator)]
                    for item in labelled_train_dataset.items
                    if annotator in item.annotators
                ]
                # get relative frequency of minority label
                relative_frequency = annotation_history.count(minority_label) / len(
                    annotation_history
                )
                if relative_frequency >= max_minority_bias:
                    max_minority_bias = relative_frequency
                    selected_annotator = annotator
                    annotator_index = item_annotators.index(annotator)
                    annotation = item.annotations[annotator_index]
                    annotation_id = item.annotation_ids[annotator_index]

        indices.append(annotator_index)
        annotations.append(annotation)
        annotators.append(selected_annotator)
        annotation_identifier.append(annotation_id)
    return {
        "annotators": annotators,
        "annotations": annotations,
        "annotation_indices": indices,
        "annotation_ids": annotation_identifier,
    }

def compute_annotator_representations(
    annotators, annotator2items, data_id2embedding, encoded_labels
):
    """
    Computes a mean representation for each annotator based on the embeddings of items they've annotated and
    the corresponding labels, then applies dimensionality reduction to these mean representations.

    This function first aggregates item embeddings and their respective label embeddings for each annotator,
    averages these embeddings to create a single representation per annotator, and finally reduces the dimensionality
    of these representations using a Singular Value Decomposition (SVD).

    Parameters:
    - annotators (list): A list of annotator identifiers.
    - annotator2items (dict): A mapping from annotator identifiers to a list of items they have annotated. Each item
                              in the list has an 'index' attribute to access its embedding and a 'raw_labels'
                              attribute to access its labels to retrieve an annotation for an annotator.
    - data_id2embedding (dict): A mapping from item indices to their embeddings.
    - encoded_labels (dict): A mapping from label values to their embeddings.

    Returns:
    - annotator2reduced_representation (dict): A dictionary where keys are annotator identifiers and values are the
                                                reduced dimensionality representations of the annotators' mean
                                                embeddings, normalized to unit length.
    """
    annotator2representation = {
        annotator: np.mean(
            np.hstack(
                [
                    [
                        data_id2embedding[item.index]
                        for item in annotator2items[annotator]
                    ],
                    [
                        encoded_labels[
                            item.raw_labels[item.annotators.index(annotator)]
                        ]
                        for item in annotator2items[annotator]
                    ],
                ]
            ),
            axis=0,
        )
        for annotator in annotators
    }

    # Convert to tensor and perform PCA-like dimensionality reduction
    all_annotator_embeddings = torch.tensor(
        list(annotator2representation.values()), dtype=torch.float32
    ).to("cuda")
    mean = torch.mean(all_annotator_embeddings, dim=0)
    all_annotator_embeddings_centered = all_annotator_embeddings - mean
    _, _, V = torch.svd(all_annotator_embeddings_centered)
    principal_components = V[:, :10]  # Assume k=10 for PCA
    reduced_data = torch.mm(all_annotator_embeddings_centered, principal_components)
    norm_reduced_data = torch.nn.functional.normalize(reduced_data, p=2, dim=1)

    # Map back to annotator keys
    annotator2reduced_representation = {
        annotator: norm_reduced_data[i] for i, annotator in enumerate(annotators)
    }
    return annotator2reduced_representation


def select_least_similar_annotator_representation(
    sampled_items, unlabelled_dataset, labelled_train_dataset, encoded_labels, device
):
    """
    This method selects the creates a simple representation for each annotator based on the items together with the respective
    label that they have annotated. Then the method computes the pair-wise cosine similarity between the annotators and selects
    the annotator that has the lowest similarity with the other annotators that are available for that item.
    """
    indices, annotations, annotators, annotation_identifier = [], [], [], []
    annotators_already_used = [item.annotators for item in labelled_train_dataset.items]
    annotators_already_used = set(
        [item for sublist in annotators_already_used for item in sublist]
    )

    # get the data_id2embedding mapping
    data_id2embedding = unlabelled_dataset.text2semantic_representation
    # get the annotator2items mapping
    annotator2items = labelled_train_dataset.get_annotator2items()
    # create a representation for all annotators with history based on the items they have annotated and the respective label
    if annotators_already_used:
        annotator2reduced_representation = compute_annotator_representations(
            annotators=annotators_already_used,
            annotator2items=annotator2items,
            data_id2embedding=data_id2embedding,
            encoded_labels=encoded_labels,
        )
    for item in sampled_items:
        item_annotators = item.annotators
        annotators_with_history = set(item_annotators).intersection(
            annotators_already_used
        )

        # If there is one (or more) annotator(s) that have not yet annotated anything, select one of them randomly
        if annotators_with_history != set(item_annotators):
            annotators_without_history = set(item_annotators).difference(
                annotators_with_history
            )
            selected_annotator = random.choice(list(annotators_without_history))
            annotator_index = item_annotators.index(selected_annotator)
            annotation = item.annotations[annotator_index]
            annotation_id = item.annotation_ids[annotator_index]

        # If all annotators have already annotated, select the annotator that has the lowest similarity with the other
        # annotators
        else:
            # retrieve all annotator representations
            all_annotator_embeddings = [
                annotator2reduced_representation[annotator]
                for annotator in annotators_with_history
            ]
            all_annotator_embeddings = torch.stack(all_annotator_embeddings)
            all_annotator_embeddings = (
                all_annotator_embeddings.clone().detach().to(device)
            )
            # compute pair-wise similarity as a matrix operation (we can use the dot product since the embeddings are normalized to unit length)
            similarity_matrix = torch.mm(
                all_annotator_embeddings, all_annotator_embeddings.t()
            )
            # get the mean pairwise similarity for each annotator
            mean_similarity = torch.mean(similarity_matrix, dim=1)
            # select the annotator with the lowest mean similarity
            selected_annotator_index = torch.argmin(mean_similarity).item()
            selected_annotator = list(annotators_with_history)[selected_annotator_index]
            annotator_index = item_annotators.index(selected_annotator)
            annotation = item.annotations[annotator_index]
            annotation_id = item.annotation_ids[annotator_index]

        indices.append(annotator_index)
        annotations.append(annotation)
        annotators.append(selected_annotator)
        annotation_identifier.append(annotation_id)
    return {
        "annotators": annotators,
        "annotations": annotations,
        "annotation_indices": indices,
        "annotation_ids": annotation_identifier,
    }
