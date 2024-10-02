from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import tqdm
from sentence_transformers import SentenceTransformer
from scipy.special import softmax
from torch.utils.data import Dataset as TorchDataset
from transformers import PreTrainedTokenizer
from sklearn.model_selection import train_test_split

MFTC_CLASSES = ["foundation_present", "foundation_missing"]
MHS_TARGET_TASKS = ["dehumanize", "respect", "genocide"]
MFTC_TARGET_TASKS = [
    "non-moral",
    "care",
    "fairness",
    "loyalty",
    "authority",
    "purity",
    "harm",
    "cheating",
    "betrayal",
    "subversion",
    "degradation",
]
LIKERT_CLASSES = [
    "0",
    "1",
    "2",
    "3",
    "4",
]
DICES_TARGET_TASKS = [
    "Q1_whole_conversation_evaluation",
    "Q_overall",
]
# See: https://github.com/google-research-datasets/dices-dataset/tree/main/990
DICES_TASK_LABELS = {
    "Q1_whole_conversation_evaluation": {
        "It is nonsensical (repetitive, incomprehensible, gibberish)": 0,
        "None of the above - it is in English, it makes sense, and I am familiar with the topic": 1,
        "I am not familiar with the topic of the conversation": 2,
        "It is in a language other than English": 3,
    },
    "Q_overall": {"Yes": 0, "No": 1, "Unsure": 2},
}
TASK2LABEL_NAMES = {
    "DICES-overall": {0: "Unsafe", 1: "Safe", 2: "Unsure"},
    "DICES-quality": {
        0: "Nonsensical",
        1: "understandable",
        2: "Not familiar with topic",
        3: "Other language",
    },
    "MFTC-care": {0: "Care not present", 1: "Care present"},
    "MFTC-fairness": {0: "Fairness not present", 1: "Fairness present"},
    "MFTC-loyalty": {0: "Loyalty not present", 1: "Loyalty present"},
    "MFTC-authority": {0: "Authority not present", 1: "Authority present"},
    "MFTC-purity": {0: "Purity not present", 1: "Purity present"},
    "MFTC-harm": {0: "Harm not present", 1: "Harm present"},
    "MFTC-cheating": {0: "Cheating not present", 1: "Cheating present"},
    "MFTC-betrayal": {0: "Betrayal not present", 1: "Betrayal present"},
    "MFTC-subversion": {0: "Subversion not present", 1: "Subversion present"},
    "MFTC-degradation": {0: "Degradation not present", 1: "Degradation present"},
    "MHS-dehumanize": {
        0.0: "strongly disagree",
        1.0: "disagree",
        2.0: "neutral",
        3.0: "agree",
        4.0: "strongly agree",
    },
    "MHS-genocide": {
        0.0: "strongly disagree",
        1.0: "disagree",
        2.0: "neutral",
        3.0: "agree",
        4.0: "strongly agree",
    },
    "MHS-respect": {
        0.0: "strongly disagree",
        1.0: "disagree",
        2.0: "neutral",
        3.0: "agree",
        4.0: "strongly agree",
    },
}


def generate_soft_labels_raw_annotations(
    annotations, labels, soft_label_method="normalize"
):
    """
    Generate soft labels from raw annotations. The soft labels represent the probability of each class.
    Example: annotations = [0, 0, 1, 0, 1], soft_labels = [0.6, 0.4] for a binary classification problem.
    :param annotations: a list of raw annotations for one item. Each element represents the label of one annotator.
    :return: a dictionary of soft labels, where the key is the class and the value is the probability of the class.
    """
    num_annotations = len(annotations)
    # get unqiue classes from num_labels
    unique_classes = list(labels)
    class_counts = {cls: annotations.count(cls) for cls in unique_classes}
    if soft_label_method == "normalize":
        return {cls: count / num_annotations for cls, count in class_counts.items()}
    elif soft_label_method == "softmax":
        votes = np.array([class_counts[cls] for cls in unique_classes])
        soft_labels = softmax(votes)
        return {cls: soft_labels[i] for i, cls in enumerate(unique_classes)}
    else:
        raise ValueError(
            f"Invalid soft label method: {soft_label_method}. Must be one of ['normalize', 'softmax']"
        )


def load_unaggregated_data(data_args, test_set=False, fallback_split=True):
    """
    Load the unaggregated data.

    Args:
        data_args: data_args object
        test_set: If True, return also the test set.
        fallback_split: If True, use a random train-test split if no predefined split is available.


    """
    dataset = pd.read_csv(data_args.dataset_path, sep="\t")

    if "MFTC" in data_args.dataset_path:
        task = data_args.task_name.split("-")[1]
        task_fold_prefix = f"{task}_"
    else:
        task_fold_prefix = ""

    # Add a unique identifier for each single annotation if data_args.annotation_id_col is None. In
    # this case we cannot drop duplicates based on annotation_id_col because we assume each row
    # contains unique annotations
    if data_args.annotation_id_col is None:
        dataset["annotation_id"] = range(len(dataset))
        data_args.annotation_id_col = "annotation_id"
        unique_data = dataset
    else:
        # create a unique version of dataset based on annotation_id col.
        unique_data = dataset.drop_duplicates(subset=[data_args.annotation_id_col])

    if data_args.load_predefined_split:
        # set dtype of data_id_col to int
        data_folder = Path(data_args.dataset_path).parent
        train_dataset = pd.read_csv(
            data_folder / f"{task_fold_prefix}train_ids.csv",
            names=[data_args.data_id_col],
            header=0,
        )
        val_dataset = pd.read_csv(
            data_folder / f"{task_fold_prefix}val_ids.csv",
            names=[data_args.data_id_col],
            header=0,
        )
    elif data_args.load_kfold_split:
        fold = data_args.fold
        data_folder = Path(data_args.dataset_path).parent
        train_dataset = pd.read_csv(
            data_folder / f"{task_fold_prefix}train_ids_{fold}.csv",
            names=[data_args.data_id_col],
            header=0,
        )
        val_dataset = pd.read_csv(
            data_folder / f"{task_fold_prefix}val_ids_{fold}.csv",
            names=[data_args.data_id_col],
            header=0,
        )
    elif fallback_split:
        train_dataset, val_dataset = train_test_split(unique_data, test_size=0.2)
    else:
        train_dataset = unique_data
        val_dataset = pd.DataFrame(columns=unique_data.columns)

    # split unaggregated into train and eval based on id in corresponding aggregated dataset
    unaggregated_train = unique_data[
        unique_data[data_args.data_id_col].isin(train_dataset[data_args.data_id_col])
    ]
    unaggregated_val = unique_data[
        unique_data[data_args.data_id_col].isin(val_dataset[data_args.data_id_col])
    ]

    if test_set:
        if data_args.load_predefined_split:
            test_dataset = pd.read_csv(
                data_folder / f"{task_fold_prefix}test_ids.csv",
                names=[data_args.data_id_col],
                header=0,
            )
        elif data_args.load_kfold_split:
            test_dataset = pd.read_csv(
                data_folder / f"{task_fold_prefix}test_ids_{fold}.csv",
                names=[data_args.data_id_col],
                header=0,
            )
        else:
            train_dataset, test_dataset = train_test_split(train_dataset, test_size=0.2)
            unaggregated_train = unique_data[
                unique_data[data_args.data_id_col].isin(
                    train_dataset[data_args.data_id_col]
                )
            ]
        unaggregated_test = unique_data[
            unique_data[data_args.data_id_col].isin(test_dataset[data_args.data_id_col])
        ]
        return unaggregated_train, unaggregated_val, unaggregated_test
    else:
        return unaggregated_train, unaggregated_val


class LabelledDataItem:
    def __init__(self, index, text, num_labels, max_annotations):
        """
        Initializes a LabelledDataItem instance. Each instance corresponds to a unique data item with all
        @param index:
        @param text:
        @param num_labels:
        @param max_annotations:
        """
        self.index = index
        self.text = text
        self.num_labels = num_labels
        self.raw_labels = []  # List to store raw annotations
        self.max_annotations = max_annotations
        self.current_annotations = 0
        self.soft_label = None
        self.annotators = []  # Set to store unique annotators
        self.annotation_ids = (
            []
        )  # List to store unique annotation identifiers from un-aggregated dataset

    def add_annotation(self, annotation, annotator, annotation_id):
        if self.current_annotations < self.max_annotations:
            self.raw_labels.append(annotation)
            self.current_annotations += 1
            self.annotators.append(annotator)
            self.annotation_ids.append(annotation_id)
            self.update_soft_label()
        else:
            # raise an Exception if the maximum number of annotations has been reached
            raise Exception(
                f"Maximum number of annotations reached for item {self.index}"
            )

    def update_soft_label(self):
        if self.raw_labels:
            self.soft_label = generate_soft_labels_raw_annotations(
                annotations=self.raw_labels,
                labels=range(self.num_labels),
                soft_label_method="softmax",
            )
            # convert to a sorted list (sorted by label == key of the dictionary retrieved by generate_soft_labels_raw_annotations)
            self.soft_label = [
                self.soft_label[label] for label in sorted(self.soft_label.keys())
            ]

    def __str__(self):
        return (
            f"DataItem(index={self.index}, text='{self.text}', current_annotations={self.current_annotations},"
            f" raw_labels={str(self.raw_labels)}, self.soft_label={self.soft_label}, max annotations={self.max_annotations} annotators={self.annotators})"
        )


class UnlabelledItem:
    def __init__(
        self,
        data_id: int,
        text: str,
        annotations: list,
        annotators: list,
        annotation_ids: list,
    ):
        """
        Initializes an UnlabelledItem instance. Each instance corresponds to a unique data item with all
        annotations and annotators.

        Args:
            data_id (int): A unique identifier for the data item.
            text (str): The textual content of the data item.
            annotations (list): A list of annotations for the item (all individual labels assigned by annotators).
            annotators (list): A list of annotators who have annotated the item.
            annotation_ids (list): A list of unique identifiers for each annotation (that can be used
                to retrieve all information back from the unaggregated dataset).
        """
        self.data_id = data_id
        self.text = text
        self.annotations = (
            # List[str] or List[int], depending on annotation types
            annotations
        )
        self.annotators = annotators  # List[str] of annotators
        self.max_annotations = len(annotations)
        self.annotation_ids = (
            # List of unique identifiers, e.g., List[int] or List[str]
            annotation_ids
        )

    def __str__(self):
        return f"UnlabelledItem(data_id={self.data_id}, text='{self.text}', annotations={self.annotations}, annotators={self.annotators})"


class LabelledDataset(TorchDataset):
    """
    This class stores the labelled dataset that will be used in the training loop of the active learning pipeline.
    The dataset contains a number of items, each item contains text, current annotations, soft label etc.
    The items are not encoded on-the fly but added to the cache once the text has been encoded once. The soft label is
    not added to the cache such that it will always reflect the current annotations.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, max_length: int):
        self.items = []  # List of DataItem instances
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.cache = {}  # Cache to store encoded text

    def add_item(self, item: LabelledDataItem):
        self.items.append(item)
        # Remove text encoding from cache if it exists
        self.cache.pop(item.index, None)

    def total_annotations(self):
        """
        Returns the total number of annotations in the dataset.

        Returns:
            int: Total number of annotations.
        """
        total = 0
        for item in self.items:
            total += item.current_annotations
        return total

    def all_item_indices(self):
        return {item.index for item in self.items}

    def get_item_by_index(self, index):
        target = [item for item in self.items if item.index == index][0]
        return target

    def get_annotators(self):
        # return the set of unique annotators in that dataset
        return {annotator for item in self.items for annotator in item.annotators}

    def get_annotator2items(self):
        annotator2items = defaultdict(list)
        for item in self.items:
            for annotator in item.annotators:
                annotator2items[annotator].append(item)
        return annotator2items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        if idx >= len(self.items):
            raise IndexError("Index out of range")

        item = self.items[idx]
        # Check if the text encoding is in the cache
        if item.index not in self.cache:
            # If not in cache, encode the text and cache it
            encoded_text = self.tokenizer.encode_plus(
                item.text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            self.cache[item.index] = {
                "input_ids": encoded_text["input_ids"].squeeze(0),
                "attention_mask": encoded_text["attention_mask"].squeeze(0),
                "index": item.index,
                "text": item.text,
                "num_annotations": item.current_annotations,
            }

        # Retrieve the cached text encoding
        cached_data = self.cache[item.index]

        # Add the current soft label to the cached data
        soft_label = item.soft_label
        if soft_label is None:
            # raise Exception, this is an empty item
            print("empty item")

        # Prepare the data to return
        data = {
            **cached_data,
            "labels": torch.tensor(soft_label, dtype=torch.float32),
            "index": item.index,
        }

        return data


class UnlabelledDataset:
    def __init__(
        self,
        unaggregated_df,
        text_column,
        annotator_column,
        annotation_column,
        data_id_column,
        annotation_id_column,
        task_name,
    ):
        """
        Initializes an instance of the class.

        Args:
            unaggregated_df (pd.DataFrame): DataFrame containing unaggregated data.
            text_column (str): Name of the column in the DataFrame containing the text data.
            annotator_column (str): Name of the column specifying annotators.
            annotation_column (str): Name of the column containing annotations.
            data_id_column (str): Name of the column containing unique identifiers for data items.
                (is the same as index in LabelledDataItem and corresponds to the same data item.)
            annotation_id_column (str): Name of the column containing unique identifiers for annotations.
        """
        self.unaggregated_df = unaggregated_df
        self.text_column = text_column
        self.annotator_column = annotator_column
        self.annotation_column = annotation_column
        self.data_id_column = data_id_column
        self.annotation_id_column = annotation_id_column
        self.length = len(unaggregated_df)
        self.items = []
        self._build_dataset(unaggregated_df)
        self.task_name = task_name

    def _build_dataset(self, unaggregated_df):
        # Group by index to collate all annotations and annotators for each item
        for index, group in unaggregated_df.groupby(self.data_id_column):
            # Text is the same for all rows of the same index
            text = group[self.text_column].iloc[0]
            annotations = group[self.annotation_column].tolist()
            annotators = group[self.annotator_column].tolist()

            # Create an UnlabelledItem and add it to the dataset
            self.items.append(
                UnlabelledItem(
                    data_id=index,
                    text=text,
                    annotations=annotations,
                    annotators=annotators,
                    annotation_ids=group[self.annotation_id_column].tolist(),
                )
            )

    def get_item_by_index(self, index):
        target = [item for item in self.items if item.data_id == index][0]
        return target

    def build_text2semantic_representation(self):
        """
        Computes and stores semantic representations for all texts within the dataset using a pre-trained
        Sentence Transformer model. This method is particularly useful for annotator sampling strategies
        that rely on semantic similarity between text embeddings.

        The method leverages the 'all-MiniLM-L6-v2' Sentence Transformer model, if available, using
        a GPU for faster processing. Each text's semantic representation is keyed by
        its index in the dataset, allowing for direct association between texts and their embeddings.

        Attributes updated:
        - text2semantic_representation (dict): A dictionary where keys are dataset item indices and values
          are the corresponding semantic embeddings computed by the Sentence Transformer model.
        """
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
        text2semantic_representation = {}
        all_texts = [item.text for item in self.items]
        all_embeddings = model.encode(all_texts)
        for i, item in tqdm.tqdm(enumerate(self.items)):
            text2semantic_representation[item.data_id] = all_embeddings[i]
        self.text2semantic_representation = text2semantic_representation

    def get_annotators(self):
        """return a set of all unique annotators in the dataset"""
        return {annotator for item in self.items for annotator in item.annotators}

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        if idx >= len(self.items):
            raise IndexError("Index out of range")
        return self.items[idx]

    def total_annotations_count(self):
        """
        Returns the total number of annotations available in the dataset.

        Returns:
            int: Total number of annotations.
        """
        total_annotations = sum(len(item.annotations) for item in self.items)
        return total_annotations

    def pop_single_annotation(self, item_to_remove, annotation_id_to_remove):
        """
        Removes a single annotation from the dataset. This method is used to remove annotations
        from the dataset.

        Args:
            item (UnlabelledItem): The item to remove the annotation from.
            annotation_index (int): The index of the annotation to remove.
        """

        # get the item of the given item index
        item_indices = [item.data_id for item in self.items]
        # check at which index the item is in the list
        real_index = item_indices.index(item_to_remove.data_id)
        # get the item
        item = self.items[real_index]

        # get the index of the annotation_id to remove
        annotation_index_to_remove = item.annotation_ids.index(annotation_id_to_remove)

        # remove the annotation_id from the list
        item.annotation_ids.pop(annotation_index_to_remove)
        # remove the annotation from the list
        item.annotations.pop(annotation_index_to_remove)
        # remove the annotator from the list
        item.annotators.pop(annotation_index_to_remove)

        item.max_annotations -= 1
        # If the item has no annotations left, remove it from the dataset
        if item.max_annotations == 0:
            self.items.pop(real_index)
            self.length -= 1

    def pop_annotations(self, annotations_to_remove):
        """
        Removes the annotations at the given indices and annotation indices from the dataset.
        :param annotations_to_remove: a list of dicts with keys 'item' and 'annotation_index' that specify the
        item index and the annotation index to remove.
        :return:
        """
        # Sort the annotations from high index to low index to remove by item index. This is
        # necessary to avoid index errors when removing annotations from the dataset.
        annotations_to_remove = sorted(
            annotations_to_remove, key=lambda x: x["item"].data_id, reverse=True
        )
        for annotation in annotations_to_remove:
            self.pop_single_annotation(annotation["item"], annotation["annotation_id"])


def prepare_labelled_dataset(eval_set, model_args, tokenizer):
    labelled_eval_dataset = LabelledDataset(
        max_length=model_args.max_seq_length,
        tokenizer=tokenizer,
    )
    for item in eval_set.items:
        labelled_item = LabelledDataItem(
            index=item.data_id,
            text=item.text,
            max_annotations=item.max_annotations,
            num_labels=model_args.num_labels,
        )
        for annotation, annotator, unique_id in zip(
            item.annotations, item.annotators, item.annotation_ids
        ):
            labelled_item.add_annotation(
                annotation=annotation, annotator=annotator, annotation_id=unique_id
            )
        labelled_eval_dataset.add_item(labelled_item)
    return labelled_eval_dataset
