import os
import sys
from dataclasses import dataclass, field
from pprint import pprint

from transformers import HfArgumentParser, TrainingArguments


@dataclass
class ModelTrainingArguments:
    # Model Configuration
    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    tokenizer_name: str = field(
        default=None,
        metadata={
            "help": "Name of the tokenizer identifier from huggingface.co/models"
        },
    )
    max_seq_length: int = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    num_labels: int = field(
        default=2, metadata={"help": "Number of labels of the task to be trained on"}
    )

    cache_dir: str = field(
        default="/tmp/hf_cache",
        metadata={"help": "Directory to cache the pretrained model"},
    )
    label_normalize_strategy: str = field(
        default="softmax",
        metadata={
            "help": "Strategy to normalize the labels in compute_metrics()",
            "choices": ["softmax", "epsilon", "none"],
        },
    )


@dataclass
class DataLoadingArguments:
    """arguments pertaining to what data we are going to input our model for training and eval"""

    # Data Configuration
    text_col: str = field(
        default="text", metadata={"help": "Column which stores the text"}
    )
    annotator_col: str = field(
        default="annotator",
        metadata={"help": "Column storing the label for classification"},
    )
    annotation_col: str = field(
        default="annotation",
        metadata={"help": "Column storing the label for classification"},
    )
    data_id_col: str = field(
        default="id",
        metadata={"help": "Column storing the unique id of each datapoint"},
    )
    annotation_id_col: str = field(
        default=None,
        metadata={"help": "Column storing the unique id of each annotation"},
    )

    dataset_path: str = field(
        default="dataset.csv", metadata={"help": "The dataset for active learning"}
    )
    task_name: str = field(
        default="task", metadata={"help": "The task name for active learning"}
    )
    load_predefined_split: bool = field(
        default=False,
        metadata={"help": "Flag to indicate whether to load predefined splits"},
    )
    load_kfold_split: bool = field(
        default=False,
        metadata={"help": "Flag to indicate whether to load kfold splits"},
    )
    fold: int = field(
        default=0,
        metadata={"help": "Fold number for kfold split"},
    )
    result_log_file: str = field(
        default="results.jsonl",
        metadata={"help": "File to store the results of active learning"},
    )


@dataclass
class PassiveLearningArguments:
    """
    Arguments for the passive learning experiments
    """

    target_task: str = field(
        default=None,
        metadata={"help": "Target MHS, MFTC or DICES task for training"},
    )

    evaluate: bool = field(
        default=False,
        metadata={
            "help": "Whether to run evaluate on the test set for passive learning"
        },
    )

    num_trials: int = field(
        default=1,
        metadata={
            "help": "Number of trials for hyperparameter search. If set to 0, perform no training. \
                If set to 1, perform training without hyperparameter search. If set to >1, perform \
                hyperparameter search with the specified number of trials."
        },
    )

    training_size: int = field(
        default=-1, metadata={"help": "Number of datapoints to train on"}
    )


@dataclass
class ActiveLearningArguments:
    """
    Arguments pertaining to active learning setup, including data, model, and active learning configurations.
    """

    # Active Learning Configuration
    sample_size: int = field(
        default=10,
        metadata={
            "help": "Number of samples to select in each active learning iteration"
        },
    )
    num_rounds: int = field(
        default=10, metadata={"help": "Number of rounds for active learning"}
    )
    evaluate: bool = field(
        default=False,
        metadata={
            "help": "Whether to run evaluate on the test set for active learning"
        },
    )
    warmup_flag: bool = field(
        default=False, metadata={"help": "Flag to indicate if warmup is needed"}
    )
    warmup_size: int = field(
        default=-1,
        metadata={
            "help": "Number of annotations to warmup the model on. If smaller \
                              or equal than 0, the warmup size is set to the sample size"
        },
    )
    datapoints_selection_strategy: str = field(
        default="random",
        metadata={"help": "Strategy to select datapoints for annotation"},
    )
    annotator_selection_strategy: str = field(
        default="random",
        metadata={
            "help": "Strategy to select annotators for annotation",
            "choices": [
                "random",
                "label_minority",
                "semantic_diversity",
                "agreement",
                "annotator_representation",
                "n_or_all",
            ],
        },
    )
    annotation_sample_size: int = field(
        default=1,
        metadata={
            "help": "Number of annotations to sample per datapoint, only used for n_or_all strategy"
        },
    )
    entropy_model_path: str = field(
        default=None,
        metadata={"help": "Path to entropy model if DAAL strategy is used"},
    )
    training_size: int = field(
        default=-1, metadata={"help": "Number of datapoints to train on"}
    )
    al_save_strategy: str = field(
        default="start-half-end",
        metadata={
            "help": "strategy to employ for saving checkpoints",
            "choices": ["start-half-end", "all"],
        },
    )
    dump_predictions_during_eval: bool = field(
        default=False,
        metadata={
            "help": "Flag to indicate whether to store predictions for samples in the eval set"
        },
    )
    decay_learning_rate: bool = field(
        default=False,
        metadata={
            "help": "Flag to indicate whether to decay learning rate (linearly) over active learning rounds"
        },
    )
    best_model_metric: str = field(
        default="eval_jsdiv",
        metadata={
            "help": "Metric to use for selecting the best model during active learning"
        },
    )
    best_model_metric_greater_is_better: bool = field(
        default=False,
        metadata={
            "help": "Flag to indicate whether the best model metric is better when higher"
        },
    )


def parse_arguments(passive=False):
    learning_arguments = (
        PassiveLearningArguments if passive else ActiveLearningArguments
    )
    parser = HfArgumentParser(
        (
            DataLoadingArguments,
            ModelTrainingArguments,
            learning_arguments,
            TrainingArguments,
        )
    )

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        (
            data_args,
            model_args,
            learning_args,
            training_args,
        ) = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        (
            data_args,
            model_args,
            learning_args,
            training_args,
        ) = parser.parse_args_into_dataclasses()

    pprint(learning_args)
    pprint(model_args)
    pprint(data_args)
    pprint(training_args)
    return data_args, model_args, learning_args, training_args
