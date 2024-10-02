import json
import logging
import random
import sys
from collections import defaultdict
from pathlib import Path
from copy import copy

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sentence_transformers import SentenceTransformer
import numpy as np

from annotator_diversity.datasets import (
    LabelledDataItem,
    prepare_labelled_dataset,
    TASK2LABEL_NAMES,
)
from annotator_diversity.metrics import additional_evaluations
from annotator_diversity.query_annotations import (
    randomly_select_annotations,
    select_based_on_minority_label,
    select_based_on_semantic_similarity,
    select_least_similar_annotator_representation,
    select_n_or_all,
)
from annotator_diversity.query_datapoints import (
    randomly_select_datapoints,
    select_datapoints_by_uncertainty,
)

from preprocess_data import compute_normalized_entropy


class ActiveLearning:
    def __init__(
        self,
        main_model,
        trainer,
        model_args,
        datapoints_selection_strategy,
        annotator_selection_strategy,
        labelled_train_dataset,
        labelled_eval_dataset,
        unlabelled_dataset,
        sample_size,
        annotation_sample_size,
        num_rounds,
        entropy_model_path=None,
        annotator_selection_model=None,
        warmup_flag=False,
        warmup_size=-1,
        log_filename=None,
        al_save_strategy=None,
        **kwargs,
    ):
        """
        Args:
        @param main_model: classifier that predicts probabilities for each class
        @param trainer: SoftTrainer instance that trains the model using the cross-entropy loss
        @param model_args: arguments for the model
        @param datapoints_selection_strategy: how are datapoints sampled from the unlabelled dataset?
        @param annotator_selection_strategy: how are annotators and their corresponding annotations sampled for each of
        the selected datapoints?
        @param labelled_train_dataset: initial labelled dataset that contains items and their soft labels.
        new items are incrementally added to this dataset during the active learning process and the dataset is being updated.
        @param labelled_eval_dataset: an evaluation dataset with instances and their "gold" soft labels.
        @param unlabelled_dataset: the unaggreated dataset that contains all the instances and their full set of annotations.
        @param sample_size: how many datapoints are sampled in each active learning iteration?
        @param num_rounds: how many active learning iterations are performed?
        @param entropy_model: only has to be specified if the DAAL strategy is used for selecting datapoints
        @param annotator_selection_model: only has to be specified if needed / used for selecting annotators
        @param warmup_flag: if true, a warmup phase is performed before the active learning cycle starts which pre-trains
        the model on a random sample of datapoints and annotations
        @param warmup_size: number of annotations to warmup the model on. If smaller or equal than 0,
        the warmup size is set to the sample size
        @param log_filename: filename for logging the results
        """
        self.main_model = main_model
        self.entropy_model_path = entropy_model_path
        if self.entropy_model_path is not None:
            # use the config to load the corresponding model
            self.entropy_model = AutoModelForSequenceClassification.from_pretrained(
                self.entropy_model_path, use_safetensors=True
            )
            print("Entropy model loaded")

        self.annotator_selection_model = annotator_selection_model
        self.trainer = trainer
        self.datapoints_selection_strategy = datapoints_selection_strategy
        self.annotator_selection_strategy = annotator_selection_strategy
        self.labelled_train_dataset = labelled_train_dataset
        self.labelled_eval_dataset = labelled_eval_dataset
        self.unlabelled_dataset = unlabelled_dataset
        self.sample_size = sample_size
        self.num_rounds = num_rounds
        self.warmup_flag = warmup_flag
        self.warmup_size = warmup_size if warmup_size > 0 else sample_size
        self.init_logger(log_filename)
        self.al_save_strategy = al_save_strategy
        data_logging_filename = log_filename.split(".")[0] + "_data_logging.jsonl"
        self.init_datalogger(data_logging_filename)
        self.model_args = model_args
        self.num_epochs_per_round = self.trainer.args.num_train_epochs
        self.total_trainer_steps = 0
        self.total_epochs_trained = 0
        self.counters = defaultdict(lambda: 0)
        self.annotation_sample_size = annotation_sample_size
        self.dump_predictions_during_eval = kwargs["dump_predictions_during_eval"]
        self.decay_learning_rate = kwargs["decay_learning_rate"]
        self.best_model_metric = kwargs["best_model_metric"]
        self.best_metric_greater_is_better = kwargs[
            "best_model_metric_greater_is_better"
        ]
        self.best_model_tracker = defaultdict(lambda: None)
        self.initial_learning_rate = copy(self.trainer.args.learning_rate)
        self.current_learning_rate = copy(self.trainer.args.learning_rate)

        if (
            self.annotator_selection_strategy == "annotator_representation"
            or self.annotator_selection_strategy == "semantic_diversity"
        ):
            encoder = SentenceTransformer("all-MiniLM-L6-v2")
            self.label2semantic_representation = {}
            for i in range(self.model_args.num_labels):
                labelindex2label_name = TASK2LABEL_NAMES[
                    self.unlabelled_dataset.task_name
                ]
                self.label2semantic_representation[i] = encoder.encode(
                    labelindex2label_name[i]
                )
            # also pre-initialize the item2embedding dictionary so it is only done once
            self.unlabelled_dataset.build_text2semantic_representation()

        if self.warmup_flag:
            # do a warmup phase for the annotator selection model
            self.warmup_for_annotator_selection()

    def select_datapoints(self):
        """
        Selects datapoints from the unlabelled dataset based on the strategy. Each strategy is called based on a string
        (keyword).
        @return: a list of UnlabelledItem instances that are selected from the unlabelled dataset
        """
        # Implementation for selecting datapoints based on the strategy
        # here we can add more strategies to select the datapoints, e.g. uncertainty sampling and daal
        if self.datapoints_selection_strategy == "random":
            selected_items = randomly_select_datapoints(
                self.unlabelled_dataset, self.sample_size
            )
            return selected_items
        elif self.datapoints_selection_strategy == "uncertainty":
            train_data_to_predict = prepare_labelled_dataset(
                self.unlabelled_dataset,
                self.model_args,
                tokenizer=self.labelled_train_dataset.tokenizer,
            )

            selected_items = select_datapoints_by_uncertainty(
                unlabelled_to_predict=train_data_to_predict,
                unlabelled_dataset=self.unlabelled_dataset,
                trainer=self.trainer,
                n=self.sample_size,
            )
            return selected_items
        else:
            raise NotImplementedError(
                f"Unknown datapoints selection strategy: {self.datapoints_selection_strategy}"
            )

    def select_annotators(self, selected_items):
        """
        Selects annotators and their corresponding annotations for each of the selected datapoints based on the strategy.
        Each strategy is called based on a string (keyword).
        @param selected_items: a list of UnlabelledItem instances that are selected from the unlabelled dataset.
        @return: a dictionary with keys "annotations", "annotators", and "annotation_indices". The values are lists of
        annotations, annotators, and annotation indices that correspond to the selected items. The annotation indices
        correspond to the current indices of the annotations that are left in the unlabelled dataset.
        """
        if self.annotator_selection_strategy == "random":
            selected_annotations = randomly_select_annotations(selected_items)
            return selected_annotations
        elif self.annotator_selection_strategy == "semantic_diversity":
            selected_annotations = select_based_on_semantic_similarity(
                sampled_items=selected_items,
                labelled_train_dataset=self.labelled_train_dataset,
                unlabelled_dataset=self.unlabelled_dataset,
                device=self.trainer.args.device,
            )
            return selected_annotations
        elif self.annotator_selection_strategy == "label_minority":
            selected_annotations = select_based_on_minority_label(
                sampled_items=selected_items,
                labelled_train_dataset=self.labelled_train_dataset,
            )
            return selected_annotations
        elif self.annotator_selection_strategy == "annotator_representation":
            selected_annotations = select_least_similar_annotator_representation(
                sampled_items=selected_items,
                unlabelled_dataset=self.unlabelled_dataset,
                labelled_train_dataset=self.labelled_train_dataset,
                encoded_labels=self.label2semantic_representation,
                device=self.trainer.args.device,
            )
            return selected_annotations
        elif self.annotator_selection_strategy == "n_or_all":
            selected_annotations = select_n_or_all(
                sampled_items=selected_items, n=self.annotation_sample_size
            )
            return selected_annotations
        else:
            raise NotImplementedError(
                f"Unknown annotator selection strategy: {self.annotator_selection_strategy}"
            )

    def warmup_for_annotator_selection(self):
        """
        We warmup the model with a random sample of annotations. The number of datapoints and annotations
        that are selected for the warmup phase is determined by the warmup_size parameter.
        """
        # Set learning round to -1 for warmup phase
        learning_round = -1

        # Create a list of all annotator-item combinations
        annotator_item_combinations = []
        for item in self.unlabelled_dataset:
            for annotator in item.annotators:
                annotator_item_combinations.append((annotator, item))

        # Pick warmup_size random elements from annotator_item_combinations
        random_combinations = random.sample(
            annotator_item_combinations, self.warmup_size
        )

        selected_items = []
        selected_annotations = {
            "annotations": [],
            "annotators": [],
            "annotation_indices": [],
            "annotation_ids": [],
        }

        for annotator, item in random_combinations:
            item_annotators = item.annotators
            annotator_index = item_annotators.index(annotator)
            selected_annotations["annotations"].append(
                item.annotations[annotator_index]
            )
            selected_annotations["annotators"].append(annotator)
            selected_annotations["annotation_indices"].append(annotator_index)
            selected_annotations["annotation_ids"].append(
                item.annotation_ids[annotator_index]
            )
            selected_items.append(item)

        self.update_datasets(
            selected_items=selected_items, selected_annotations=selected_annotations
        )

        number_of_training_items = len(self.labelled_train_dataset)
        number_of_training_annotators = len(
            self.labelled_train_dataset.get_annotators()
        )

        self.logger.debug(
            "Warmup phase for annotation selection training. The number of items selected is: {} and the number of annotators is: {}".format(
                number_of_training_items, number_of_training_annotators
            )
        )

        self.train_model(learning_round)
        additional_results = self.evaluate_model(learning_round)

        # log round information
        self.log_round_info(
            learning_round=learning_round,
            selected_datapoints=selected_items,
            selected_annotations=selected_annotations,
            additional_results=additional_results,
        )

    def train_model(self, round_number):
        """
        Continue training the model based on the labeled data. If the model was already trained in a previous round,
        load the model from the previous round and continue training.
        """
        prev_model = Path(self.trainer.args.output_dir) / f"round_{round_number - 1}"
        skip_load = False
        if round_number == 0:
            # if we don't have a warmup phase, skip loading the model from round -1
            skip_load = not self.warmup_flag

        if prev_model.exists() and not skip_load:
            self.logger.debug(f"Loading from previous model {prev_model}")

            num_batches_per_epoch = np.ceil(
                len(self.trainer.train_dataset)
                / self.trainer.args.per_device_train_batch_size,
            )

            num_epochs_trained_estimated = (
                self.trainer.state.global_step // num_batches_per_epoch
            )
            diff_true_estimated_epochs = (
                self.total_epochs_trained - num_epochs_trained_estimated
            )
            self.trainer.args.num_train_epochs = (
                self.total_epochs_trained
                + self.num_epochs_per_round
                - diff_true_estimated_epochs
            )

            self.trainer.train(resume_from_checkpoint=prev_model)

        else:
            self.trainer.train()

        # Keep track of total number of steps trained
        self.total_trainer_steps += self.trainer.state.global_step
        # and also the epochs trained
        self.total_epochs_trained += self.num_epochs_per_round

        # Increment counters
        self.counters["total_trainer_steps"] = self.total_trainer_steps
        self.counters["total_epochs_trained"] = self.total_epochs_trained
        self.counters["total_rounds"] = round_number
        self.counters["total_annotations_seen"] += (
            self.labelled_train_dataset.total_annotations() * self.num_epochs_per_round
        )
        self.counters["total_unique_annotations_seen"] = (
            self.labelled_train_dataset.total_annotations()
        )
        self.counters["total_samples_seen"] += (
            len(self.labelled_train_dataset) * self.num_epochs_per_round
        )

        if self.decay_learning_rate:
            if round_number > 0:
                self.current_learning_rate -= self.current_learning_rate * 0.1
        self.trainer.args.learning_rate = self.current_learning_rate

        # Save model and state
        self.save_active_learning_round(round_number)

    def save_active_learning_round(self, round_number):
        do_save = False
        if self.al_save_strategy == "all":
            do_save = True
        elif self.al_save_strategy == "start-half-end" and round_number in [
            1,
            2,
            3,
            self.num_rounds // 2,
            self.num_rounds - 1,
        ]:
            do_save = True

        if do_save:
            save_path = Path(self.trainer.args.output_dir) / f"round_{round_number}"
            self.trainer.args.learning_rate = self.current_learning_rate
            self.trainer.optimizer = None
            self.trainer.save_model(
                Path(self.trainer.args.output_dir) / f"round_{round_number}"
            )
            self.trainer.state.save_to_json(save_path / "trainer_state.json")

    def load_best_model_from_round(self, round_number):
        """
        Load the best model from a specific round.
        """
        best_model_path = Path(self.trainer.args.output_dir) / f"round_{round_number}"
        self.logger.debug(
            f"Loading best model from round {round_number} from {best_model_path}"
        )
        self.trainer._load_from_checkpoint(best_model_path)

    def track_best_model(self, learning_round, results):
        if self.al_save_strategy != "all":
            self.logger.warn(
                "Warning: Loading the best model after training may not work since not all models are saved."
            )
        if self.best_model_metric in results:
            if (
                (self.best_model_tracker["best_model_metric"] is None)
                or (
                    self.best_metric_greater_is_better
                    and results[self.best_model_metric]
                    > self.best_model_tracker["best_model_metric"]
                )
                or (
                    not self.best_metric_greater_is_better
                    and results[self.best_model_metric]
                    < self.best_model_tracker["best_model_metric"]
                )
            ):
                self.best_model_tracker["best_model_metric"] = results[
                    self.best_model_metric
                ]
                self.best_model_tracker["best_model_metric_round"] = learning_round
                self.best_model_tracker["best_model_metric_total_steps"] = (
                    self.total_trainer_steps
                )
                self.best_model_tracker["best_model_metric_total_epochs"] = (
                    self.total_epochs_trained
                )
                self.best_model_tracker["best_model_metric_total_unqiue_samples"] = (
                    self.counters["total_unique_annotations_seen"]
                )

    def evaluate_model(self, learning_round, override_test_set=None):
        """
        Evaluate the model on the labelled (by the oracle) evaluation dataset.
        """

        if override_test_set is not None:
            eval_data = override_test_set
            prefix = "test"
        else:
            eval_data = self.labelled_eval_dataset
            prefix = "eval"

        output = self.trainer.predict(eval_data, metric_key_prefix=prefix)

        results = output.metrics

        additional_eval = additional_evaluations(
            output.predictions,
            eval_data.items,
            self.model_args.label_normalize_strategy,
        )

        # Log here all plottable results
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

        self.track_best_model(learning_round, results)

        # Log results
        log_dict = {
            "results": results,
            "best_model": dict(self.best_model_tracker),
            "counters": dict(self.counters),
        }
        self.logger.info(json.dumps(log_dict))

        # Log the results to the trainer
        self.trainer.log(results)
        self.trainer.log(
            {
                "total_trainer_steps": self.total_trainer_steps,
                "total_rounds": learning_round,
            }
        )

        if self.dump_predictions_during_eval:
            dump_predictions = {
                "predictions": output.predictions.tolist(),
                "eval_idx": [x["index"] for x in eval_data],
                "eval_labels": [x["labels"].tolist() for x in eval_data],
            }
            self.datalogger.info(json.dumps(dump_predictions))

        # return additional evaluations to log also all non-plottable results
        return additional_eval

    def init_logger(self, log_filename=None):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(logging.StreamHandler(sys.stdout))
        if log_filename is not None:
            self.logger.addHandler(logging.FileHandler(log_filename))
        else:
            self.logger.addHandler(logging.FileHandler("active_learning.log"))

    def init_datalogger(self, log_filename=None):
        self.datalogger = logging.getLogger(__name__ + ".datalogger")
        self.datalogger.setLevel(logging.INFO)
        self.datalogger.propagate = False

        # Clear existing handlers to avoid duplicate logging
        self.datalogger.handlers = []

        # Setup file handler, assuming you want a different default file name here
        file_handler = logging.FileHandler(
            log_filename if log_filename else "data_logging.log"
        )
        self.datalogger.addHandler(file_handler)

    def run_active_learning_cycle(self):
        # Main loop for running the active learning cycle
        for learning_round in range(0, self.num_rounds):
            self.logger.debug(f"Round {learning_round + 1}/{self.num_rounds}")
            # step 1: select datapoints
            selected_datapoints = self.select_datapoints()
            # step 2: select annotators and their corresponding annotations
            selected_annotations = self.select_annotators(selected_datapoints)
            # step 3: update datasets, add selected datapoints to the labelled dataset and remove the annotations
            # from the unlabelled dataset
            self.update_datasets(selected_datapoints, selected_annotations)

            self.logger.debug(
                f"Current size of labelled dataset: {len(self.labelled_train_dataset)}"
            )
            self.logger.debug(
                f"Current number of annotations: {self.labelled_train_dataset.total_annotations()}"
            )

            # step 4: train and evaluate the model
            self.train_model(learning_round)
            # step 5: update the total number of trainer steps to count the number of steps across all rounds of active learning
            additional_results = self.evaluate_model(learning_round)

            # log round information
            self.log_round_info(
                learning_round=learning_round,
                selected_datapoints=selected_datapoints,
                selected_annotations=selected_annotations,
                additional_results=additional_results,
            )

            # Log current round, size of labelled dataset, and number of annotations and the evaluation metrics
            self.logger.debug(f"End of Round {learning_round + 1}/{self.num_rounds}")

    def log_round_info(
        self,
        learning_round,
        selected_datapoints,
        selected_annotations,
        additional_results,
    ):
        item_indices_set = {item.data_id for item in selected_datapoints}
        item_indices = sorted(list(item_indices_set))
        index_map = {data_id: index for index, data_id in enumerate(item_indices)}
        # sort selected_datapoints and selected_annotations based on the index_map
        annotations_all = [
            selected_annotations["annotations"][index_map[item.data_id]]
            for item in selected_datapoints
        ]
        annotators_all = [
            selected_annotations["annotators"][index_map[item.data_id]]
            for item in selected_datapoints
        ]
        annotation_ids_all = [
            selected_annotations["annotation_ids"][index_map[item.data_id]]
            for item in selected_datapoints
        ]

        # retrieve the selected items from UnlabelledDataset
        selected_items_unlabelled = [
            item
            for item in self.unlabelled_dataset.items
            if item.data_id in item_indices_set
        ]
        items_current_status = [
            item
            for item in self.labelled_train_dataset.items
            if item.index in item_indices_set
        ]

        selected_items_unlabelled.sort(key=lambda x: index_map[x.data_id])
        items_current_status.sort(key=lambda x: index_map[x.index])

        entropy_selected_items = [
            compute_normalized_entropy(item.annotations, self.model_args.num_labels)
            for item in selected_items_unlabelled
        ]
        current_entropy_selected_items = [
            compute_normalized_entropy(item.raw_labels, self.model_args.num_labels)
            for item in items_current_status
        ]

        # Retrieve evaluation annotators
        eval_annotators = [item.annotators for item in self.labelled_eval_dataset.items]

        # Some sanity checks
        worst_annotators = additional_results["worst_jsdiv_per_item"][
            "worst_annotators"
        ]
        assert len(eval_annotators) == len(
            worst_annotators
        ), "Length of eval_annotators does not match length of worst_annotators"
        # check if all elements of the lists contained in most_displaced_annotators are also in eval_annotators
        for i, annotators in enumerate(worst_annotators):
            for annotator in annotators:
                assert (
                    annotator in eval_annotators[i]
                ), f"Annotator {annotator} not in eval_annotators for item {i}"

        # create a dictionary with all the loggable information
        log_dict = {
            "learning_round": learning_round,
            "selected_datapoints": item_indices,
            "selected_annotations": annotations_all,
            "selected_annotators": annotators_all,
            "selected_annotation_ids": annotation_ids_all,
            "original_entropy": entropy_selected_items,
            "current_entropy": current_entropy_selected_items,
            "eval_annotators": eval_annotators,
            "worst_eval_annotators_per_item": additional_results[
                "worst_jsdiv_per_item"
            ]["worst_annotators"],
            "worst_jsdivs_per_item": additional_results["worst_jsdiv_per_item"][
                "worst_jsdivs"
            ],
            "worst_mean_jsdivs_per_item": additional_results["worst_jsdiv_per_item"][
                "mean"
            ],
            "worst_eval_annotators_per_annotator": additional_results[
                "worst_jsdiv_per_annotator"
            ]["worst_annotators"],
            "worst_jsdiv_per_annotator": additional_results[
                "worst_jsdiv_per_annotator"
            ]["worst_jsdiv"],
            "worst_mean_jsdiv_per_annotator": additional_results[
                "worst_jsdiv_per_annotator"
            ]["mean"],
            "worst_eval_annotators_f1": additional_results["f1_per_annotator"][
                "worst_annotators"
            ],
            "worst_f1_per_annotator": additional_results["f1_per_annotator"][
                "worst_f1"
            ],
            "mean_f1_per_annotator": additional_results["f1_per_annotator"]["mean"],
        }
        self.datalogger.info(json.dumps(log_dict))

    def update_datasets(self, selected_items, selected_annotations):
        """
        Updates the labelled and unlabelled datasets based on the selected items and annotations. Adds the selected items
        to the labelled dataset and removes either the selected annotation from the item or the complete item if
        it's annotations are exhausted from the unlabelled dataset.
        Args:
            selected_items (list): A list of UnlabelledItem instances.
            selected_annotations (dict): A dictionary with keys "annotations", "annotators", "annotation_indices", and "annotation_ids".
        """
        annotations_to_remove = []
        for i, item in enumerate(selected_items):
            annotation = selected_annotations["annotations"][i]
            annotator = selected_annotations["annotators"][i]
            annotation_id = selected_annotations["annotation_ids"][i]
            # Check if the item already exists in the labelled dataset
            if item.data_id in self.labelled_train_dataset.all_item_indices():
                existing_item = self.labelled_train_dataset.get_item_by_index(
                    item.data_id
                )
            else:
                # Create a new item if it doesn't exist in the labelled dataset
                existing_item = LabelledDataItem(
                    index=item.data_id,
                    text=item.text,
                    max_annotations=item.max_annotations,
                    num_labels=self.model_args.num_labels,
                )
                self.labelled_train_dataset.add_item(existing_item)
            # check if annotation is a list, that means several annotations were selected and need to be added / removed.
            if isinstance(annotation, list):
                for j in range(len(annotation)):
                    # Sanity check we are selecting the right annotation from right sample
                    assert annotator[j] in item.annotators
                    existing_item.add_annotation(
                        annotation=annotation[j],
                        annotator=annotator[j],
                        annotation_id=annotation_id[j],
                    )
                    annotations_to_remove.append(
                        {"item": item, "annotation_id": annotation_id[j]}
                    )
            else:
                # Sanity check we are selecting the right annotation from right sample
                assert annotator in item.annotators
                # Add the annotation to the item
                existing_item.add_annotation(
                    annotation=annotation,
                    annotator=annotator,
                    annotation_id=annotation_id,
                )
                annotations_to_remove.append(
                    {"item": item, "annotation_id": annotation_id}
                )

        # Remove the extracted annotations from the unlabelled dataset
        self.unlabelled_dataset.pop_annotations(annotations_to_remove)
        self.trainer.train_dataset = self.labelled_train_dataset
