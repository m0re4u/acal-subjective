import math
import time
from typing import Any, Dict, List, Optional

from torch import nn
from torch.utils.data import Dataset
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers.trainer_callback import (
    DefaultFlowCallback,
    TrainerControl,
    TrainerState,
)
from transformers.trainer_utils import (
    IntervalStrategy,
    speed_metrics,
    RemoveColumnsCollator,
)

from annotator_diversity.metrics import worst_jsdiv_per_item


class SoftTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        self.label_normalize_strategy = kwargs.pop("label_normalize_strategy")
        super().__init__(*args, **kwargs)
        self.ent_loss = nn.CrossEntropyLoss()

        # Add the number of annotations seen during training
        # NOTE: this will not be saved to the checkpoint
        self.total_annotations_seen_during_training = 0

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        loss = self.ent_loss(outputs.logits, inputs["labels"])
        return (loss, outputs) if return_outputs else loss

    def _get_collator_with_removed_columns(
        self, data_collator, description: Optional[str] = None
    ):
        """
        Wrap the data collator in a callable removing unused columns.

        Removes columns that are not accepted by the model, with the exception of the
        `num_annotations` column, which is used to determine the number of annotations
        observed during training.
        """
        if not self.args.remove_unused_columns:
            return data_collator
        self._set_signature_columns_if_needed()
        signature_columns = self._signature_columns
        signature_columns += ["num_annotations"]

        remove_columns_collator = RemoveColumnsCollator(
            data_collator=data_collator,
            signature_columns=signature_columns,
            logger=None,
            description=description,
            model_name=self.model.__class__.__name__,
        )
        return remove_columns_collator

    def training_step(self, model, inputs):
        num_annotations_observed = inputs.pop("num_annotations").sum().item()
        self.total_annotations_seen_during_training += num_annotations_observed
        return super().training_step(model, inputs)

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys):
        inputs.pop("num_annotations")
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init `compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (`Dataset`, *optional*):
                Pass a dataset if you wish to override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns
                not accepted by the `model.forward()` method are automatically removed. It must implement the `__len__`
                method.
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is "eval" (default)

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        if eval_dataset is None:
            eval_dataset = self.eval_dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        start_time = time.time()

        eval_loop = (
            self.prediction_loop
            if self.args.use_legacy_prediction_loop
            else self.evaluation_loop
        )
        output = eval_loop(
            eval_dataloader,
            description="Evaluation",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        jsdiv_worst = worst_jsdiv_per_item(
            output.predictions,
            eval_dataset.items,
            self.label_normalize_strategy,
        )
        output.metrics[f"{metric_key_prefix}_jsdiv_worst"] = jsdiv_worst
        output.metrics[f"{metric_key_prefix}_total_annotations_seen"] = (
            self.total_annotations_seen_during_training
        )

        total_batch_size = self.args.eval_batch_size * self.args.world_size
        if f"{metric_key_prefix}_jit_compilation_time" in output.metrics:
            start_time += output.metrics[f"{metric_key_prefix}_jit_compilation_time"]
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        self.log(output.metrics)

        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, output.metrics
        )

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output.metrics


class SoftTrainerFlowCallback(DefaultFlowCallback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.global_step_from_previous_round = 0

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        When we check the global step, this includes the number of steps taken in the active learning loop, so we need
        to check the number of steps taken in _this_ round's training loop.
        """
        # Log
        if state.global_step == 1 and args.logging_first_step:
            control.should_log = True
        if (
            args.logging_strategy == IntervalStrategy.STEPS
            and state.global_step % state.logging_steps == 0
        ):
            control.should_log = True

        # Evaluate
        if (
            args.evaluation_strategy == IntervalStrategy.STEPS
            and state.global_step % state.eval_steps == 0
            and args.eval_delay <= state.global_step
        ):
            control.should_evaluate = True

        # Save
        if (
            args.save_strategy == IntervalStrategy.STEPS
            and state.save_steps > 0
            and state.global_step % state.save_steps == 0
        ):
            control.should_save = True

        # End training
        if state.global_step - self.global_step_from_previous_round >= state.max_steps:
            self.global_step_from_previous_round = state.global_step
            control.should_training_stop = True

        return control


def load_model(model_args):
    if model_args.cache_dir is not None:
        return AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            num_labels=model_args.num_labels,
            cache_dir=model_args.cache_dir,
        )
    else:
        return AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path, num_labels=model_args.num_labels
        )
