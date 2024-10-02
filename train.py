from ray import tune
from ray.tune.schedulers import PopulationBasedTraining
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    set_seed,
)

import wandb
from annotator_diversity.datasets import (
    UnlabelledDataset,
    prepare_labelled_dataset,
    load_unaggregated_data,
)
from annotator_diversity.args import parse_arguments
from annotator_diversity.metrics import compute_metrics, CustomPassiveLoggingCallback, compute_metrics_passive
from annotator_diversity.soft_trainer import SoftTrainer, load_model

wandb.init(project="annotator-diversity")


def launch_training(
        model,
        train_data,
        val_data,
        test_data,
        training_args,
        learning_args,
        model_args,
        data_args,
):
    # Initial setup, create an 'unlabelled' dataset from the unaggregated training and validation set
    # (in fact it is labelled, but we use the existing labels as an oracle and synthesize the active learning process)
    train_set = UnlabelledDataset(
        unaggregated_df=train_data,
        text_column=data_args.text_col,
        annotator_column=data_args.annotator_col,
        annotation_column=data_args.annotation_col,
        data_id_column=data_args.data_id_col,
        annotation_id_column=data_args.annotation_id_col,
        task_name=data_args.task_name,
    )
    val_set = UnlabelledDataset(
        unaggregated_df=val_data,
        text_column=data_args.text_col,
        annotator_column=data_args.annotator_col,
        annotation_column=data_args.annotation_col,
        data_id_column=data_args.data_id_col,
        annotation_id_column=data_args.annotation_id_col,
        task_name=data_args.task_name,
    )

    try:
        tok = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path, use_fast=True
        )
    except OSError:
        print(
            f"Cannot load tokenizer from cache, initializing tokenizer using name {model_args.tokenizer_name}"
        )
        tok = AutoTokenizer.from_pretrained(model_args.tokenizer_name, use_fast=True)

    labelled_train_dataset = prepare_labelled_dataset(
        train_set, model_args=model_args, tokenizer=tok
    )
    labelled_val_dataset = prepare_labelled_dataset(
        val_set, model_args=model_args, tokenizer=tok
    )

    coll = DataCollatorWithPadding(tokenizer=tok, padding="longest")

    annotations_per_epoch = labelled_train_dataset.total_annotations()
    samples_per_epoch = len(labelled_train_dataset)

    trainer = SoftTrainer(
        model=model,
        args=training_args,
        train_dataset=labelled_train_dataset,
        eval_dataset=labelled_val_dataset,
        data_collator=coll,
        label_normalize_strategy=model_args.label_normalize_strategy,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=3),
            CustomPassiveLoggingCallback(
                annotations_per_epoch, samples_per_epoch, data_args.result_log_file
            ),
        ],
        compute_metrics=lambda x: compute_metrics_passive(
            eval_pred=x, label_normalize_strategy=model_args.label_normalize_strategy, eval_data=labelled_val_dataset
        ),
    )

    if learning_args.num_trials == 1:
        trainer.train()
    elif learning_args.num_trials > 1:
        tune_config = {
            "per_device_train_batch_size": 8,
            "per_device_eval_batch_size": 32,
            "num_train_epochs": tune.choice([2, 3, 4, 5]),
        }
        scheduler = PopulationBasedTraining(
            time_attr="training_iteration",
            metric="eval_loss",
            mode="min",
            perturbation_interval=1,
            hyperparam_mutations={
                "weight_decay": tune.uniform(0.0, 0.3),
                "learning_rate": tune.uniform(1e-5, 5e-5),
            },
        )

        trainer.hyperparameter_search(
            hp_space=lambda _: tune_config,
            backend="ray",
            n_trials=learning_args.num_trials,
            resources_per_trial={"cpu": 1, "gpu": 1},
            scheduler=scheduler,
            storage_path="./ray_results",
        )

    if learning_args.evaluate:
        test_set = UnlabelledDataset(
            unaggregated_df=test_data,
            data_id_column=data_args.data_id_col,
            text_column=data_args.text_col,
            annotation_id_column=data_args.annotation_id_col,
            annotator_column=data_args.annotator_col,
            annotation_column=data_args.annotation_col,
            task_name=data_args.task_name,
        )
        labelled_test_dataset = prepare_labelled_dataset(
            test_set, model_args=model_args, tokenizer=tok
        )

        # This will probably be the second to last checkpoint
        print(f"Best model checkpoint: {trainer.state.best_model_checkpoint}")
        trainer.evaluate(labelled_test_dataset, metric_key_prefix="test")


if __name__ == "__main__":
    data_args, model_args, learning_args, training_args = parse_arguments(passive=True)
    set_seed(training_args.seed)
    model = load_model(model_args)

    unaggregated_train, unaggregated_val, unaggregated_test = load_unaggregated_data(
        data_args, test_set=True
    )

    # Optionally reduce the training set size for testing purposes
    if learning_args.training_size > 0:
        unaggregated_train = unaggregated_train.sample(
            learning_args.training_size, random_state=training_args.seed
        )

    launch_training(
        model=model,
        train_data=unaggregated_train,
        val_data=unaggregated_val,
        test_data=unaggregated_test,
        training_args=training_args,
        learning_args=learning_args,
        model_args=model_args,
        data_args=data_args,
    )
