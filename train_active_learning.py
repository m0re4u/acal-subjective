import torch
from transformers import AutoTokenizer, set_seed
from transformers.trainer_callback import DefaultFlowCallback

from annotator_diversity.active_learning import ActiveLearning
from annotator_diversity.args import parse_arguments
from annotator_diversity.datasets import (
    LabelledDataset,
    UnlabelledDataset,
    prepare_labelled_dataset,
    load_unaggregated_data,
)
from annotator_diversity.metrics import compute_metrics
from annotator_diversity.soft_trainer import (
    SoftTrainer,
    SoftTrainerFlowCallback,
    load_model,
)

# check for GPUs or CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU in use:")
else:
    print("using the CPU")
    device = torch.device("cpu")

# set up wandb
import wandb

wandb.login()
wandb.init(project="annotator-diversity")


def init_unlabelled_dataset(dataset, data_args):
    return UnlabelledDataset(
        unaggregated_df=dataset,
        data_id_column=data_args.data_id_col,
        text_column=data_args.text_col,
        annotation_id_column=data_args.annotation_id_col,
        annotator_column=data_args.annotator_col,
        annotation_column=data_args.annotation_col,
        task_name=data_args.task_name,
    )


def run_training_with_active_learning(
    model,
    train_data,
    eval_data,
    test_data,
    training_args,
    active_learning_args,
    model_args,
    data_args,
):
    # Initial setup, create an 'unlabelled' dataset from the unaggregated training set (in fact it is labelled, but we
    # use the existing labels as an oracle and synthesize the active learning process)
    unlabelled_dataset = init_unlabelled_dataset(train_data, data_args)
    eval_set = init_unlabelled_dataset(eval_data, data_args)
    test_set = init_unlabelled_dataset(test_data, data_args)

    try:
        tok = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path, use_fast=True
        )
    except OSError:
        print(
            f"Cannot load tokenizer from cache, initializing tokenizer using name {model_args.tokenizer_name}"
        )
        tok = AutoTokenizer.from_pretrained(model_args.tokenizer_name, use_fast=True)

    labelled_eval_dataset = prepare_labelled_dataset(
        eval_set, model_args=model_args, tokenizer=tok
    )
    labelled_test_dataset = prepare_labelled_dataset(
        test_set, model_args=model_args, tokenizer=tok
    )
    labelled_train_dataset = LabelledDataset(
        max_length=model_args.max_seq_length,
        tokenizer=tok,
    )
    trainer = SoftTrainer(
        model=model,
        args=training_args,
        train_dataset=labelled_eval_dataset,
        eval_dataset=labelled_eval_dataset,
        label_normalize_strategy=model_args.label_normalize_strategy,
        compute_metrics=lambda x: compute_metrics(
            x, model_args.label_normalize_strategy
        ),
    )
    trainer.pop_callback(DefaultFlowCallback)
    trainer.add_callback(SoftTrainerFlowCallback())
    # Initialize active learning
    al = ActiveLearning(
        main_model=model,
        model_args=model_args,
        unlabelled_dataset=unlabelled_dataset,
        **vars(active_learning_args),
        trainer=trainer,
        labelled_train_dataset=labelled_train_dataset,
        labelled_eval_dataset=labelled_eval_dataset,
        log_filename=data_args.result_log_file,
    )
    # Run active learning
    al.run_active_learning_cycle()

    if active_learning_args.evaluate:
        if al.best_model_tracker["best_model_metric_round"] is not None:
            # Load the best model
            print(
                f"Loading best model from round {al.best_model_tracker['best_model_metric_round']}"
            )
            al.load_best_model_from_round(
                al.best_model_tracker["best_model_metric_round"]
            )
        # Evaluate the final model
        al.evaluate_model(0, labelled_test_dataset)


if __name__ == "__main__":
    data_args, model_args, active_learning_args, training_args = parse_arguments(
        passive=False
    )
    set_seed(training_args.seed)

    model = load_model(model_args)

    unaggregated_train, unaggregated_eval, unaggregated_test = load_unaggregated_data(
        data_args, test_set=True
    )

    # Optionally reduce the training set size for testing purposes
    if active_learning_args.training_size > 0:
        unaggregated_train = unaggregated_train.sample(
            active_learning_args.training_size, random_state=training_args.seed
        )

    run_training_with_active_learning(
        model=model,
        train_data=unaggregated_train,
        eval_data=unaggregated_eval,
        test_data=unaggregated_test,
        training_args=training_args,
        active_learning_args=active_learning_args,
        model_args=model_args,
        data_args=data_args,
    )
