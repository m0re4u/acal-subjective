#!/bin/bash

# Predefined arguments
args=(
  # Model arguments
  --output_dir active_trainer
  --model_name_or_path prajjwal1/bert-tiny
  --max_seq_length 512
  --cache_dir /tmp/hf_cache
  --label_normalize_strategy epsilon
  # Data arguments
  --dataset_path data/MFTC/MFTC_unaggregated.tsv
  --task_name MFTC-care
  --num_labels 2
  --text_col text
  --data_id_col tweet_id
  --annotator_col annotator
  --annotation_col care
  --result_log_file representation_uncertainty_MFTC_care.jsonl
  # active learning args
  --sample_size 1250
  --num_rounds 20
  --datapoints_selection_strategy uncertainty
  --annotator_selection_strategy annotator_representation
  --warmup_flag true
  # standard training args
  --per_device_train_batch_size 128
  --per_device_eval_batch_size 128
  --report_to wandb
  --learning_rate 5e-5
  --lr_scheduler_type constant
  --num_train_epochs 1
  --load_kfold_split true
  --logging_strategy epoch
  --evaluation_strategy no
  --ignore_data_skip true
  --load_best_model_at_end false
  --metric_for_best_model eval_jsdiv
  --save_strategy no
)

python ./train_active_learning.py "${args[@]}" "$@"
