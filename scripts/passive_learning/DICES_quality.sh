#!/bin/bash
# Predefined arguments
args=(
  # Model arguments
  --output_dir passive_trainer
  --model_name_or_path prajjwal1/bert-tiny
  --max_seq_length 512
  --cache_dir /tmp/hf_cache
  --label_normalize_strategy epsilon
  # Data arguments
  --dataset_path data/DICES/DICES_unaggregated.tsv
  --task_name DICES-quality
  --num_labels 4
  --text_col response
  --data_id_col item_id
  --annotator_col rater_id
  --annotation_col Q1_whole_conversation_evaluation
  --load_kfold_split true
  --result_log_file passive_DICES_quality.jsonl
  # passive learning args
  --target_task quality
  --evaluate false
  --num_trials 1
  --training_size -1
  # standard training args
  --per_device_train_batch_size 128
  --per_device_eval_batch_size 128
  --report_to wandb
  --learning_rate 3e-4
  --lr_scheduler_type constant
  --num_train_epochs 50
  --weight_decay 0.01
  --load_best_model_at_end true
  --metric_for_best_model eval_jsdiv
  --greater_is_better false
  --logging_strategy epoch
  --evaluation_strategy steps
  --eval_steps 5
  --save_strategy steps
  --save_total_limit 5
  --save_steps 5
)

python ./train.py "${args[@]}" "$@"
