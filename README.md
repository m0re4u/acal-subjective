# Annotator-Centric Active Learning for Subjective NLP Tasks

This repository contains the code for the ACAL framework proposed in the publication ["Annotator-Centric Active Learning for Subjective NLP Tasks"](https://arxiv.org/pdf/2404.15720) by Michiel van der Meer, Neele Falk, Pradeep K. Murukannaiah and Enrico Liscio published at EMNLP 2024. 
The code for running and evaluation the experiments with ACAL is located in the `annotator_diversity` module. The following readme describes how to install the dependencies, download the data, preprocess the data, and run experiments.
## Installation
1. Install the package in developer mode `pip install -e .`. This will also install the dependencies.
2.  We want to install sentence-transformers without installing its dependencies (it would overwrite some of the specific versions that we installed initially), so we install it after all the other ones with `pip install sentence-transformers --no-deps`
3. Done ...


## Data
- MFTC: download [here](https://surfdrive.surf.nl/files/index.php/s/mw7AmxSBsElvouL).
- MHS: the dataset is automatically downloaded when running the `preprocess_data.py` script (from hugginggface: `ucberkeley-dlab/measuring-hate-speech`)
- DICES: download the csv files from github [1](https://raw.githubusercontent.com/google-research-datasets/dices-dataset/main/350/diverse_safety_adversarial_dialog_350.csv) and [2](https://raw.githubusercontent.com/google-research-datasets/dices-dataset/main/990/diverse_safety_adversarial_dialog_990.csv), put into the data dir under `DICES/`.

Put the corresponding files in the directories `data/MFTC/`, `data/MHS/`, and `data/DICES/`.

## Preprocessing
Run `preprocess_data.py` after having downloaded all the data files. This will generate the data and the train / val / test splits. The data is stored in the `data/` directory. 

## Methods
- **Passive Learning** (full, regular training) using `train.py`, where the models learns to predict probability distributions for each sample (using a soft loss).
- **Active Learning with random selection** using `train_active_learning.py`.

Active learning experiments can be run via bash scripts inside `scripts/active_learning/`. The script calls the function ``train_active_learning.py`` and specifies a number of parameters that can be changed. There is a pre-defined script for each dataset and each label category. The format of the data needs to be an unaggregated file (each row corresponds to one annotation) with a text, a data ID (id for each uniquer datapoint), and a label.
Example scripts for passive learning via bash scripts are also provided in `scripts/passive_learning/`.

## Predefined Arguments

### Model Arguments
- **`--output_dir active_trainer`**: Directory where the model's output will be saved after training and evaluation.
- **`--model_name_or_path prajjwal1/bert-tiny`**: Path or name of the pre-trained model to use (here, `bert-tiny` from the Hugging Face model hub).
- **`--max_seq_length 512`**: Maximum number of tokens per input sequence. Any sequence longer than this will be truncated.
- **`--cache_dir /tmp/hf_cache`**: Directory used to cache model weights and other related files.
- **`--label_normalize_strategy epsilon`**: Specifies the strategy for label normalization, using the "epsilon" approach.

### Data Arguments
- **`--dataset_path data/DICES/DICES_unaggregated.tsv`**: Path to the unaggregated dataset file in TSV format.
- **`--task_name DICES-overall`**: The name of the task being performed (e.g., which classification task on DICES/MFTC/MHS).
  -    (options are: 'DICES-overall', 'DICES-quality', 'MFTC-care', 'MFTC-fairness', 'MFTC-loyalty', 'MFTC-authority', 'MFTC-purity', 'MFTC-harm', 'MFTC-cheating', 'MFTC-betrayal', 'MFTC-subversion', 'MFTC-degradation', 'MHS-dehumanize', 'MHS-genocide', 'MHS-respect')
- **`--num_labels 3`**: Number of output labels or classes in the dataset.
- **`--text_col response`**: Name of the column in the dataset that contains the text data.
- **`--data_id_col item_id`**: Name of the column that contains the unique identifier for each data item.
- **`--annotator_col rater_id`**: Name of the column that contains the annotator or rater IDs.
- **`--annotation_col Q_overall`**: Name of the column that contains the label or annotation.
- **`--result_log_file active_random_DICES_overall.jsonl`**: File where logs of results will be saved in JSONL format.

### Active Learning Arguments
- **`--sample_size 792`**: Number of data points to sample in each round of active learning.
- **`--num_rounds 70`**: Number of active learning rounds to perform.
- **`--datapoints_selection_strategy random`**: Strategy for selecting data points, in this case, randomly.
  - options are: 'random', 'uncertainty'
- **`--annotator_selection_strategy random`**: Strategy for selecting annotators, also random.
  -  (options are: 'random', 'label_minority', 'semantic_diversity', 'annotator_representation', 
- **`--warmup_flag true`**: Flag to indicate whether to use a warm-up phase before active learning.

### Standard Training Arguments
- **`--per_device_train_batch_size 128`**: Number of training samples per batch for each device (e.g., GPU or CPU).
- **`--per_device_eval_batch_size 128`**: Number of evaluation samples per batch for each device.
- **`--report_to wandb`**: Specifies the reporting tool to use, here `Weights and Biases (wandb)` for experiment tracking.
- **`--learning_rate 1e-5`**: The initial learning rate for the optimizer.
- **`--lr_scheduler_type constant`**: The type of learning rate scheduler, in this case, a constant learning rate.
- **`--num_train_epochs 1`**: Number of epochs to train the model for.
- **`--load_kfold_split true`**: Flag to load predefined K-fold split data for training.
- **`--logging_strategy epoch`**: Specifies when to log training metrics, here at the end of each epoch.
- **`--evaluation_strategy no`**: Evaluation strategy, with "no" indicating no intermediate evaluation during training.
- **`--ignore_data_skip true`**: Ignore any skipped data points during training (useful for resuming from checkpoints).
- **`--load_best_model_at_end false`**: Whether to load the best model at the end of training, here set to `false`.
- **`--metric_for_best_model eval_jsdiv`**: Metric to be used for determining the best model, here `Jensen-Shannon divergence`.
- **`--save_strategy no`**: Save strategy, with "no" indicating the model will not be saved after training.

## Reference
If you use this code, please cite the following paper:
```
@inproceedings{meer2024acal,
  title={Annotator-Centric Active Learning for Subjective NLP Tasks},
  author={van der Meer, Michiel and Falk, Neele and Murukannaiah, Pradeep K. and Liscio, Enrico},
  booktitle={EMNLP},
  year={2024}
}
```