import json
import math
from collections import Counter

import pandas as pd
import tqdm
import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split, KFold

from annotator_diversity.datasets import (
    generate_soft_labels_raw_annotations,
    MHS_TARGET_TASKS,
    MFTC_TARGET_TASKS,
    DICES_TARGET_TASKS,
    DICES_TASK_LABELS,
)


def compute_normalized_entropy(annotations, num_labels):
    """
    Compute the normalized entropy of a list of annotations, example:
    annotations = [1, 1, 1, 2, 2, 2, 3, 3, 3]
    num_labels = 3
    normalized_entropy = 1.0
    :param annotations: a list of raw annotations for one item. Each element represents the label of one annotator.
    :param num_labels: how many classes are possible?
    :return: normalized entropy, a float between 0 and 1. A higher value means more disagreement.
    """
    # frequency for each label
    data_count = Counter(annotations)
    # number of annotations
    total_count = len(annotations)
    entropy = 0.0

    for count in data_count.values():
        # probability of each label
        probability = count / total_count
        # entropy of each label
        entropy -= probability * math.log2(probability)
    # normalized entropy to the number of labels, normalized entropy is between 0 and 1.
    normalized_entropy = entropy / math.log2(num_labels)
    return normalized_entropy


def create_MFTC_unaggregated():
    """
    This function creates a unaggregated version of the MFTC dataset. Each row represents a single annotation, indicating
    which moral foundations are present in the tweet according to the annotator. The dataframe is saved as a TSV file in
    the data/MFTC folder.
    :return: None
    """
    with open("data/MFTC/MFTC_V4_text.json") as f:
        data_mftc = json.load(f)
    records = []
    for corpus in data_mftc:
        corpus_name = corpus["Corpus"]
        for tweet in corpus["Tweets"]:
            tweet_id = tweet["tweet_id"]
            # create a one-hot vector for the moral foundations
            fountation_vector = [0] * len(MFTC_TARGET_TASKS)
            for annotation in tweet["annotations"]:
                # set the foundation to 1 if it is annotated
                annotated_foundations = annotation["annotation"].split(",")
                # use list comprehension to get the index of each foundation in the list of annotated foundations
                annotated_foundations_index = [
                    MFTC_TARGET_TASKS.index(foundation)
                    for foundation in annotated_foundations
                    if foundation in MFTC_TARGET_TASKS
                ]
                # set the foundation to 1 if it is annotated
                fountation_vector = [
                    1 if i in annotated_foundations_index else 0
                    for i in range(len(MFTC_TARGET_TASKS))
                ]
                records.append(
                    {
                        # Ignore tweet ID
                        "tweet_id": tweet_id,
                        "corpus": corpus_name,
                        "text": tweet["tweet_text"],
                        "annotation": annotation["annotation"],
                        "annotator": annotation["annotator"],
                        "non-moral": fountation_vector[0],
                        "care": fountation_vector[1],
                        "harm": fountation_vector[2],
                        "fairness": fountation_vector[3],
                        "cheating": fountation_vector[4],
                        "loyalty": fountation_vector[5],
                        "betrayal": fountation_vector[6],
                        "authority": fountation_vector[7],
                        "subversion": fountation_vector[8],
                        "purity": fountation_vector[9],
                        "degradation": fountation_vector[10],
                    }
                )
    df_mftc = pd.DataFrame.from_records(records)

    # Aggregate annotations for the same tweet and annotator into a single multi-label annotation
    new_records = []
    for group_id, group_df in tqdm.tqdm(df_mftc.groupby(["text", "annotator"])):
        if len(group_df) > 1:
            aggregated_labels = group_df[MFTC_TARGET_TASKS].max(axis=0)
            tweet_text = group_id[0]
            annotator = group_id[1]
            # We just take the first
            tweet_id = group_df["tweet_id"].unique()[0]
            corpus = group_df["corpus"].unique()[0]  # We just take the first
            new_records.append(
                {
                    "tweet_id": tweet_id,
                    "text": tweet_text,
                    "annotator": annotator,
                    "corpus": corpus,
                    **aggregated_labels.to_dict(),
                }
            )
        else:
            new_records.append(group_df.to_dict("records")[0])
    df_mftc = pd.DataFrame.from_records(new_records)

    # assert that there are no duplicated text/annotator pairs
    assert len(df_mftc.groupby(["text", "annotator"])) == len(df_mftc)

    df_mftc.to_csv("data/MFTC/MFTC_unaggregated.tsv", sep="\t", index=False)


def create_MFTC_aggregated():
    """
    This function creates an aggregated version of the MFTC dataset. Each row represents a single tweet, and the
    annotations are aggregated computing the probability of each moral foundation being present in the tweet. For
    each moral foundation, the normalized entropy is computed as well. All entropy scores are averaged so we obtain an
    average disagreement score for each tweet. The dataframe is saved as a TSV file in the data/MFTC folder.
    :return: None
    """
    mftc = pd.read_csv("data/MFTC/MFTC_unaggregated.tsv", sep="\t")

    # create an aggregated dataframe with the tweet_id and text only

    aggregated_data = []
    for tweet_id, group_data in tqdm.tqdm(mftc.groupby("tweet_id")):
        corpus = group_data["corpus"].unique()[0]
        # Convert group data to list once for non-repeated values@
        group_data_list = group_data.to_dict("records")[0]
        row_data = {
            "tweet_id": tweet_id,
            "text": group_data_list["text"],
            "num_annotators": group_data["annotator"].nunique(),
            "corpus": corpus,
        }
        for mf in MFTC_TARGET_TASKS:
            annotations = group_data[mf].tolist()
            soft_label = generate_soft_labels_raw_annotations(
                annotations=annotations, labels=range(2)
            )
            soft_label = soft_label[1]
            entropy = compute_normalized_entropy(annotations, 2)
            row_data[mf] = soft_label  # e.g., non-moral
            row_data[f"{mf}_disagreement"] = entropy  # e.g., non-moral_entropy
        aggregated_data.append(row_data)

    df = pd.DataFrame(aggregated_data)
    # add the mean of the disagreement scores for each tweet
    df["avg_disagreement"] = df[
        [f"{mf}_disagreement" for mf in MFTC_TARGET_TASKS]
    ].mean(axis=1)
    df.to_csv("data/MFTC/MFTC_aggregated.tsv", sep="\t", index=False)


def create_MHS_unaggregated():
    dataset = load_dataset("ucberkeley-dlab/measuring-hate-speech")
    df = dataset["train"].to_pandas()
    keep_df = df.groupby(["comment_id"]).filter(lambda x: len(x) >= 3 and len(x) <= 6)
    columns = [
        "comment_id",
        "annotator_id",
        "text",
        "respect",
        "dehumanize",
        "genocide",
    ]
    keep_df[columns].to_csv("data/MHS/MHS_unaggregated.tsv", sep="\t", index=False)


def create_MHS_aggregated(soft_label_method="normalize"):
    dataset = load_dataset("ucberkeley-dlab/measuring-hate-speech")
    df = dataset["train"].to_pandas()
    # Keep only those rows that have between 3 and 6 annotations
    keep_df = df.groupby(["comment_id"]).filter(lambda x: len(x) >= 3 and len(x) <= 6)
    aggregated_data = []
    for comment_id, group_data in tqdm.tqdm(keep_df.groupby("comment_id")):
        # Get the first element to obtain the text
        text = group_data.to_dict("records")[0]["text"]
        row_data = {
            "comment_id": comment_id,
            "text": text,
            "num_annotators": group_data["annotator_id"].nunique(),
        }
        for col in MHS_TARGET_TASKS:
            annotations = group_data[col].tolist()
            soft_label = generate_soft_labels_raw_annotations(
                annotations=annotations,
                labels=range(5),
                soft_label_method=soft_label_method,
            )
            entropy = compute_normalized_entropy(annotations, num_labels=5)

            for likert_score in sorted(soft_label.keys()):
                row_data[f"{col}_{likert_score}"] = soft_label[likert_score]
            row_data[f"{col}_disagreement"] = entropy
        aggregated_data.append(row_data)

    agg_df = pd.DataFrame(aggregated_data)
    # add the mean of the disagreement scores for each tweet
    agg_df["avg_disagreement"] = agg_df[
        [f"{col}_disagreement" for col in MHS_TARGET_TASKS]
    ].mean(axis=1)
    agg_df.to_csv(
        f"data/MHS/MHS_aggregated_{soft_label_method}.tsv", sep="\t", index=False
    )


def create_DICES_unaggregated():
    df = pd.read_csv("data/DICES/diverse_safety_adversarial_dialog_990.csv")
    # id,rater_id,rater_gender,rater_race,rater_raw_race,rater_age,rater_education,phase,item_id,answer_time_ms,answer_timestamp,context,response,degree_of_harm,harm_type,safety_gold,safety_gold_reason,Q1_whole_conversation_evaluation,Q2_harmful_content_medical_legal_financial_or_relationship_advice,Q2_harmful_content_dangerous_content,Q2_harmful_content_derogation_of_boat,Q2_harmful_content_insensitive,Q2_harmful_content_obscene_and_profane,Q2_harmful_content_personal_information,Q2_harmful_content_regulated_goods,Q2_harmful_content_sexually_suggestive_content,Q2_harmful_content_other,Q3_bias_targeting_beliefs,Q3_bias_targeting_gender_sexual_orientation,Q3_bias_targeting_inherited_attributes,Q3_bias_targeting_status,Q3_bias_other,Q4_misinformation,Q5_political_affiliation,Q6_policy_guidelines_polarizing_topics,Q6_policy_guidelines_making_endorsement,Q6_policy_guidelines_other,Q2_harmful_content_overall,Q3_bias_overall,Q6_policy_guidelines_overall,Q_overall

    columns = [
        "item_id",
        "rater_id",
        "context",
        "response",
        "degree_of_harm",
        "harm_type",
        "Q1_whole_conversation_evaluation",
        "Q_overall",
    ]
    # create_label_mapping
    mapping = {}

    for target_task in DICES_TARGET_TASKS:
        mapping[target_task] = {}
        for label, label_idx in DICES_TASK_LABELS[target_task].items():
            mapping[target_task][label] = label_idx

    # apply label mapping to target task columns
    for target_task in DICES_TARGET_TASKS:
        df[target_task] = df[target_task].apply(lambda x: mapping[target_task][x])

    df[columns].to_csv("data/DICES/DICES_unaggregated.tsv", sep="\t", index=False)


def create_DICES_aggregated(soft_label_method):
    df = pd.read_csv("data/DICES/diverse_safety_adversarial_dialog_990.csv")
    aggregated_data = []
    for item_id, group_data in tqdm.tqdm(df.groupby("item_id")):
        # Get the first element to obtain the text
        text = (
                group_data.to_dict("records")[0]["context"]
                + " "
                + group_data.to_dict("records")[0]["response"]
        )
        row_data = {
            "item_id": item_id,
            "text": text,
            "num_annotators": group_data["rater_id"].nunique(),
        }
        for col in DICES_TARGET_TASKS:
            annotations = group_data[col].tolist()
            soft_label = generate_soft_labels_raw_annotations(
                annotations=annotations,
                labels=DICES_TASK_LABELS[col].keys(),
                soft_label_method=soft_label_method,
            )
            entropy = compute_normalized_entropy(
                annotations, num_labels=len(DICES_TASK_LABELS[col])
            )

            for score in sorted(soft_label.keys()):
                row_data[f"{col}_{score}"] = soft_label[score]
            row_data[f"{col}_disagreement"] = entropy
        aggregated_data.append(row_data)

    agg_df = pd.DataFrame(aggregated_data)
    # Add the mean of the disagreement scores for each sample
    agg_df["avg_disagreement"] = agg_df[
        [f"{col}_disagreement" for col in DICES_TARGET_TASKS]
    ].mean(axis=1)
    agg_df.to_csv(
        f"data/DICES/DICES_aggregated_{soft_label_method}.tsv", sep="\t", index=False
    )


def create_splits(indices):
    # Create array of indices
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    val_idx, test_idx = train_test_split(test_idx, test_size=0.5, random_state=42)
    return pd.DataFrame(train_idx), pd.DataFrame(val_idx), pd.DataFrame(test_idx)


def create_kfold_splits(indices, k=5):
    """Return k-fold splits for the given indices."""
    kf = KFold(n_splits=k, random_state=42, shuffle=True)
    splits = []
    for train_idx, eval_idx in kf.split(indices):
        val_idx, test_idx = train_test_split(eval_idx, test_size=0.5, random_state=42)
        train_df = pd.DataFrame(indices[train_idx])
        val_df = pd.DataFrame(indices[val_idx])
        test_df = pd.DataFrame(indices[test_idx])
        splits.append((train_df, val_df, test_df))
    return splits


def create_splits_MHS():
    dataset = load_dataset("ucberkeley-dlab/measuring-hate-speech")
    df = dataset["train"].to_pandas()
    # Keep only those rows that have between 3 and 6 annotations
    keep_df = df.groupby(["comment_id"]).filter(lambda x: len(x) >= 3 and len(x) <= 6)
    train, val, test = create_splits(keep_df.comment_id.unique())
    # store to file
    train.to_csv("data/MHS/train_ids.csv", sep="\t", header=None, index=False)
    val.to_csv("data/MHS/val_ids.csv", sep="\t", header=None, index=False)
    test.to_csv("data/MHS/test_ids.csv", sep="\t", header=None, index=False)


def create_kfold_splits_MHS():
    dataset = load_dataset("ucberkeley-dlab/measuring-hate-speech")
    df = dataset["train"].to_pandas()
    # Keep only those rows that have between 3 and 6 annotations
    keep_df = df.groupby(["comment_id"]).filter(lambda x: len(x) >= 3 and len(x) <= 6)
    splits = create_kfold_splits(keep_df.comment_id.unique())
    for i, (train, val, test) in enumerate(splits):
        train.to_csv(f"data/MHS/train_ids_{i}.csv", sep="\t", header=None, index=False)
        val.to_csv(f"data/MHS/val_ids_{i}.csv", sep="\t", header=None, index=False)
        test.to_csv(f"data/MHS/test_ids_{i}.csv", sep="\t", header=None, index=False)


def filter_mftc(df, task):
    # check that non-moral is 0 if the morality class is 1
    x = (df["non-moral"] == 1) & (df[task] == 1)
    drop_idx = x[x].index
    print(f"Dropping {len(drop_idx)} rows")
    df = df.drop(drop_idx)
    return df


def get_label_ratio(df, task):
    moral_active_samples = df[task] > 0
    non_moral_samples = ~moral_active_samples
    return moral_active_samples.sum() / non_moral_samples.sum()


def balance_MFTC(df, task):
    this_df = df.copy()
    mftc_filtered = filter_mftc(this_df, task)
    moral_active_samples = mftc_filtered[task] > 0
    only_active_samples = mftc_filtered[moral_active_samples]
    tweet_ids = only_active_samples["tweet_id"]
    mftc_balanced = mftc_filtered[mftc_filtered["tweet_id"].isin(tweet_ids)]

    unselected_tweets = mftc_filtered[~mftc_filtered["tweet_id"].isin(tweet_ids)]
    # Strategy: we keep sampling new tweet ids randomly, untill the ratio is 0.5
    label_ratio = get_label_ratio(mftc_balanced, task)
    while label_ratio > 0.5:
        # Select new tweet idsf
        new_tweet_ids = unselected_tweets.sample(100)["tweet_id"]
        # Add samples with those ids to the balanced dataset
        mftc_balanced = pd.concat(
            [
                mftc_balanced,
                mftc_filtered[mftc_filtered["tweet_id"].isin(new_tweet_ids)],
            ]
        )
        # Remove the selected tweet ids from the unselected tweets
        tweet_ids = mftc_balanced["tweet_id"]
        unselected_tweets = mftc_filtered[~mftc_filtered["tweet_id"].isin(tweet_ids)]
        # Check the new ratio
        label_ratio = get_label_ratio(mftc_balanced, task)

    return mftc_balanced


def create_splits_MFTC(task):
    mftc = pd.read_csv("data/MFTC/MFTC_unaggregated.tsv", sep="\t")
    mftc = balance_MFTC(mftc, task)
    train, val, test = create_splits(mftc.tweet_id.unique())
    # store to file
    train.to_csv(f"data/MFTC/{task}_train_ids.csv", sep="\t", header=None, index=False)
    val.to_csv(f"data/MFTC/{task}_val_ids.csv", sep="\t", header=None, index=False)
    test.to_csv(f"data/MFTC/{task}_test_ids.csv", sep="\t", header=None, index=False)


def create_kfold_splits_MFTC(task):
    mftc = pd.read_csv("data/MFTC/MFTC_unaggregated.tsv", sep="\t")
    mftc = balance_MFTC(mftc, task)
    splits = create_kfold_splits(mftc.tweet_id.unique())
    for i, (train, val, test) in enumerate(splits):
        train.to_csv(
            f"data/MFTC/{task}_train_ids_{i}.csv", sep="\t", header=None, index=False
        )
        val.to_csv(
            f"data/MFTC/{task}_val_ids_{i}.csv", sep="\t", header=None, index=False
        )
        test.to_csv(
            f"data/MFTC/{task}_test_ids_{i}.csv", sep="\t", header=None, index=False
        )


def create_splits_DICES():
    dices_990 = pd.read_csv("data/DICES/diverse_safety_adversarial_dialog_990.csv")
    train, val, test = create_splits(dices_990.item_id.unique())
    # store to file
    train.to_csv("data/DICES/train_ids.csv", sep="\t", header=None, index=False)
    val.to_csv("data/DICES/val_ids.csv", sep="\t", header=None, index=False)
    test.to_csv("data/DICES/test_ids.csv", sep="\t", header=None, index=False)


def create_kfold_splits_DICES():
    dices_990 = pd.read_csv("data/DICES/diverse_safety_adversarial_dialog_990.csv")
    splits = create_kfold_splits(dices_990.item_id.unique())
    for i, (train, val, test) in enumerate(splits):
        train.to_csv(
            f"data/DICES/train_ids_{i}.csv", sep="\t", header=None, index=False
        )
        val.to_csv(f"data/DICES/val_ids_{i}.csv", sep="\t", header=None, index=False)
        test.to_csv(f"data/DICES/test_ids_{i}.csv", sep="\t", header=None, index=False)


def get_suggested_sample_size(num_annotations, num_samples, mode="ACAL"):
    if mode == "ACAL":
        a = np.floor(0.05 * num_annotations)
        if num_samples < a:
            return num_samples
        return a
    elif mode == "AL":
        return np.floor(0.1 * num_samples)


def print_data_table_MHS():
    # samples
    # annotators
    # annotations
    # annotations per sample
    mhs = pd.read_csv("data/MHS/MHS_unaggregated.tsv", sep="\t")
    print(f"Number of unique samples: {mhs.comment_id.nunique()}")
    print(f"Number of unique annotators: {mhs.annotator_id.nunique()}")
    print(f"Number of annotations: {len(mhs)}")
    print(
        f"Average number of annotations per sample: {len(mhs) / mhs.comment_id.nunique():.2f}"
    )
    sample_size_acal = get_suggested_sample_size(len(mhs), mhs.comment_id.nunique())
    print(f"Suggested sample size (ACAL): {sample_size_acal}")
    sample_size_al = get_suggested_sample_size(
        len(mhs), mhs.comment_id.nunique(), mode="AL"
    )
    print(f"Suggested sample size (AL): {sample_size_al}")


def print_data_table_MFTC(task):
    mftc_task_train = pd.read_csv(
        f"data/MFTC/{task}_train_ids_0.csv", index_col=None, header=None
    )[0].to_list()
    mftc_task_val = pd.read_csv(
        f"data/MFTC/{task}_val_ids_0.csv", index_col=None, header=None
    )[0].to_list()
    mftc_task_test = pd.read_csv(
        f"data/MFTC/{task}_test_ids_0.csv", index_col=None, header=None
    )[0].to_list()
    print(f"Task: {task}")

    mftc = pd.read_csv("data/MFTC/MFTC_unaggregated.tsv", sep="\t")
    # select only the samples that are in the train, val, and test sets
    mftc_task_train = mftc[mftc["tweet_id"].isin(mftc_task_train)]
    mftc_task_val = mftc[mftc["tweet_id"].isin(mftc_task_val)]
    mftc_task_test = mftc[mftc["tweet_id"].isin(mftc_task_test)]
    mftc_task = pd.concat([mftc_task_train, mftc_task_val, mftc_task_test])
    print(f"Number of unique samples: {mftc_task.tweet_id.nunique()}")
    print(f"Number of unique annotators: {mftc_task.annotator.nunique()}")
    print(f"Number of annotations: {len(mftc_task)}")
    print(
        f"Average number of annotations per sample: {len(mftc_task) / mftc_task.tweet_id.nunique():.2f}"
    )
    sample_size_acal = get_suggested_sample_size(
        len(mftc_task_train), mftc_task_train.tweet_id.nunique()
    )
    print(f"Suggested sample size (ACAL): {sample_size_acal}")
    sample_size_al = get_suggested_sample_size(
        len(mftc_task_train), mftc_task_train.tweet_id.nunique(), mode="AL"
    )
    print(f"Suggested sample size (AL): {sample_size_al}")


def print_data_table_DICES():
    dices_990 = pd.read_csv("data/DICES/diverse_safety_adversarial_dialog_990.csv")
    print(f"Number of unique samples: {dices_990.item_id.nunique()}")
    print(f"Number of unique annotators: {dices_990.rater_id.nunique()}")
    print(f"Number of annotations: {len(dices_990)}")
    print(
        f"Average number of annotations per sample: {len(dices_990) / dices_990.item_id.nunique():.2f}"
    )

    train_idx = pd.read_csv(f"data/DICES/train_ids_0.csv", index_col=None, header=None)[
        0
    ].to_list()
    train_dices = dices_990[dices_990["item_id"].isin(train_idx)]
    sample_size_acal = get_suggested_sample_size(
        len(train_dices), train_dices.item_id.nunique()
    )
    print(f"Suggested sample size (ACAL): {sample_size_acal}")
    sample_size_al = get_suggested_sample_size(
        len(train_dices), train_dices.item_id.nunique(), mode="AL"
    )
    print(f"Suggested sample size (AL): {sample_size_al}")


if __name__ == "__main__":
    generate_new = False

    if generate_new:
        create_kfold_splits_DICES()
        create_splits_DICES()
        create_DICES_aggregated("normalize")
        create_DICES_aggregated("softmax")
        create_DICES_unaggregated()

        create_kfold_splits_MHS()
        create_splits_MHS()
        create_MHS_aggregated("normalize")
        create_MHS_aggregated("softmax")
        create_MHS_unaggregated()

        create_MFTC_unaggregated()
        create_MFTC_aggregated()

    print("== DICES ==")
    print_data_table_DICES()

    print("== MHS ==")
    print_data_table_MHS()

    print("== MFTC ==")
    for task in ["care", "loyalty", "betrayal"]:
        if generate_new:
            create_splits_MFTC(task)
            create_kfold_splits_MFTC(task)
        print_data_table_MFTC(task)
