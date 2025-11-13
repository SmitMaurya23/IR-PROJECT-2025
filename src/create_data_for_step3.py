import os
import pandas as pd
import glob
import numpy as np

from util import load_samples


N_NEGATIVES = int(os.getenv("N_NEGATIVES", 1))
N_POSITIVE_REPLICATES = int(os.getenv("N_POSITIVE_REPLICATES", 1))

RAW_DATA_DIR = "data"
QUERY_PATH = os.path.join(RAW_DATA_DIR, "COLIEE2025statute_data-English/train")
ARTICLE_PATH = os.path.join(RAW_DATA_DIR, "full_en_civil_code_df_24.csv")

STEP2_DIR = "checkpoints/step2_rankllama_retrieval"
STEP2_TRAIN = os.path.join(STEP2_DIR, "train_df_threshold.-3.5.csv")
STEP2_EVAL = os.path.join(STEP2_DIR, "eval_df_threshold.-3.5.csv")
STEP2_TEST = os.path.join(STEP2_DIR, "test_df_threshold.-3.5.csv")

STEP3_DATA_DIR = "checkpoints/step3_final_retrieval/data"


def make_sub_train(train_df_step3, sub_train_ratio):
    query_ids = train_df_step3["query_id"].unique()
    query_ids.sort()

    np.random.seed(42)
    chosen = np.random.choice(query_ids, int(sub_train_ratio*len(query_ids)), replace=False)

    return train_df_step3[train_df_step3["query_id"].isin(chosen)]


def top_filter(group_df, n_positive_replicates, n_negatives):
    positive_df = group_df[group_df["label"] == 1]
    negative_df = group_df[group_df["label"] == 0]

    negative_df = negative_df.sort_values(by=["step2_score", "step1_score"],
                                          ascending=False,
                                          ignore_index=True)
    negative_df = negative_df[:n_negatives]

    return pd.concat([*[positive_df]*n_positive_replicates, negative_df], ignore_index=True)


if __name__ == '__main__':
    # 1. Load data
    # Load the article data
    en_article_df = pd.read_csv(ARTICLE_PATH)
    en_article_df.rename(columns={"article": "article_id", "content": "article_content"}, inplace=True)

    # Load the query data
    query_files = glob.glob(f"{QUERY_PATH}/*.xml")

    queries = []
    for query_file in query_files:
        queries += load_samples(query_file)

    en_query_df = pd.DataFrame(queries)
    en_query_df = en_query_df.rename(columns={"index": "query_id",
                                              "content": "query_content",
                                              "result": "task3_label",
                                              "label": "task4_label"})


    # 2. Load train/eval/test splits from the step 2 of the pipeline
    train_df_step2 = pd.read_csv(STEP2_TRAIN).drop(columns=["keep"])
    eval_df_step2 = pd.read_csv(STEP2_EVAL).drop(columns=["keep"])
    test_df_step2 = pd.read_csv(STEP2_TEST).drop(columns=["keep"])

    # # TODO: Remove this block
    # # Merge R04 to train_df such that:
    # # - new train_df_step2 = old train_df_step2 + R04 from eval_df_step2
    # # - new eval_df_step2 = old test_df_step2
    # # - new test_df_step2 = old test_df_step2
    # train_df_step2 = pd.concat([train_df_step2, eval_df_step2], ignore_index=True)
    # eval_df_step2 = test_df_step2.copy()
    # test_df_step2 = test_df_step2.copy()
    # #


    # 3. Make data for step 3
    train_df_step3 = train_df_step2.groupby("query_id")[train_df_step2.columns]\
                                   .apply(top_filter, N_POSITIVE_REPLICATES, N_NEGATIVES)\
                                   .reset_index(drop=True)\
                                   .sample(frac=1.0, random_state=42, ignore_index=True)
    eval_df_step3 = eval_df_step2.groupby("query_id")[eval_df_step2.columns]\
                                 .apply(top_filter, 1, N_NEGATIVES)\
                                 .reset_index(drop=True)\
                                 .sample(frac=1.0, random_state=42, ignore_index=True)
    test_df_step3 = test_df_step2.groupby("query_id")[test_df_step2.columns]\
                                 .apply(top_filter, 1, N_NEGATIVES)\
                                 .reset_index(drop=True)\
                                 .sample(frac=1.0, random_state=42, ignore_index=True)

    # 4. Get query and article content
    train_df_step3 = train_df_step3.merge(en_query_df[["query_id", "query_content"]], how="left")\
                                   .merge(en_article_df[["article_id", "article_content"]], how="left")
    eval_df_step3 = eval_df_step3.merge(en_query_df[["query_id", "query_content"]], how="left")\
                                 .merge(en_article_df[["article_id", "article_content"]], how="left")
    test_df_step3 = test_df_step3.merge(en_query_df[["query_id", "query_content"]], how="left")\
                                 .merge(en_article_df[["article_id", "article_content"]], how="left")

    sub_train_df_step3 = make_sub_train(train_df_step3, 0.3)


    # 5. Save the dataset
    # train_path = os.path.join(STEP3_DATA_DIR, f"train.n_positive_{N_POSITIVE_REPLICATES}.n_negatives_{N_NEGATIVES}.jsonl")
    # eval_path = os.path.join(STEP3_DATA_DIR, f"eval.n_positive_{N_POSITIVE_REPLICATES}.n_negatives_{N_NEGATIVES}.jsonl")
    # test_path = os.path.join(STEP3_DATA_DIR, f"test.n_positive_{N_POSITIVE_REPLICATES}.n_negatives_{N_NEGATIVES}.jsonl")

    train_path = os.path.join(STEP3_DATA_DIR, f"train.jsonl")
    eval_path = os.path.join(STEP3_DATA_DIR, f"eval.jsonl")
    test_path = os.path.join(STEP3_DATA_DIR, f"test.jsonl")
    sub_train_path = os.path.join(STEP3_DATA_DIR, f"sub_train.jsonl")

    train_df_step3.to_json(train_path, orient="records", lines=True)
    eval_df_step3.to_json(eval_path, orient="records", lines=True)
    test_df_step3.to_json(test_path, orient="records", lines=True)
    sub_train_df_step3.to_json(sub_train_path, orient="records", lines=True)
