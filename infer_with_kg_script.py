# %%
import os
import pandas as pd
import numpy as np
import glob
import itertools
import pickle
from tqdm import tqdm
import torch

from FlagEmbedding import BGEM3FlagModel

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel, PeftConfig

from scipy.special import softmax

from src.util import load_samples
from src.data.data_collator import LegalDataCollatorWithPadding

# %% [markdown]
# ## Hyperparameters

# %%
TEST2025_EN_PATH = "data/COLIEE2025statute_data-English/train/R05_en.xml"

SELECTED_ID = "R05"

RAW_DATA_DIR = "data"
DATA_OUTPUT_DIR = "data/synthesys"
QUERY_PATH = os.path.join(RAW_DATA_DIR, "COLIEE2025statute_data-English/train")
ARTICLE_PATH = os.path.join(RAW_DATA_DIR, "full_en_civil_code_df_24.csv")

CHECKPOINT_DIR = "checkpoints_kg"
STEP1_CHECKPOINT_DIR = f"{CHECKPOINT_DIR}/step1_bge_pre_retrieval"
STEP2_CHECKPOINT_DIR = f"{CHECKPOINT_DIR}/step2_rankllama_retrieval"
STEP3_CHECKPOINT_DIR = f"{CHECKPOINT_DIR}/step3_final_retrieval"

ACCEPTED_MODELS = [
    "e5_mistral_7b_instruct",
    "gemma_2_9b_it",
    "gemma_2_27b_it",
    "phi_3_medium_4k_instruct",
]


# TODO: fix bug
BUG_ARTICLE_POSTFIX = "(1)"  # In the R04's task 3 label, there are some ground truth labels having "(1)" postfix. We need to remove them.


INFERENCE_DIR = f"{CHECKPOINT_DIR}/inference"


# Step 1
BGE_TOP = 100
BGE_SEQUENCE_MAX_LENGTH = 1024
HISTOGRAM_N_POSITIVE_REPLICATES = 300


# Step 2
RANKLLAMA_MAX_LENGTH = 1024
RANKLLAMA_THRESHOLD = -3.5  # preserve about 50 candidates for each query
RANKLLAMA_TOP = 50


# Step 4
CUT_OFF_THRESHOLD = 0.3687529996711829
WEIGHTS = np.array([0.23716786, 0.21487627, 0.3068145 , 0.24114137])

# %%
WEIGHTS

# %% [markdown]
# ## Step 0: Create dataset

# %%
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

# %%


# %%
test_query_df = en_query_df[en_query_df["query_id"].str.startswith(SELECTED_ID)].copy(deep=True)

del en_query_df

if len(test_query_df) == 0:
    queries = load_samples(TEST2025_EN_PATH)

    test_query_df = pd.DataFrame(queries)
    test_query_df = test_query_df.rename(columns={"index": "query_id",
                                                    "content": "query_content",
                                                    "result": "task3_label",
                                                    "label": "task4_label"})


# %%
if "task3_label" in test_query_df.columns:
    test_query_df = test_query_df.drop(columns=["task3_label"])

# %%


# %% [markdown]
# ## Step 1: BGE Pre-retrieval
#
# ### 1.1. BGE Embedding

# %%
model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True, device='cuda')

# article embedding
article_embeddings = model.encode(en_article_df["article_content"].tolist(),
                                  batch_size=32,
                                  max_length=BGE_SEQUENCE_MAX_LENGTH
                                  )['dense_vecs']


# query embedding
query_embeddings = model.encode(test_query_df["query_content"].tolist(),
                                batch_size=32,
                                max_length=BGE_SEQUENCE_MAX_LENGTH
                                )['dense_vecs']

# %%
import joblib
###LOAD####
article_embeddings = joblib.load(f"./{STEP1_CHECKPOINT_DIR}/article_embeddings.pkl")
query_embeddings = joblib.load(f"./{STEP1_CHECKPOINT_DIR}/query_embeddings.pkl")

# %%
article_embedding_dict = dict(zip(en_article_df["article_id"].tolist(), article_embeddings))
query_embedding_dict = dict(zip(test_query_df["query_id"].tolist(), query_embeddings))

# %% [markdown]
# ### 1.2. Retrieval with Histogram-based Gradient Boosting
# Data

# %%
def make_pairs(query_id, labels):
    return list(itertools.product([query_id], labels))


def distance_function(query_emb, article_emb):
    return query_emb - article_emb


def get_distance(query_id, article_id, query_embedding_dict, article_embedding_dict):
    query_emb = query_embedding_dict[query_id]
    article_emb = article_embedding_dict[article_id]

    return distance_function(query_emb, article_emb)


query_article_pairs = test_query_df.apply(lambda x: make_pairs(x["query_id"], en_article_df["article_id"].values), axis=1)
query_article_pairs = sum(query_article_pairs, [])

X_test = list(map(lambda x: get_distance(*x, query_embedding_dict, article_embedding_dict), query_article_pairs))
X_test = np.array(X_test)

# %%
import joblib

# %% [markdown]
# Infer

# %%
def get_top_preds(group):
    group = group.sort_values("step1_score", ascending=False)

    # cut_off_score = group.iloc[BGE_TOP]["step1_score"]
    # group["keep"] = group["step1_score"] > cut_off_score - 1e-5

    group["keep"] = False
    group.iloc[:BGE_TOP, group.columns.get_loc("keep")] = True

    return group


model = joblib.load(open(f"{STEP1_CHECKPOINT_DIR}/histogram_classifier.pkl", "rb"))
y_pred = model.predict_proba(X_test)

test_df_step1 = pd.DataFrame(query_article_pairs, columns=["query_id", "article_id"])
test_df_step1["step1_score"] = y_pred[:, 1]

test_df_step1 = test_df_step1.groupby("query_id")[test_df_step1.columns.tolist()]\
                             .apply(get_top_preds)\
                             .reset_index(drop=True)

test_df_step1 = test_df_step1[test_df_step1["keep"] == True]

# %% [markdown]
# ## Step 2: RankLlama for 2nd-stage retrieval

# %%
def get_model(peft_model_name):
    config = PeftConfig.from_pretrained(peft_model_name)
    base_model = AutoModelForSequenceClassification.from_pretrained(config.base_model_name_or_path,
                                                                    num_labels=1,
                                                                    torch_dtype=torch.bfloat16,
                                                                    device_map="auto")
    model = PeftModel.from_pretrained(base_model, peft_model_name)
    model = model.merge_and_unload()
    model.eval()
    return model


def make_rankllama_prompt(row):
    """
    Build one of two prompts:
    - Prompt 1 (original candidate)
    - Prompt 2 (KG-connected candidate with relation + original relevance)
    Expects row to contain at least:
      - row['query_content'] or fallback ''
      - row['article_content'] or fallback ''
      - row.get('is_kg_added', False)
      - row.get('relation') for KG edges (optional)
      - row.get('source_relevance_label') or row.get('relevance_label') for original relevance (optional)
    """
    query_text = str(row.get("query_content", "") or "")
    article_text = str(row.get("article_content", "") or "")

    # normalize relation and relevance inputs
    relation_type = str(row.get("relation") or row.get("relation_type") or "N/A")
    # prefer explicit source_relevance_label (e.g., from parent), else fall back to any relevance_label
    relevance_level = row.get("source_relevance_label") or row.get("relevance_label") or row.get("source_relevance") or "Unknown"
    relevance_level = str(relevance_level)

    # Prompt 1 (original article)
    prompt1 = f"""You are a legal retrieval assistant. Given a query and
a legal article, determine how relevant the article is
to answering the query.
Query: {query_text}
Article: {article_text}
Please provide a relevance score from 0 to 1."""
    # Prompt 2 (KG-connected article)
    prompt2 = f"""You are a legal retrieval assistant .Given a query and
a legal article connected through a legal knowledge
graph , determine its relevance to the query.
Query:{query_text}
Article:{article_text}
Relationtooriginalarticle:{relation_type}
Original article relevance:{relevance_level}
Please evaluate how relevant this connected article is
to the query on a scale 0 to 1."""

    return prompt2 if bool(row.get("is_kg_added", False)) else prompt1




import torch
import numpy as np
from tqdm import tqdm

import torch  # ensure torch is available for dtype/device handling

def get_scores(model, tokenizer, df, batch_size, max_len, data_collator, debug_dtype=False):
    """
    Robust get_scores: converts logits to float32 before moving to numpy to avoid
    errors when logits are in bfloat16/float16.
    Uses dynamic prompting via make_rankllama_prompt(row) which should inspect
    row['is_kg_added'], relation, source_article_id, source_step1_score, etc.
    """
    model.eval()
    device = next(model.parameters()).device
    scores = []

    for i in tqdm(range(0, len(df), batch_size), desc="scoring"):
        batch = df.iloc[i:i+batch_size]

        # Build prompts using dynamic prompt factory which uses KG metadata when present
        # make_rankllama_prompt(row) must return a single string prompt for that row
        texts = batch.apply(lambda r: make_rankllama_prompt(r), axis=1).tolist()

        # Tokenize -> list-of-dicts (keeps your existing collator flow)
        tokenized = tokenizer(texts, max_length=max_len, truncation=True)
        inputs_list = [dict(zip(tokenized.keys(), vals)) for vals in zip(*tokenized.values())]

        collated = data_collator(inputs_list)
        collated = {k: v.to(device) for k, v in collated.items()}

        with torch.no_grad():
            outputs = model(**collated)

        logits = getattr(outputs, "logits", None)

        # fallback default scores
        if logits is None:
            batch_scores = [0.0] * len(texts)
        else:
            # Ensure logits are on CPU and in float32 before converting to numpy
            # (handles bfloat16/float16 returned by some model setups)
            if isinstance(logits, torch.Tensor):
                # move to cpu and cast to float32
                logits_cpu = logits.detach().cpu().to(torch.float32)

                if debug_dtype:
                    print("logits original dtype:", logits.dtype, "-> converted dtype:", logits_cpu.dtype)

                if logits_cpu.dim() == 2:
                    if logits_cpu.size(1) == 1:
                        batch_scores = logits_cpu[:, 0].numpy()
                    elif logits_cpu.size(1) == 2:
                        probs = torch.softmax(logits_cpu, dim=1)[:, 1]
                        batch_scores = probs.numpy()
                    else:
                        probs = torch.softmax(logits_cpu, dim=1)[:, 1] if logits_cpu.size(1) > 1 else logits_cpu[:, 0]
                        batch_scores = probs.numpy()
                elif logits_cpu.dim() == 1:
                    batch_scores = logits_cpu.numpy()
                else:
                    # fallback for token logits etc.
                    batch_scores = logits_cpu.view(logits_cpu.size(0), -1)[:, 0].numpy()
            else:
                # non-tensor logits (rare)
                batch_scores = np.array(logits).astype(np.float32).squeeze()

        # ensure python floats and length matches batch
        batch_scores_arr = np.asarray(batch_scores).squeeze()
        # if single scalar returned for batch, replicate to batch size
        if batch_scores_arr.shape == ():
            batch_scores_arr = np.repeat(float(batch_scores_arr), len(texts))
        # If lengths mismatch, pad/truncate conservatively
        if len(batch_scores_arr) != len(texts):
            if len(batch_scores_arr) < len(texts):
                batch_scores_arr = np.concatenate([batch_scores_arr, np.zeros(len(texts) - len(batch_scores_arr), dtype=np.float32)])
            else:
                batch_scores_arr = batch_scores_arr[:len(texts)]

        scores.extend([float(x) for x in batch_scores_arr])

    return np.array(scores).squeeze().tolist()




def get_top_preds(group):
    group = group.sort_values("step2_score", ascending=False)

    cut_off_score = group.iloc[RANKLLAMA_TOP]["step2_score"]
    group["keep"] = group["step2_score"] > cut_off_score - 1e-5

    return group

# %% [markdown]
# Infer

# %%
# Load the tokenizer and model
model = get_model('castorini/rankllama-v1-7b-lora-passage')
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')

tokenizer.pad_token = "<unk>"
model.config.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

# %%
from kg.phase1kg.db_connection import Neo4jConnection

conn = Neo4jConnection("neo4j:// 10.50.22.71:7687", "neo4j", "Smitmaurya@24")
result = conn.execute_query("RETURN 'Connected to Neo4j!' AS message")
for record in result:
    print(record["message"])


# %%
# Requires: neo4j, pandas, numpy, tqdm
from neo4j import GraphDatabase
import pandas as pd
import numpy as np
from tqdm import tqdm
import json

NODE_ID_PROP = "number"  # change if your Article node uses a different id prop
driver = conn.driver  # your Neo4j connection driver

# ----------------- helpers -----------------
def _parse_vector(v):
    """Safely parse a Neo4j-stored vector which may be a list, numpy array or JSON/string."""
    if v is None:
        return None
    if isinstance(v, (list, tuple, np.ndarray)):
        return np.asarray(v, dtype=np.float32)
    if isinstance(v, str):
        try:
            # try json first
            parsed = json.loads(v)
            return np.asarray(parsed, dtype=np.float32)
        except Exception:
            try:
                # fallback to eval (last resort)
                parsed = eval(v)
                return np.asarray(parsed, dtype=np.float32)
            except Exception:
                return None
    # any other type
    try:
        return np.asarray(v, dtype=np.float32)
    except Exception:
        return None

def cosine_sim(a, b):
    """Robust cosine similarity (returns 0.0 if any vector is invalid)."""
    if a is None or b is None:
        return 0.0
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.size == 0 or b.size == 0:
        return 0.0
    denom = (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)
    return float(np.dot(a, b) / denom)

def assign_relevance_labels_by_rank(df, score_col="step2_score"):
    """
    For each query, label by rank:
      - top 30 -> 'Highly Relevant'
      - next 30 -> 'Moderately Relevant'
      - rest -> 'Less Relevant'
    Uses score_col; if absent falls back to 'step1_score'.
    """
    sc = score_col if score_col in df.columns else "step1_score"
    def _label(group):
        group = group.sort_values(sc, ascending=False).reset_index(drop=True)
        group["relevance_label"] = "Less Relevant"
        # top 30 -> Highly
        group.loc[0:29, "relevance_label"] = "Highly Relevant"
        # next 30 -> Moderately (30..59)
        if len(group) > 30:
            group.loc[30:59, "relevance_label"] = "Moderately Relevant"
        return group
    if df.shape[0] == 0:
        return df
    labeled = df.groupby("query_id", as_index=False, group_keys=False).apply(_label)
    return labeled.reset_index(drop=True)

# ----------------- candidate_expansion with vector gating -----------------
def candidate_expansion(temp_df,
                        en_article_df: pd.DataFrame,
                        en_query_df: pd.DataFrame,
                        max_candidates: int = 150,
                        parent_limit: int = 30,
                        batch_size: int = 16,
                        bge_similarity_threshold: float = 0.6):
    """
    Expand temp_df (top-K per query) by traversing parents top->bottom (only first `parent_limit`
    parents) and adding 1-hop Article neighbors via REFERENCES|REFERENCED_BY **only if**
    cosine_sim(source_node.vector, neighbor.node.vector) > bge_similarity_threshold.

    Also assigns relevance labels at the end (prefers step2_score if present).
    """
    df = temp_df.copy(deep=True).reset_index(drop=True)
    df["step1_score"] = df["step1_score"].astype(float)

    # Build gold-map: query_id -> set(positive_article_ids)
    gold_map = {}
    if en_query_df is not None and "query_id" in en_query_df.columns and "task3_label" in en_query_df.columns:
        for _, r in en_query_df.iterrows():
            qid = r["query_id"]
            vals = r["task3_label"]
            if vals is None or (isinstance(vals, float) and np.isnan(vals)):
                gold_map[qid] = set()
            elif isinstance(vals, (list, np.ndarray, pd.Series)):
                gold_map[qid] = set([str(x) for x in vals])
            elif isinstance(vals, str):
                try:
                    parsed = eval(vals)
                    if isinstance(parsed, (list, tuple, set, np.ndarray)):
                        gold_map[qid] = set([str(x) for x in parsed])
                    else:
                        gold_map[qid] = set([str(parsed)])
                except Exception:
                    parts = [v.strip() for v in vals.strip("[] ").split(",") if v.strip()]
                    gold_map[qid] = set(parts)
            else:
                gold_map[qid] = set([str(vals)])
    else:
        gold_map = {}

    # Prepare article content/label maps if available
    article_content_map = {}
    if en_article_df is None:
        en_article_df = globals().get("en_article_df", None)
    if en_article_df is not None:
        if "article_id" in en_article_df.columns and "article_content" in en_article_df.columns:
            article_content_map = dict(zip(en_article_df["article_id"].astype(str), en_article_df["article_content"]))

    # Build initial grouped dict from provided top-K
    grouped = {}
    for _, row in df.iterrows():
        q = row["query_id"]
        aid = str(row["article_id"])
        grouped.setdefault(q, {})
        grouped[q].setdefault(aid, {
            "query_id": q,
            "article_id": aid,
            "step1_score": float(row["step1_score"]),
            "label": row.get("label", 0.0),
            "keep": row.get("keep", True),
            "query_content": row.get("query_content", None),
            "article_content": row.get("article_content", article_content_map.get(aid, None)),
            "parent_node": None,
            "relation": None
        })

    # neighbor cache: src_id -> list of dicts {nid, rel, nvec (np.array), sim (float)}
    neighbor_cache = {}

    def fetch_neighbors_batch(article_ids):
        """
        Batch fetch neighbors with relation types and node.vector for both source and neighbors.
        Stores neighbor_cache[src] = list of dicts {nid, rel, nvec, sim} where sim is computed
        between src_vector and nvec (if both available).
        Returns dict src->list((nid, rel, nvec, sim)).
        """
        to_query = [aid for aid in article_ids if str(aid) not in neighbor_cache]
        out = {str(aid): [] for aid in article_ids}
        if not to_query:
            for aid in article_ids:
                out[str(aid)] = neighbor_cache.get(str(aid), [])
            return out
        if driver is None:
            for aid in article_ids:
                out[str(aid)] = neighbor_cache.get(str(aid), [])
            return out

        # Cypher: return src vector and neighbors with their vectors
        cypher = f"""
        UNWIND $aids AS aid
        MATCH (a:Article {{{NODE_ID_PROP}: aid}})
        OPTIONAL MATCH (a)-[r:REFERENCES|REFERENCED_BY]-(n:Article)
        RETURN aid AS src, a.vector AS src_vector,
               collect(DISTINCT {{id: n.{NODE_ID_PROP}, rel: type(r), nvec: n.vector}}) AS neighbors
        """
        for i in range(0, len(to_query), batch_size):
            chunk = to_query[i:i+batch_size]
            params_chunk = [int(x) if NODE_ID_PROP == "number" and str(x).isdigit() else x for x in chunk]
            with driver.session() as session:
                res = session.run(cypher, aids=params_chunk)
                for r in res:
                    src = str(r["src"])
                    src_vec_raw = r.get("src_vector", None)
                    src_vec = _parse_vector(src_vec_raw)
                    neighs = r.get("neighbors", []) or []
                    parsed = []
                    for item in neighs:
                        nid = item.get("id")
                        if nid is None:
                            continue
                        nid = str(nid)
                        rel = item.get("rel", None)
                        nvec_raw = item.get("nvec", None)
                        nvec = _parse_vector(nvec_raw)
                        sim = cosine_sim(src_vec, nvec) if (src_vec is not None and nvec is not None) else 0.0
                        parsed.append({"nid": nid, "rel": rel, "nvec": nvec, "sim": sim})
                    neighbor_cache[src] = parsed

        for aid in article_ids:
            out[str(aid)] = neighbor_cache.get(str(aid), [])
        return out

    # Iterate queries with progress; parents limited to parent_limit
    for qid in tqdm(list(grouped.keys()), desc="queries"):
        rows = grouped[qid]
        parents_sorted = sorted(rows.values(), key=lambda x: -float(x["step1_score"]))
        parents_limited = parents_sorted[:parent_limit]
        cand_count = len(rows)

        parent_ids = [p["article_id"] for p in parents_limited]

        for pid in tqdm(parent_ids, desc=f"parents for {qid}", leave=False):
            if cand_count >= max_candidates:
                break
            parent_row = rows[pid]
            parent_score = float(parent_row["step1_score"])

            neigh_map = fetch_neighbors_batch([pid])
            neighbors = neigh_map.get(str(pid), [])

            for nmeta in neighbors:
                if cand_count >= max_candidates:
                    break
                n_id = str(nmeta["nid"])
                rel = nmeta.get("rel")
                sim_src_neighbor = float(nmeta.get("sim", 0.0))

                # Skip self and require similarity threshold
                if n_id == str(pid):
                    continue
                if sim_src_neighbor <= bge_similarity_threshold:
                    # do NOT add or update neighbor if similarity not above threshold
                    continue

                # If neighbor not present, add it
                if n_id not in rows:
                    gold_set = gold_map.get(qid, set())
                    neighbor_label = 1.0 if n_id in gold_set else 0.0
                    rows[n_id] = {
                        "query_id": qid,
                        "article_id": n_id,
                        "step1_score": parent_score,   # inherit parent's score
                        "label": neighbor_label,
                        "keep": True,
                        "query_content": parent_row.get("query_content"),
                        "article_content": article_content_map.get(n_id, None),
                        "parent_node": pid,
                        "relation": rel,
                        # store similarity and indication that it was KG-added
                        "source_sim": sim_src_neighbor,
                        "is_kg_added": True
                    }
                    cand_count += 1
                else:
                    # already present: update only if parent's score is higher AND sim > threshold
                    existing_score = float(rows[n_id]["step1_score"])
                    if parent_score > existing_score and sim_src_neighbor > bge_similarity_threshold:
                        rows[n_id]["step1_score"] = parent_score
                        rows[n_id]["parent_node"] = pid
                        rows[n_id]["relation"] = rel
                        rows[n_id]["source_sim"] = sim_src_neighbor
                        rows[n_id]["is_kg_added"] = True
                        # keep gold label if present
                        if n_id in gold_map.get(qid, set()):
                            rows[n_id]["label"] = 1.0
                        # else preserve existing label if any

    # flatten grouped into DataFrame
    expanded_rows = []
    for q, amap in grouped.items():
        for aid, vals in amap.items():
            expanded_rows.append(vals)
    expanded_df = pd.DataFrame(expanded_rows)

    # ensure expected columns present
    expected_cols = ["query_id", "article_id", "step1_score", "label", "keep",
                     "query_content", "article_content", "parent_node", "relation",
                     "is_kg_added", "source_sim"]
    for col in expected_cols:
        if col not in expanded_df.columns:
            expanded_df[col] = None

    # sort within each query by descending step1_score
    expanded_df = expanded_df.groupby("query_id", sort=False, group_keys=False).apply(
        lambda g: g.sort_values("step1_score", ascending=False)
    ).reset_index(drop=True)

    # Assign relevance labels using step2_score if it exists (RankLLaMA output),
    # otherwise fallback to step1_score.
    score_col_to_use = "step2_score" if "step2_score" in expanded_df.columns else "step1_score"
    expanded_df = assign_relevance_labels_by_rank(expanded_df, score_col=score_col_to_use)

    return expanded_df

# %%
data_collator = LegalDataCollatorWithPadding(tokenizer)

test_df_step2 = test_df_step1.copy(deep=True)

# %%
test_df_step2

# %%
expanded_test_df_step2=candidate_expansion(test_df_step2,en_article_df,test_query_df)

# %%

expanded_test_df_step2 = expanded_test_df_step2.merge(test_query_df[["query_id", "query_content"]], how="left")
expanded_test_df_step2 = expanded_test_df_step2.merge(en_article_df[["article_id", "article_content"]], how="left")


# %%
expanded_test_df_step2

# %%
test_step2_scores = get_scores(model, tokenizer, expanded_test_df_step2, 16, RANKLLAMA_MAX_LENGTH, data_collator)

# %%
expanded_test_df_step2["step2_score"] = test_step2_scores
expanded_test_df_step2 = expanded_test_df_step2.groupby("query_id")[expanded_test_df_step2.columns.tolist()]\
                             .apply(get_top_preds)\
                             .reset_index(drop=True)

expanded_test_df_step2 = expanded_test_df_step2[expanded_test_df_step2["keep"] == True]

# %%
expanded_test_df_step2.to_json(f"{CHECKPOINT_DIR}/inference/{SELECTED_ID}_step2.jsonl", lines=True, orient="records")

# %%
expanded_test_df_step2

# %% [markdown]
# ## Step 3: LLM Inference

# %%
# refer to run.sh to get the predicted logits

# %%
expanded_test_df_step2 = pd.read_json(f"{CHECKPOINT_DIR}/inference/{SELECTED_ID}_step2.jsonl", lines=True, orient="records")

# %%


# %%
def load_logits(file_paths):
    logits = []
    for file_path in file_paths:
        preds = np.load(file_path)
        preds = softmax(preds, axis=1)

        logits.append(preds[:, 1])

    return np.array(logits).T


all_logit_path = [INFERENCE_DIR + f"/{SELECTED_ID}eval/" + model + f"/{SELECTED_ID}_step2_logits.npy" for model in ACCEPTED_MODELS]
all_logit_path.sort()

all_logits = load_logits(all_logit_path)

# %% [markdown]
# ## Step 4: Ensemble

# %%
# TODO: review the top filter
def top_filter(group_df):
    group_df = group_df.sort_values(by=["step3_score"],
                                          ascending=False,
                                          ignore_index=True)
    return group_df[:2]


def fill_none_predicted(row, step3_top2_df):
    if type(row["article_id"]) == list:
        return row
    row["article_id"] = step3_top2_df[step3_top2_df["query_id"] == row["query_id"]]["article_id"].values[0]

    return row

# %%


# %%
preds = (np.dot(all_logits, WEIGHTS) > CUT_OFF_THRESHOLD).astype(int)

test_df_step3 = expanded_test_df_step2.copy(deep=True)
test_df_step3["keep"] = preds & (test_df_step3["step2_score"] > RANKLLAMA_THRESHOLD)
test_df_step3 = test_df_step3[test_df_step3["keep"] == True]

# %%
test_df_step3

# %%
step3_score_df = expanded_test_df_step2.copy(deep=True)
step3_score_df["step3_score"] = np.dot(all_logits, WEIGHTS)

step3_top2_df = step3_score_df.drop_duplicates(subset=["query_id", "article_id"])\
    .groupby("query_id")[step3_score_df.columns]\
    .apply(top_filter)\
    .reset_index(drop=True)
step3_top2_df = step3_top2_df.groupby("query_id")["article_id"].apply(list).reset_index()


submission_df = test_df_step3.copy(deep=True)
submission_df = submission_df.groupby("query_id")["article_id"].apply(list).reset_index()
submission_df = submission_df.merge(test_query_df, on="query_id", how="right")


# In some cases, we can't find any predicted articles. We need to fill them with the top 2 articles from step 3
submission_df = submission_df.apply(lambda x: fill_none_predicted(x, step3_top2_df), axis=1)

# %%
submission_df.to_json(f"./{CHECKPOINT_DIR}/final/{SELECTED_ID}_submission.jsonl", lines=True, orient="records")

# %%
submission_df

# %%
import json
import numpy as np
import pandas as pd

# -----------------------
# Config: choose max k to evaluate precision@k / recall@k
# If None, will use the maximal prediction length in your dataframe.
MAX_K = None   # or set e.g. 5 or 10

# -----------------------
# Helpers
def precision_at_k_single(preds, gold_set, k):
    """Standard precision@k: |relevant in top-k| / k.
       If len(preds) < k, missing slots are treated as non-relevant (denominator still k).
    """
    if k <= 0:
        return 0.0
    topk = preds[:k]
    hits = sum(1 for p in topk if p in gold_set)
    return hits / k

def recall_at_k_single(preds, gold_set, k):
    """Recall@k: |relevant in top-k| / |gold_set|. If gold_set empty -> 0.0."""
    if len(gold_set) == 0:
        return 0.0
    topk = preds[:k]
    hits = sum(1 for p in topk if p in gold_set)
    return hits / len(gold_set)

def average_precision(preds, gold_set):
    """AP: sum_{i:pred_i in gold} (precision@i) / |gold_set| ; returns 0 if gold_set empty."""
    if len(gold_set) == 0:
        return 0.0
    hits = 0
    sum_precisions = 0.0
    for i, p in enumerate(preds, start=1):
        if p in gold_set:
            hits += 1
            sum_precisions += hits / i
    if hits == 0:
        return 0.0
    return sum_precisions / len(gold_set)

def f_beta(prec, rec, beta=2.0):
    if prec == 0 and rec == 0:
        return 0.0
    b2 = beta * beta
    return (1 + b2) * (prec * rec) / (b2 * prec + rec)

# -----------------------
# Load gold
with open(f"./kg/data_parsed/{SELECTED_ID}_data.json", "r", encoding="utf-8") as f:
    gold = json.load(f)

# normalize gold ids to strings
for qid, info in gold.items():
    gold[qid]["retrieved_list"] = [str(x).strip() for x in info.get("retrieved_list", [])]

# -----------------------
# Prepare preds df (uses in-memory step3_top2_df)
preds_df = submission_df.copy()

# Ensure preds are lists of strings
def ensure_list_of_str(x):
    if isinstance(x, list):
        return [str(v) for v in x]
    if isinstance(x, str):
        s = x.strip()
        if s.startswith("[") and s.endswith("]"):
            # crude parse: split on commas
            items = [it.strip().strip("'\"") for it in s[1:-1].split(",") if it.strip() != ""]
            return [str(it) for it in items]
        return [s]
    # fallback
    return [str(x)]

preds_df["preds"] = preds_df["article_id"].apply(ensure_list_of_str)

# Determine K for precision@k / recall@k
if MAX_K is None:
    max_pred_len = preds_df["preds"].apply(len).max()
    # but also consider gold length if you prefer; here we pick max prediction length
    K = int(max(1, max_pred_len))
else:
    K = int(MAX_K)

# -----------------------
# Evaluate per query
rows = []
# matrices for precision@k & recall@k
prec_at_k_matrix = []  # list of lists per query
rec_at_k_matrix = []

for _, r in preds_df.iterrows():
    qid = r["query_id"]
    preds = r["preds"]
    gold_list = gold.get(qid, {}).get("retrieved_list", [])
    gold_set = set(gold_list)

    # overall precision using full predicted list (len(preds) as denom)
    full_prec = (sum(1 for p in preds if p in gold_set) / len(preds)) if len(preds) > 0 else 0.0
    full_rec = (sum(1 for p in preds if p in gold_set) / len(gold_set)) if len(gold_set) > 0 else 0.0

    # F2 using full_prec, full_rec
    f2 = f_beta(full_prec, full_rec, beta=2.0)

    # AP
    ap = average_precision(preds, gold_set)

    # precision@k and recall@k for k=1..K
    prec_k = [precision_at_k_single(preds, gold_set, k) for k in range(1, K+1)]
    rec_k = [recall_at_k_single(preds, gold_set, k) for k in range(1, K+1)]

    prec_at_k_matrix.append(prec_k)
    rec_at_k_matrix.append(rec_k)

    rows.append({
        "query_id": qid,
        "n_pred": len(preds),
        "n_gold": len(gold_list),
        "precision_full": full_prec,
        "recall_full": full_rec,
        "f2_full": f2,
        "AP": ap
    })

df_metrics = pd.DataFrame(rows)

# -----------------------
# Aggregate summary
mean_precision_full = df_metrics["precision_full"].mean()
mean_recall_full = df_metrics["recall_full"].mean()
mean_f2_full = df_metrics["f2_full"].mean()

map_all = df_metrics["AP"].mean()
map_relevant = df_metrics.loc[df_metrics["n_gold"]>0, "AP"].mean()

# Precision@k and Recall@k averaged across queries (treating missing preds as non-relevant)
prec_at_k_arr = np.mean(np.array(prec_at_k_matrix), axis=0) if len(prec_at_k_matrix)>0 else np.zeros(K)
rec_at_k_arr = np.mean(np.array(rec_at_k_matrix), axis=0) if len(rec_at_k_matrix)>0 else np.zeros(K)

# -----------------------
# Print summary
print("=== Aggregate retrieval metrics ===")
print(f"Queries evaluated : {len(df_metrics)}")
print(f"Mean Precision (full predicted lists) : {mean_precision_full:.4f}")
print(f"Mean Recall (full predicted lists)    : {mean_recall_full:.4f}")
print(f"Mean F2 (β=2)                         : {mean_f2_full:.4f}")
print(f"MAP (all queries; AP=0 for empty)    : {map_all:.4f}")
print(f"MAP (only queries with >=1 gold)     : {map_relevant:.4f}")
print()
print("Precision@k (k=1..{}):".format(K))
for k, val in enumerate(prec_at_k_arr, start=1):
    print(f"  P@{k}: {val:.4f}")
print()
print("Recall@k (k=1..{}):".format(K))
for k, val in enumerate(rec_at_k_arr, start=1):
    print(f"  R@{k}: {val:.4f}")

# -----------------------
# Expose results for further inspection
# df_metrics contains per-query metrics
# prec_at_k_arr and rec_at_k_arr contain averaged P@k / R@k across queries
# You can examine per-query AP distribution:
df_metrics_sorted = df_metrics.sort_values("AP", ascending=False).reset_index(drop=True)

# show top/bottom problematic queries
print("\nTop 5 queries by AP:")
display(df_metrics_sorted.head(5))
print("\nBottom 5 queries by AP (including zero AP):")
display(df_metrics_sorted.tail(5))


# %%
df_metrics.head(100)


