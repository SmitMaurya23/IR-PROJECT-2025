from sklearn.metrics import fbeta_score
import numpy as np
import json
from neo4j import GraphDatabase

driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "pass"))

k=50

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def average_precision_at_k(predicted, relevant, k):
    score = 0.0
    hits = 0
    for i, p in enumerate(predicted[:k]):
        if p in relevant:
            hits += 1
            score += hits / (i + 1)
    return score / min(len(relevant), k)

# Load test queries
with open("../data/test_embedded.json", "r", encoding="utf-8") as f:
    test_queries = json.load(f)

# Load article vectors from Neo4j
with driver.session() as session:
    result = session.run("""
        MATCH (a:Article)
        WHERE a.vector IS NOT NULL
        RETURN a.id AS id, a.vector AS vector
    """)
    articles = [(record["id"], np.array(record["vector"])) for record in result]

# Evaluation
recall_scores, map_scores, f2_scores = [], [], []

for qid, entry in test_queries.items():
    qvec = np.array(entry["vector"])
    relevant = set(entry["retrieved_list"])

    # Rank articles by similarity
    ranked = sorted(
        [(aid, cosine_similarity(qvec, vec)) for aid, vec in articles],
        key=lambda x: x[1],
        reverse=True
    )
    top_k = [aid for aid, _ in ranked[:k]]

    # Recall@5
    retrieved_relevant = [aid for aid in top_k if aid in relevant]
    recall = len(retrieved_relevant) / len(relevant)
    recall_scores.append(recall)

    # MAP@5
    map_score = average_precision_at_k(top_k, relevant, k=5)
    map_scores.append(map_score)

    # F2@5
    y_true = [1 if aid in relevant else 0 for aid in top_k]
    y_pred = [1] * len(top_k)
    f2 = fbeta_score(y_true, y_pred, beta=2)
    f2_scores.append(f2)

# Aggregate
print(f"Recall@{k}: {np.mean(recall_scores):.4f}")
print(f"MAP@{k}:    {np.mean(map_scores):.4f}")
print(f"F2@{k}:     {np.mean(f2_scores):.4f}")
