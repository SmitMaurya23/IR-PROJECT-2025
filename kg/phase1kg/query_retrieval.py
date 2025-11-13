from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("BAAI/bge-m3")
query = "If I transfer my property to my son. Will I be arrested?"
query_vector = model.encode(query, normalize_embeddings=True)

from neo4j import GraphDatabase

driver = GraphDatabase.driver("bolt://10.50.22.71:7687", auth=("neo4j", "pass"))

with driver.session() as session:
    result = session.run("""
        MATCH (a:Article)
        WHERE a.vector IS NOT NULL
        RETURN a.id AS id, a.vector AS vector
    """)
    articles = [(record["id"], np.array(record["vector"])) for record in result]

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

ranked = sorted(
    [(article_id, cosine_similarity(query_vector, vec)) for article_id, vec in articles],
    key=lambda x: x[1],
    reverse=True
)


top_k = 10
for article_id, score in ranked[:top_k]:
    print(f"Article {article_id} — Score: {score:.4f}")
