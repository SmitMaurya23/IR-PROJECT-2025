from sentence_transformers import SentenceTransformer
import json

model = SentenceTransformer("BAAI/bge-m3")
with open("data/article_dict.json", "r", encoding="utf-8") as f:
    articles = json.load(f)

vectors = {}
for article_id, article in articles.items():
    text = article["full_text"]
    embedding = model.encode(text, normalize_embeddings=True).tolist()
    vectors[article_id] = embedding


with open("../data/article_embedded.json", "w", encoding="utf-8") as f:
    json.dump(vectors, f, indent=2)


