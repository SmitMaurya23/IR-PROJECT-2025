from sentence_transformers import SentenceTransformer
import json
import numpy as np

model = SentenceTransformer("BAAI/bge-m3")

def embed_queries(json_path, output_path):
    with open(json_path, "r", encoding="utf-8") as f:
        queries = json.load(f)

    embedded = {}
    for qid, entry in queries.items():
        text = entry["query"]
        vector = model.encode(text, normalize_embeddings=True).tolist()
        embedded[qid] = {
            "query": text,
            "vector": vector,
            "retrieved_list": entry["retrieved_list"]
        }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(embedded, f, indent=2)

embed_queries("../data/train_data.json", "../data/train_embedded.json")
embed_queries("../data/test_data.json", "../data/test_embedded.json")
