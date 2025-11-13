import json
from db_connection import Neo4jConnection

def ingest_parts(conn, json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        parts = json.load(f)

    for part_id, part in parts.items():
        query = """
        MERGE (p:Part {id: $id})
        SET p.number = $number,
            p.description = $description
        """
        conn.execute_query(query, {
            "id": part_id,
            "number": part["number"],
            "description": part["description"]
        })

if __name__ == "__main__":
    conn = Neo4jConnection("bolt://localhost:7687", "neo4j", "pass")
    ingest_parts(conn, "../data/part_dict.json")
    conn.close()
