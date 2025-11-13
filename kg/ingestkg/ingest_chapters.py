import json
from db_connection import Neo4jConnection

def ingest_chapters(conn, json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        chapters = json.load(f)

    for chapter_id, chapter in chapters.items():
        # Create Chapter node
        query_node = """
        MERGE (c:Chapter {id: $id})
        SET c.number = $number,
            c.description = $description,
            c.part = $part
        """
        conn.execute_query(query_node, {
            "id": chapter_id,
            "number": chapter["number"],
            "description": chapter["description"],
            "part": chapter["part"]
        })

        # Create relationship to Part
        query_rel = """
        MATCH (p:Part {id: $part_id})
        MATCH (c:Chapter {id: $chapter_id})
        MERGE (p)-[:HAS_CHAPTER]->(c)
        MERGE (c)-[:PART_OF]->(p)
        """
        conn.execute_query(query_rel, {
            "part_id": chapter["part"],
            "chapter_id": chapter_id
        })

if __name__ == "__main__":
    conn = Neo4jConnection("bolt://localhost:7687", "neo4j", "pass")
    ingest_chapters(conn, "../data/chapter_dict.json")
    conn.close()
