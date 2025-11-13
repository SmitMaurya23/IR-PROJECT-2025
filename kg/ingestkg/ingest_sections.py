import json
from db_connection import Neo4jConnection

def ingest_sections(conn, json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        sections = json.load(f)

    for section_id, section in sections.items():
        # Create Section node
        query_node = """
        MERGE (s:Section {id: $id})
        SET s.number = $number,
            s.description = $description,
            s.chapter = $chapter,
            s.part = $part
        """
        conn.execute_query(query_node, {
            "id": section_id,
            "number": section["number"],
            "description": section["description"],
            "chapter": section["chapter"],
            "part": section["part"]
        })

        # Create relationship to Chapter
        query_rel = """
        MATCH (c:Chapter {id: $chapter_id})
        MATCH (s:Section {id: $section_id})
        MERGE (c)-[:HAS_SECTION]->(s)
        MERGE (s)-[:PART_OF]->(c)
        """
        conn.execute_query(query_rel, {
            "chapter_id": section["chapter"],
            "section_id": section_id
        })

if __name__ == "__main__":
    conn = Neo4jConnection("bolt://localhost:7687", "neo4j", "pass")
    ingest_sections(conn, "../data/section_dict.json")
    conn.close()
