import json
from db_connection import Neo4jConnection
from collections import defaultdict

def ingest_articles(conn, json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        articles = json.load(f)

    # Step 1: Create Article nodes and PART_OF relationships
    for article_id, article in articles.items():
        query_node = """
        MERGE (a:Article {id: $id})
        SET a.number = $number,
            a.description = $description,
            a.content = $content,
            a.full_text = $full_text,
            a.chapter = $chapter,
            a.section = $section,
            a.part = $part,
            a.ontologies = $ontologies
        """
        conn.execute_query(query_node, {
            "id": article_id,
            "number": article["number"],
            "description": article["description"],
            "content": article["content"],
            "full_text": article["full_text"],
            "chapter": article["chapter"],
            "section": article["section"],
            "part": article["part"],
            "ontologies": article.get("ontologies", [])
        })

        # Link to Section or Chapter
        if article["section"]:
            query_rel = """
            MATCH (s:Section {id: $section_id})
            MATCH (a:Article {id: $article_id})
            MERGE (s)-[:HAS_ARTICLE]->(a)
            MERGE (a)-[:PART_OF]->(s)
            """
            conn.execute_query(query_rel, {
                "section_id": article["section"],
                "article_id": article_id
            })
        else:
            query_rel = """
            MATCH (c:Chapter {id: $chapter_id})
            MATCH (a:Article {id: $article_id})
            MERGE (c)-[:HAS_ARTICLE]->(a)
            MERGE (a)-[:PART_OF]->(c)
            """
            conn.execute_query(query_rel, {
                "chapter_id": article["chapter"],
                "article_id": article_id
            })

        # Ontology references
        for target_id in article.get("ontologies", []):
            query_ref = """
            MATCH (a:Article {id: $source_id})
            MATCH (b:Article {id: $target_id})
            MERGE (a)-[:REFERENCES]->(b)
            MERGE (b)-[:REFERENCED_BY]->(a)
            """
            conn.execute_query(query_ref, {
                "source_id": article_id,
                "target_id": target_id
            })

    # Step 2: Build SAME_* relationships
    section_map = defaultdict(list)
    chapter_map = defaultdict(list)
    part_map = defaultdict(list)

    for aid, a in articles.items():
        if a["section"]:
            section_map[a["section"]].append(aid)
        chapter_map[a["chapter"]].append(aid)
        part_map[a["part"]].append(aid)

    def create_pairwise_relations(article_ids, rel_type, filter_fn=None):
        for i in range(len(article_ids)):
            for j in range(i + 1, len(article_ids)):
                a1, a2 = article_ids[i], article_ids[j]
                if filter_fn and not filter_fn(a1, a2):
                    continue
                query = f"""
                MATCH (a1:Article {{id: $a1}})
                MATCH (a2:Article {{id: $a2}})
                MERGE (a1)-[:{rel_type}]->(a2)
                MERGE (a2)-[:{rel_type}]->(a1)
                """
                conn.execute_query(query, {"a1": a1, "a2": a2})

    # SAME_SECTION: within same section
    for section_id, article_ids in section_map.items():
        create_pairwise_relations(article_ids, "SAME_SECTION")

    # SAME_CHAPTER: different sections, same chapter
    for chapter_id, article_ids in chapter_map.items():
        def diff_section(a1, a2):
            return articles[a1]["section"] != articles[a2]["section"]
        create_pairwise_relations(article_ids, "SAME_CHAPTER", filter_fn=diff_section)

    # SAME_PART: different chapters, same part
    for part_id, article_ids in part_map.items():
        def diff_chapter(a1, a2):
            return articles[a1]["chapter"] != articles[a2]["chapter"]
        create_pairwise_relations(article_ids, "SAME_PART", filter_fn=diff_chapter)

if __name__ == "__main__":
    conn = Neo4jConnection("bolt://localhost:7687", "neo4j", "pass")
    ingest_articles(conn, "../data/article_dict.json")
    conn.close()
