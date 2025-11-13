# test_connection.py
from db_connection import Neo4jConnection

conn = Neo4jConnection("neo4j:// 10.50.22.71:7687", "neo4j", "pass")
result = conn.execute_query("RETURN 'Connected to Neo4j!' AS message")
for record in result:
    print(record["message"])
conn.close()
