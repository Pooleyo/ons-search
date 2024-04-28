from app.bm25 import search_bm25
from app.semantic import add_embeddings_to_db, fetch_sorted_entries

sqlite_file_path = "data/ons_data_list.db"
table_name = "datasets"

search_query = "economic data"

# results = search_bm25(search_query, sqlite_file_path, table_name)
# print("\n".join([f"Title: {result[0]}\nDescription: {result[1]}" for result in results[0:10]]))

# Add embeddings to the database
# add_embeddings_to_db(sqlite_file_path, table_name)

sorted_entries = fetch_sorted_entries(sqlite_file_path, table_name, search_query)
print(sorted_entries)
