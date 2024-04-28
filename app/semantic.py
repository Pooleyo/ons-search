import sqlite3
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

# Load the model and tokenizer from HuggingFace
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def create_embeddings(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].numpy()

def add_embeddings_to_db(file_path, table_name):
    # Connect to the SQLite database
    conn = sqlite3.connect(file_path)
    cursor = conn.cursor()
    
    # Fetch all records from the database
    cursor.execute(f"SELECT rowid, title, description FROM {table_name}")
    data = cursor.fetchall()
    
    # Add a new column for embeddings if it doesn't exist
    cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN embedding BLOB")
    
    # Process each record to create embeddings and update the database
    for row in data:
        rowid, title, description = row
        combined_text = title + " " + description
        embedding = create_embeddings(combined_text)
        cursor.execute(f"UPDATE {table_name} SET embedding = ? WHERE rowid = ?", (embedding, rowid))
    
    # Commit changes and close the connection
    conn.commit()
    conn.close()


def fetch_sorted_entries(file_path, table_name, user_query):
    # Compute the query embedding
    query_embedding = create_embeddings(user_query)
    
    # Connect to the SQLite database
    conn = sqlite3.connect(file_path)
    cursor = conn.cursor()
    
    # Fetch all embeddings and their associated titles and descriptions
    cursor.execute(f"SELECT rowid, title, description, embedding FROM {table_name}")
    data = cursor.fetchall()
    
    # Compute cosine similarities and collect results
    results = []
    for row in data:
        rowid, title, description, stored_embedding = row
        # Convert the stored embedding from a blob to a numpy array
        stored_embedding_array = np.frombuffer(stored_embedding, dtype=np.float32).reshape(1, -1)
        
        # Compute cosine similarity
        similarity = cosine_similarity(query_embedding, stored_embedding_array)[0][0]
        
        # Collect the result
        results.append((rowid, title, description, similarity))
    
    # Sort results by similarity in descending order
    results.sort(key=lambda x: x[3], reverse=True)
    
    # Commit changes and close the connection
    conn.close()

    # Return the sorted list of titles and descriptions based on similarity
    return [(title, description, similarity) for _, title, description, similarity in results]

# Example usage
file_path = "your_database.db"
table_name = "your_table"
user_query = "Example query text"
