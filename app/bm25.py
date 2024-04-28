import sqlite3
from rank_bm25 import BM25Okapi
import pandas as pd

def fetch_data_from_db(file_path, table_name):
    conn = sqlite3.connect(file_path)
    cursor = conn.cursor()
    cursor.execute(f"SELECT title, description FROM {table_name}")
    data = cursor.fetchall()
    conn.close()
    return data

def prepare_data_for_bm25(data):
    titles = [item[0] for item in data]
    descriptions = [item[1] for item in data]
    tokenized_titles = [title.lower().split() for title in titles]
    tokenized_descriptions = [description.lower().split() for description in descriptions]
    return tokenized_titles, tokenized_descriptions

def perform_bm25_search(query, tokenized_data):
    bm25 = BM25Okapi(tokenized_data)
    scores = bm25.get_scores(query.lower().split())
    return scores

def search_bm25(query, file_path, table_name):
    data = fetch_data_from_db(file_path, table_name)
    tokenized_titles, tokenized_descriptions = prepare_data_for_bm25(data)
    title_scores = perform_bm25_search(query, tokenized_titles)
    description_scores = perform_bm25_search(query, tokenized_descriptions)
    combined_scores = [sum(x) for x in zip(title_scores, description_scores)]
    sorted_scores = sorted(range(len(combined_scores)), key=lambda i: combined_scores[i], reverse=True)
    sorted_data = [data[i] for i in sorted_scores]
    return sorted_data
