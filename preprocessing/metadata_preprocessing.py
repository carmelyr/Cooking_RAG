import pandas as pd
import sqlite3
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from project import config

connection = sqlite3.connect(config.DB_PATH)
cursor = connection.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS recipes (
    recipe_id TEXT PRIMARY KEY,
    title TEXT,
    ingredients TEXT,
    directions TEXT,
    link TEXT,
    source TEXT,
    named_entities TEXT
)
""")
connection.commit()

# Load the csv file in chunks and insert into the database
for chunk in pd.read_csv(config.CSV_PATH, chunksize=10000):
    for index, row in chunk.iterrows():
        recipe_id = row["Unnamed: 0"] # In the original csv file, the first column is unnamed
        title = row["title"]
        ingredients = row["ingredients"]
        directions = row["directions"]
        link = row["link"]
        source = row["source"]
        named_entities = row["NER"]
        
        # Insert into the SQLite database
        cursor.execute("""
            INSERT OR REPLACE INTO recipes (
                recipe_id, title, ingredients, directions, link, source, named_entities
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            recipe_id,
            title,
            ingredients,
            directions,
            link,
            source,
            named_entities
        ))
    connection.commit()

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2') # Bert-based model for sentence embeddings
title_vectors = []
directions_vectors = []
recipe_ids = []

# Compute sentence embeddings for both the recipe title and the instructions
rows = cursor.execute("SELECT recipe_id, title, directions FROM recipes").fetchall()
recipe_ids = [row[0] for row in rows]
titles = [row[1] for row in rows]
directions = [row[2] for row in rows]

title_vectors = model.encode(
    titles,
    batch_size=config.TITLE_BATCH_SIZE,
    show_progress_bar=True,
    convert_to_numpy=True
).astype("float32")

directions_vectors = model.encode(
    directions,
    batch_size=config.DIRECTIONS_BATCH_SIZE,
    show_progress_bar=True,
    convert_to_numpy=True
).astype("float32")

# Normalize the vectors to length 1
faiss.normalize_L2(title_vectors)
faiss.normalize_L2(directions_vectors)

# Create a FAISS index
print("Creating FAISS index...")

# Flat = exact search
# IP = inner product
title_index = faiss.IndexFlatIP(model.get_sentence_embedding_dimension())
directions_index = faiss.IndexFlatIP(model.get_sentence_embedding_dimension())

# Add vectors to the FAISS index
title_index.add(title_vectors)
directions_index.add(directions_vectors)

print("Saving title index ...")
faiss.write_index(title_index, config.FAISS_TITLE_INDEX_PATH)
print("Saving directions index...")
faiss.write_index(directions_index, config.FAISS_DIRECTIONS_INDEX_PATH)
print("Saving recipe IDs ...")
np.save(config.RECIPE_IDS_PATH, np.array(recipe_ids))

#----- Test the FAISS index -----
query_title = "Chocolate Cake"
query_title_embedding = model.encode(query_title, batch_size=config.TITLE_BATCH_SIZE, convert_to_numpy=True).astype("float32")
query_title_embedding = query_title_embedding.reshape(1, -1) # Shape: (1, 384)
faiss.normalize_L2(query_title_embedding)

distances, indices = title_index.search(query_title_embedding, config.TOP_K_TITLE)

print("Top recipes by title:")
for idx in indices[0]:
    print(recipe_ids[idx])

query_directions = "Mix flour, sugar, cocoa powder, baking powder, and salt. Add eggs, milk, oil, and vanilla extract. Bake at 350 degrees for 30 minutes."
query_directions_embedding = model.encode(query_directions, convert_to_numpy=True).astype("float32")
query_directions_embedding = query_directions_embedding.reshape(1, -1) # Shape: (1, 384)
faiss.normalize_L2(query_directions_embedding) 

distances, indices = directions_index.search(query_directions_embedding, config.TOP_K_DIRECTIONS)
print("Top recipes by directions:")
for idx in indices[0]:
    print(recipe_ids[idx])

connection.close()