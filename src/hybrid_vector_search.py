import pandas as pd
import redis
import hashlib
import json
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os

# Constants
PRODUCTS_FILE = 'combined_products.parquet'
FIELDS_TO_COMBINE = [
    'category_left', 'brand_left', 'title_left', 'description_left', 'specTableContent_left',
    'brand_right', 'title_right', 'description_right', 'specTableContent_right'
]

# Redis Cache
cache = redis.StrictRedis(host='localhost', port=6379, db=0, decode_responses=True)

# Load Data
print("📦 Loading products...")
df = pd.read_parquet(PRODUCTS_FILE)
df['combined'] = df[FIELDS_TO_COMBINE].apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1)

# Embedding Model
print("🔠 Loading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Vector Embedding + FAISS Index Setup
index_file = "faiss.index"
embedding_file = "embeddings.npy"
if os.path.exists(index_file) and os.path.exists(embedding_file):
    print("📂 Loading existing FAISS index...")
    index = faiss.read_index(index_file)
    embeddings = np.load(embedding_file)
else:
    print("📈 Generating embeddings...")
    embeddings = model.encode(df['combined'].tolist(), show_progress_bar=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    faiss.write_index(index, index_file)
    np.save(embedding_file, embeddings)

# Cache Helpers
def hash_query(query):
    return hashlib.md5(query.encode()).hexdigest()

# Search

def search_products(query, limit=10):
    key = hash_query("hybrid:" + query)
    cached_result = cache.get(key)
    if cached_result:
        print("✅ Returning result from cache...")
        return json.loads(cached_result)

    print("🔍 Performing hybrid search...")

    # Keyword Search
    keyword_matches = df[df['combined'].str.contains(query, case=False, na=False)]

    # Vector Search
    query_embedding = model.encode([query])
    D, I = index.search(query_embedding, limit)
    vector_matches = df.iloc[I[0]]

    # Combine results (by index) and deduplicate
    hybrid = pd.concat([keyword_matches, vector_matches]).drop_duplicates().head(limit)

    results = []
    for _, row in hybrid.iterrows():
        product = {
            "brand_left": row.get('brand_left', ''),
            "title_left": row.get('title_left', ''),
            "description_left": row.get('description_left', ''),
            "brand_right": row.get('brand_right', ''),
            "title_right": row.get('title_right', ''),
            "description_right": row.get('description_right', ''),
        }
        results.append(product)

    cache.set(key, json.dumps(results))
    return results

# Main loop
while True:
    query = input("\n🔎 Search query (or type 'exit'): ")
    if query.lower() == 'exit':
        break

    results = search_products(query)

    if not results:
        print("❌ No products found for your search.")
    else:
        total_results = len(results)
        print(f"\n🎯 Showing {min(10, total_results)} of {total_results} product(s):\n")

        for idx, product in enumerate(results[:10], start=1):
            print(f"📦 Product {idx}:")
            print(f"  🔹 Brand (Left): {product['brand_left']}")
            print(f"  🔹 Title (Left): {product['title_left']}")
            print(f"  🔹 Brand (Right): {product['brand_right']}")
            print(f"  🔹 Title (Right): {product['title_right']}\n")
            print(f"  🔹 Title (Right): {product['description_right']}\n")
