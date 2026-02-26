from pathlib import Path
from typing import List
from langchain_community.embeddings import HuggingFaceEmbeddings
import json
from sklearn.metrics.pairwise import cosine_similarity

def embed_query(query: str, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    vector = embeddings.embed_documents([query])[0]
    return vector

def vector_similarity(query_vector, top_k: int = 3):
    embeddings_path = Path(__file__).parent / "embeddings.json"
    if embeddings_path.exists():
        with open(embeddings_path, 'r') as f:
            embeddings = json.load(f)
            similarities = []
            for record in embeddings:
                doc_vector = record["vector"]
                similarity = cosine_similarity([query_vector], [doc_vector])[0][0]
                similarities.append({
                    "id": record.get("id"),
                    "similarity": similarity
                })
            top_results = sorted(similarities, key=lambda x: x["similarity"], reverse=True)[:top_k]
            for result in top_results:
                print(f"Similarity: {result['similarity']:.4f} | ID: {result['id']}")
            return top_results
        
if __name__ == "__main__":
    query = input("Enter your query: ")
    query_vector = embed_query(query)
    results = vector_similarity(query_vector)
    print("\nTop 3 Results:")
    
    embeddings_path = Path(__file__).parent / "embeddings.json"
    with open(embeddings_path, 'r') as f:
        embeddings = json.load(f)
        embeddings_dict = {record["id"]: record for record in embeddings}
    
    for result in results:
        record = embeddings_dict.get(result["id"])
        if record and record.get("text") is not None:
            print(f"ID: {record.get('id')}: {record.get('text')[:100]}")

