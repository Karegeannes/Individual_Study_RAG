from pathlib import Path
from typing import List
from langchain_community.embeddings import HuggingFaceEmbeddings
import json
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import wordnet
import nltk

def embed_query(query: str, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
    #Process the query before embedding
    processed_query = query.strip().lower()
    processed_query = " ".join(processed_query.split())

    #Query expansion using WordNet synonyms
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')
    words = processed_query.split()
    expanded_terms = []
    for word in words:
        synsets = wordnet.synsets(word)
        for synset in synsets[:1]:
            for lemma in synset.lemmas()[:1]:
                synonym = lemma.name().replace('_', ' ')
                if synonym != word:
                    expanded_terms.append(synonym)
    if expanded_terms:
        processed_query = processed_query + " " + " ".join(expanded_terms)
    print(f"Processed Query: {processed_query}")

    #Embedding generation
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    vector = embeddings.embed_documents([processed_query])[0]
    
    return vector

def load_embeddings():
    embeddings_path = Path(__file__).parent / "embeddings.json"
    if embeddings_path.exists():
        with open(embeddings_path, 'r') as f:
            return json.load(f)
    return []

def vector_similarity(query_vector, embeddings: List, top_k: int = 3):
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
    embeddings_path = Path(__file__).parent / "embeddings.json"
    with open(embeddings_path, 'r') as f:
        embeddings = json.load(f)
        embeddings_dict = {record["id"]: record for record in embeddings}
    while True:
        try:
            query = input("Enter your query (or 'exit' to quit): ")
            if query.lower() == 'exit':
                print("Exiting...")
                break
            query_vector = embed_query(query)
            results = vector_similarity(query_vector, embeddings)
            print("\nTop 3 Results:")
            for result in results:
                record = embeddings_dict.get(result["id"])
                if record and record.get("text") is not None:
                    print(f"ID: {record.get('id')}: {record.get('text')[:100]}")
            print()
        except KeyboardInterrupt:
            print("\nExiting...")
            break

