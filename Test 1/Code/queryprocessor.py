from pathlib import Path
from typing import List, Dict, Any
from langchain_community.embeddings import HuggingFaceEmbeddings
import json
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import wordnet, stopwords
import nltk

#Cache the embeddings model to avoid redundant loading and improve performance
_embeddings_cache = {}

def get_embeddings_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> HuggingFaceEmbeddings:
    """ Get a cached HuggingFaceEmbeddings model or create a new one if it doesn't exist in the cache.

    Args:
        model_name (str, optional): Name of the HuggingFace model to use for embeddings. Defaults to "sentence-transformers/all-MiniLM-L6-v2".

    Returns:
        HuggingFaceEmbeddings: The cached or newly created HuggingFaceEmbeddings model.
    """
    if model_name not in _embeddings_cache:
        _embeddings_cache[model_name] = HuggingFaceEmbeddings(model_name=model_name)
    return _embeddings_cache[model_name]

def embed_query(query: str, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2") -> List[float]:
    """ Process the input query, perform query expansion using WordNet synonyms, and generate an embedding vector for the processed query using the specified HuggingFace model.

    Args:
        query (str): The input query string to be processed and embedded.
        embedding_model (str, optional): Name of the HuggingFace model to use for embeddings. Defaults to "sentence-transformers/all-MiniLM-L6-v2".

    Returns:
        List[float]: The embedding vector for the processed query.
    """
    #Process the query before embedding
    processed_query = query.strip().lower()
    processed_query = " ".join(processed_query.split())

    #Query expansion using WordNet synonyms
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    
    stop_words = set(stopwords.words('english'))
    words = processed_query.split()
    expanded_terms = set()
    for word in words:
        if word in stop_words:
            continue
        synsets = wordnet.synsets(word)
        for synset in synsets[:3]:
            for lemma in synset.lemmas()[:2]:
                synonym = lemma.name().replace('_', ' ')
                if synonym != word:
                    expanded_terms.add(synonym)
    expanded_terms = list(expanded_terms)[:10]
    if expanded_terms:
        processed_query = processed_query + " " + " ".join(expanded_terms)
    
    print(f"Processed Query: {processed_query}")

    #Embedding generation
    embeddings = get_embeddings_model(embedding_model)
    vector = embeddings.embed_documents([processed_query])[0]
    
    return vector

def load_embeddings() -> List[Dict[str, Any]]:
    """ Load and return the list of embedding records from the embeddings.json file. If the file does not exist, return an empty list.

    Returns:
        List[Dict[str, Any]]: The list of embedding records loaded from the file, or an empty list if the file does not exist.
    """
    embeddings_path = Path(__file__).parent / "embeddings.json"
    if embeddings_path.exists():
        with open(embeddings_path, 'r') as f:
            return json.load(f)
    return []

def vector_similarity(query_vector: List[float], embeddings: List[Dict[str, Any]], top_k: int = 3) -> List[Dict[str, Any]]:
    """ Calculate the cosine similarity between the query vector and each document vector in the embeddings list, and return the top k results sorted by similarity.

    Args:
        query_vector (List[float]): The vector representation of the query.
        embeddings (List[Dict[str, Any]]): The list of embedding records to compare against.
        top_k (int, optional): The number of top results to return. Defaults to 3.

    Returns:
        List[Dict[str, Any]]: The top k embedding records sorted by similarity to the query vector.
    """
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

