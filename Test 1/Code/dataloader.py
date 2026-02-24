"""
Document Loader for RAG System
Loads stored documents using LangChain
"""

from pathlib import Path
from typing import List

from langchain_community.document_loaders import DirectoryLoader, TextLoader, PDFPlumberLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
import json



def get_docs_path() -> Path:
    """Get the absolute path to the Document Corpus directory.

    Returns:
        Path: Absolute path to the Document Corpus directory
    """
    current_dir = Path(__file__).parent
    corpus_path = current_dir.parent / "Document Corpus"
    return corpus_path


def load_text_documents() -> List[Document]:
    """Load all text documents (.txt) from the Document Corpus directory.

    Raises:
        FileNotFoundError: If the Document Corpus directory does not exist

    Returns:
        List[Document]: List of loaded text documents
    """
    corpus_path = get_docs_path()
    
    if not corpus_path.exists():
        raise FileNotFoundError(f"Documents not found at: {corpus_path}")
    
    loader = DirectoryLoader(
        str(corpus_path),
        glob="**/*.txt",
        loader_cls=TextLoader,
        silent_errors=True
    )
    
    documents = loader.load()
    print(f"Loaded {len(documents)} text documents from {corpus_path}")
    return documents

def load_pdf_documents() -> List[Document]:
    """Load all PDF documents (.pdf) from the Document Corpus directory.

    Raises:
        FileNotFoundError: If the Document Corpus directory does not exist

    Returns:
        List[Document]: List of loaded PDF documents
    """
    corpus_path = get_docs_path()
    
    if not corpus_path.exists():
        raise FileNotFoundError(f"Documents not found at: {corpus_path}")
    
    loader = DirectoryLoader(
        str(corpus_path),
        glob="**/*.pdf",
        loader_cls=PDFPlumberLoader,
        silent_errors=True
    )
    
    documents = loader.load()
    print(f"Loaded {len(documents)} PDF documents from {corpus_path}")
    return documents


def load_all_documents() -> List[Document]:
    """Load all documents (text and PDF) from the Document Corpus directory.

    Returns:
        List[Document]: List of all loaded documents
    """
    all_documents = []
    
    try:
        all_documents.extend(load_text_documents())
    except Exception as e:
        print(f"Error loading text documents: {e}")
    
    try:
        all_documents.extend(load_pdf_documents())
    except Exception as e:
        print(f"Error loading PDF documents: {e}")
    
    print(f"\nTotal documents loaded: {len(all_documents)}")
    return all_documents

def split_documents(documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """Split documents into smaller chunks using RecursiveCharacterTextSplitter.

    Args:
        documents (List[Document]): List of documents to split
        chunk_size (int, optional): Maximum size of each chunk in characters. Defaults to 1000.
        chunk_overlap (int, optional): Number of characters to overlap between chunks. Defaults to 200.

    Raises:
        ValueError: If chunk_overlap is greater than or equal to chunk_size

    Returns:
        List[Document]: List of split document chunks
    """
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be less than chunk_size")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    
    split_docs = text_splitter.split_documents(documents)
    print(f"\nDocuments split into {len(split_docs)} chunks")
    
    return split_docs

def generate_embeddings(documents: List[Document], embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """Generate embeddings for the given documents using HuggingFaceEmbeddings and save them to a JSON file.

    Args:
        documents (List[Document]): List of documents to generate embeddings for
        embedding_model (str, optional): Name of the HuggingFace model to use for embeddings. Defaults to "sentence-transformers/all-MiniLM-L6-v2".

    Returns:
        _type_: List of embeddings corresponding to the input documents
    """
    embeddings_path = Path(__file__).parent / "embeddings.json"
    
    #Load old embeddings
    existing_embeddings = {}
    if embeddings_path.exists():
        with open(embeddings_path, 'r') as f:
            existing_embeddings = json.load(f)
    
    #Embed new documents
    new_docs = [doc for doc in documents if doc.page_content not in existing_embeddings]
    if not new_docs:
        print("No new documents to embed")
        return [existing_embeddings[doc.page_content] for doc in documents]
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    new_embeddings = embeddings.embed_documents([doc.page_content for doc in new_docs])
    
    #Save new embeddings
    for doc, embedding in zip(new_docs, new_embeddings):
        existing_embeddings[doc.page_content] = embedding
    with open(embeddings_path, 'w') as f:
        json.dump(existing_embeddings, f)
    
    print(f"\nGenerated embeddings for {len(new_embeddings)} new documents")
    return [existing_embeddings[doc.page_content] for doc in documents]

    #Probably worth adding functionality to remove/update old embeddings for documents that have been deleted/modified, but I'll do that later.



if __name__ == "__main__":
    # Load and split documents, then generate embeddings
    all_docs = load_all_documents()
    split_docs = split_documents(all_docs)
    generate_embeddings(split_docs)