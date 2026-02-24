"""
Document Loader for RAG System
Loads stored documents using LangChain
"""

from pathlib import Path
from typing import List

from langchain_community.document_loaders import DirectoryLoader, TextLoader, PDFPlumberLoader
from langchain_core.documents import Document



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