import os
import json
import chromadb
from chromadb.config import Settings
from langchain_neo4j import Neo4jGraph
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate

from .multiquery import generate_multi_query
from .search import vector_search, keyword_search, graph_search
from .filter_and_rerank import reciprocal_reranking, filter_reranked_docs


def init_chromadb(collection_name="legal_judgements_minilm", chromadb_path="main/data/Vector_Store/chromadb_112"):
    """
    Initializes ChromaDB with persistence settings.

    Parameters:
    - collection_name: Name of the collection to initialize in ChromaDB.

    Output:
    - Returns an initialized collection from ChromaDB.
    """
    
    client = chromadb.Client(Settings(is_persistent=True, persist_directory=chromadb_path))
    collection = client.get_or_create_collection(collection_name, metadata={"hnsw:space": "cosine"})
    print(f"ChromaDB initialized and collection '{collection_name}' ready")
    return collection

def init_neo4j(uri="bolt://localhost:7687", username="neo4j", password="password"):
    """
    Initialize and return a Neo4jGraph connection with the provided credentials.
    
    Args:
        uri (str): The Neo4j database URI
        username (str): Neo4j username
        password (str): Neo4j password
        
    Returns:
        Neo4jGraph: A connected Neo4j graph instance
    """ 
    graph = Neo4jGraph(
        url=uri,
        username=username,
        password=password
    )
    print("Successfully connected to Neo4j database")
    return graph 

def init_ollama(model_name, temperature=0):
    """
    Initializes the Ollama LLM model.

    Parameters:
    - model_name: Name of the LLM model to initialize.

    Output:
    - Returns an initialized OllamaLLM instance.
    """
     
    ollama_model = OllamaLLM(model=model_name, temperature=temperature)
    print(f"Ollama model '{model_name}' initialized successfully")
    return ollama_model 

def retrieve_documents(user_query, collection, bm25_retriever_file, chunked_docs_file, graph, graph_schema, llm_model=None, top_n=10, k=60, top_k=None, top_p=None, threshold=None):
    """
    Generates responses for filtered and reranked parent documents using an LLM.

    Parameters:
    - user_query (str): The query to retrieve documents.
    - model_name (str): The name of the LLM model to generate responses.
    - prompt_template (str): Template for prompt generation.
    - k (int): Number of query variations to generate.
    - top_n (int): Number of top results to retrieve from each search.
    - top_percent (float): Percentage of top documents to keep (0 < top_percent <= 100).
    - threshold (float): Minimum probability to include a document (0 <= threshold <= 1).

    Returns:
    - dict: A dictionary mapping parent documents to their LLM-generated responses.
    """
    
    queries = generate_multi_query(user_query)  

    vector_results, keyword_results, graph_results = [], [], []
    for i, query in enumerate(queries): 
        vector_result = vector_search(query, collection, top_n) 
        vector_results.append(vector_result) 
        
        keyword_result = keyword_search(query, bm25_retriever_file, chunked_docs_file, top_n) 
        keyword_results.append(keyword_result)

        graph_result = graph_search(query, graph, graph_schema, top_n=top_n)  
        graph_results.append(graph_result) 

 
    reranked_results = reciprocal_reranking(vector_results, keyword_results, graph_results, k)  
    filtered_docs = filter_reranked_docs(reranked_results, top_k=top_k, top_p=top_p, threshold=threshold) 
     
    return filtered_docs 