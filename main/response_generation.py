import json
import os
from langchain.prompts import PromptTemplate  
from langchain_ollama import OllamaLLM
from .search import get_graph_schema
from .retrieval import retrieve_documents
from .filter_and_rerank import retrieve_parent_document
  
TXT_DIR = 'main/data/Judgements/Judgements_515_txt'  # Directory containing the text files of judgments
model_name = "llama3.2:3b"   
 
ANSWER_GENERATION_PROMPT = prompt = PromptTemplate(
    input_variables=["query", "context"], 
    template="""
            Given the following context related to Indian legal judgments:
            '{context}'

            ask:Please answer the question '{query}' in a few words or a short sentence. 

            Instructions:
            - Be direct and concise. Strictly answer in 2 to 3 sentences or points.
            - Provide a clear answer without unnecessary elaboration.
            - If the answer is not found in the context, respond with 'Not found'.

            Example:
            user question: What clarification did the Supreme Court provide regarding the inherent powers of courts in consent decrees?
            llm answer: The Supreme Court clarified that courts have inherent powers to rectify consent decrees to ensure they are free from clerical or arithmetical errors, but such powers are limited to obvious mistakes, fraud, or misrepresentation.
            """
    )  

def generate_llm_response(query, llm_model, parent_documents): 

    responses = {}
    for parent_doc in parent_documents:
        parent_doc_id = parent_doc["parent"]
        context = retrieve_parent_document(TXT_DIR, parent_doc_id)
        chain = ANSWER_GENERATION_PROMPT | llm_model
        response = chain.invoke({"query": query, "context": context})
        responses[parent_doc_id] = {"probability": parent_doc["probability"], "response": response}

    return responses
 
def find_judgements(query, collection, graph, llm_model):
    bm25_retriever_file = "main/data/Keyword_Store/bm25_retriever_112.pkl"
    chunked_docs_file = "main/data/Keyword_Store/chunked_documents_112.pkl"
    graph_schema = get_graph_schema(graph)
    k = 60
    top_n = 10
    top_k = 3
    top_p = None
    threshold = 0.05

    parent_documents = retrieve_documents(query, collection, 
                                          bm25_retriever_file, chunked_docs_file, 
                                          graph, graph_schema, 
                                          llm_model=None, top_n=10, k=60,  
                                          top_k=3, top_p=None, threshold=None)
    
    responses = generate_llm_response(query, llm_model, parent_documents)
    if responses:
        return responses
    else:
        return []