import os  

def reciprocal_reranking(vector_results=None, keyword_results=None, graph_results=None, k=60):
    """
    Reranks results from vector and keyword search using Reciprocal Rank Fusion (RRF) 
    and merges fused scores by parent document.

    Parameters:
    - vector_results: List of results from vector-based search.
    - keyword_results: List of results from keyword-based search.
    - k: Hyperparameter for RRF to adjust the influence of rank (default is 60).

    Output:
    - Returns a reranked list of parent documents with their combined scores and associated chunks.
    """

    fused_scores = {}
    parent_scores = {}
    parent_documents = {}
 
    for result_list in vector_results:   
        for result in result_list:   
            doc_id = result["id"]
            parent = result["parent"]
            if parent not in parent_documents:
                parent_documents[parent] = []
            parent_documents[parent].append(result["document"])

            fused_scores[doc_id] = fused_scores.get(doc_id, 0) + 1 / (result["rank"] + k)

 
    for result_list in keyword_results:   
        for result in result_list:   
            doc_id = result["id"]
            parent = result["parent"]
            if parent not in parent_documents:
                parent_documents[parent] = []
            parent_documents[parent].append(result["document"])

            fused_scores[doc_id] = fused_scores.get(doc_id, 0) + 1 / (result["rank"] + k)


    for result_list in graph_results:
        for result in result_list:    
            parent = result["j"]["name"] + ".txt"
            
            if parent not in parent_documents:
                parent_documents[parent] = []
            summary = f"Summary: {result["j"]["summary"]}"
            problem = f"Problem: {result["j"]["problem"]}"
            parent_documents[parent].append([summary, problem])

            fused_scores[parent] = fused_scores.get(parent, 0) + (1/(result["rank"] + k))


    for doc_id, score in fused_scores.items():
        parent = doc_id.split("_chunk_")[0]  
        parent_scores[parent] = parent_scores.get(parent, 0) + score
    
    reranked_results = sorted(
        [
            {
                "parent": parent,
                "score": score,
                "documents": parent_documents[parent]
            }
            for parent, score in parent_scores.items()
        ],
        key=lambda x: x["score"], reverse=True
    ) 

    return reranked_results

def filter_reranked_docs(reranked_results, top_k=None, top_p=None, threshold=None):
    """
    Filters reranked documents based on a specified top percentage or a probability threshold.

    Parameters:
    - reranked_results: List of reranked documents with scores.
    - top_percent: Percentage of top documents to retain (optional, 0 < top_percent <= 100).
    - threshold: Minimum probability threshold for including documents (optional, 0 <= threshold <= 1).

    Output:
    - Returns a filtered list of documents meeting the specified criteria.
    """

    if not reranked_results:
        return []
 
    reranked_results = sorted(reranked_results, key=lambda x: x["score"], reverse=True)
 
    scores = [doc["score"] for doc in reranked_results] 
    sum_scores = sum(scores)
    probabilities = [score / sum_scores for score in scores]
      
    for doc, prob in zip(reranked_results, probabilities):
        doc["probability"] = prob

    filtered_docs = reranked_results

    if top_k:
        filtered_docs = reranked_results[:top_k] 

    if top_p:
        cutoff_index = int(len(reranked_results) * (top_p / 100))
        filtered_docs = reranked_results[:cutoff_index+1]

    if threshold:
        filtered_docs = [doc for doc in filtered_docs if doc["probability"] >= threshold]
 
    if filtered_docs:
        last_prob = filtered_docs[-1]["probability"]
        filtered_docs.extend(
            doc for doc in reranked_results[len(filtered_docs):] if doc["probability"] == last_prob
        )

    if filtered_docs == []:
        filtered_docs = [reranked_results[0]]

    return filtered_docs

def retrieve_parent_document(txt_dir, parent_doc_id):
    """
    Retrieves the content of a parent document from a specified directory.

    Parameters:
    - parent_doc_id: The name of the parent document to retrieve.

    Output:
    - Returns the content of the document as a string if found.
    - Returns None if the document is not found.

    Raises:
    - FileNotFoundError: If the file is not found in the specified directory.
    """
    
    file_path = os.path.join(txt_dir, parent_doc_id)
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        print(f"The file {parent_doc_id} was not found in the directory {txt_dir}.")
        return None