# Import necessary libraries and modules 
import json
import re
import pickle 
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from sentence_transformers import SentenceTransformer 

 
def vector_search(query, collection, top_n=5):
    """
    Performs vector-based search to retrieve top `n` document chunks.

    Parameters:
    - query: The query string for which relevant documents are to be retrieved.
    - collection: The collection from which documents are to be searched.
    - top_n: Number of top results to retrieve (default is 5).

    Output:
    - Returns a list of dictionaries containing document details such as ID, content, parent document name, and rank.
    """ 
    encoder_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    query_embedding = encoder_model.encode(query)
    results = collection.query(query_embeddings=[query_embedding.tolist()], n_results=top_n)  
    
    filtered_results = []
    for i, distance in enumerate(results['distances'][0]):
        if distance <= 1:
            filtered_results.append([results['ids'][0][i], results['documents'][0][i]])
    
    if not filtered_results:
        filtered_results.append([results['ids'][0][0], results['documents'][0][0]]) 
 
    vector_results = [
        {
            "id": doc_id,
            "document": doc_content,
            "parent": doc_id.split("_chunk_")[0],   
            "rank": rank + 1  
        }
        for rank, (doc_id, doc_content) in enumerate(filtered_results)
    ] 
    return vector_results

 
def keyword_search(query, bm25_retriever_file, chunked_docs_file, top_n=5):
    """
    Performs keyword-based search using BM25 to retrieve top `n` document chunks.

    Parameters:
    - query: The query string for which relevant documents are to be retrieved.
    - top_n: Number of top results to retrieve (default is 5).

    Output:
    - Returns a list of dictionaries containing document details such as ID, content, parent document name, and rank.
    """
    
    with open(bm25_retriever_file, "rb") as f:
        retriever = pickle.load(f)

    with open(chunked_docs_file, "rb") as f:
        corpus = pickle.load(f)
    
    processed_query = query.split() 
    results = retriever.get_top_n(processed_query, corpus, top_n) 
 
    keyword_results = [
        {
            "id": f'{result.metadata.get("source", "unknown")}_chunk_{i}',
            "document": result.page_content,
            "parent": result.metadata.get("source", "unknown"),   
            "rank": i + 1  
        }
        for i, result in enumerate(results)
    ] 
    
    return keyword_results


def get_graph_schema(graph_store):
    """
    Retrieves the structured schema of the graph database.
    
    Parameters:
    - graph_store: An instance of the graph store to retrieve the schema from.
    
    Returns:    
    - A formatted string representing the structured schema of the graph.
    """

    structured_schema = graph_store.get_structured_schema
    print("Structured schema:", structured_schema)
     
    formatted_node_props = []
    for label, properties in structured_schema.get("node_props", {}).items():
        props_str = ", ".join(
            [f"{prop['property']}: {prop['type']}" for prop in properties]
        )
        formatted_node_props.append(f"{label} {{{props_str}}}")
     
    formatted_rel_props = []
    for rel_type, properties in structured_schema.get("rel_props", {}).items():
        props_str = ", ".join(
            [f"{prop['property']}: {prop['type']}" for prop in properties]
        )
        formatted_rel_props.append(f"{rel_type} {{{props_str}}}")
     
    formatted_rels = [
        f"(:{el['start']})-[:{el['type']}]->(:{el['end']})"
        for el in structured_schema.get("relationships", [])
    ]
    
    return "\n".join(
        [
            "Node properties are the following:",
            ",".join(formatted_node_props),
            "Relationship properties are the following:",
            ",".join(formatted_rel_props),
            "The relationships are the following:",
            ",".join(formatted_rels),
        ]
    )

def graph_search(user_query, graph, schema, llm_model_name="llama3.2:3b", top_n=5): 
    """
    Searches the Neo4j graph database for relevant legal judgments based on the user query.
    
    Parameters:
    - user_query (str): The query string provided by the user.
    - graph (Neo4jGraph): An instance of the Neo4j graph to perform the search on.
    - schema (str): The structured schema of the graph database.
    - llm_model: The language model used for processing the query.
    
    Returns:
    - A list of dictionaries containing relevant judgments and their scores.
    """

    extraction_prompt = PromptTemplate(
        input_variables=["user_query", "graph_schema"],
        template="""
            You are an information extraction specialist for legal searches. Your task is to extract only the relevant entities from a user query that will help in finding the most applicable legal judgments.

            Based on the following legal knowledge graph schema:
            {graph_schema}

            Extract ONLY the entities that are clearly mentioned in the user's query. Do not invent or assume information that isn't explicitly stated.

            For each entity type, extract the relevant text if present:
            - Judgment name: Extract any case names (e.g., "Smith vs. Jones")
            - Problem: Extract any legal issues or problems mentioned
            - Summary: Extract any descriptions of case outcomes or summaries
            - Appellant: Extract any mentioned appellants/petitioners
            - Respondent: Extract any mentioned respondents

            Format your response as a JSON object with only the fields that could be extracted from the query:
            \'{{
            "nameQuery": "extracted judgment name or empty string if not found",
            "problemQuery": "extracted problem or empty string if not found",
            "summaryQuery": "extracted summary or empty string if not found",
            "appellantQuery": "extracted appellant or empty string if not found",
            "respondentQuery": "extracted respondent or empty string if not found"
            }}\'

            User Query:
            {user_query}
    """
    )

    cypher_query = """
            WITH 
            COALESCE($nameQuery, "") AS nameQuery,
            COALESCE($problemQuery, "") AS problemQuery,
            COALESCE($summaryQuery, "") AS summaryQuery,
            COALESCE($appellantQuery, "") AS appellantQuery,
            COALESCE($respondentQuery, "") AS respondentQuery,
            0.3 AS threshold

            MATCH (j:Judgment)
            WITH j, threshold,
            CASE WHEN nameQuery <> "" 
                THEN apoc.text.sorensenDiceSimilarity(j.name, nameQuery) ELSE 0 END AS nameScore,
            CASE WHEN problemQuery <> "" 
                THEN apoc.text.sorensenDiceSimilarity(j.problem, problemQuery) ELSE 0 END AS problemScore,
            CASE WHEN summaryQuery <> "" 
                THEN apoc.text.sorensenDiceSimilarity(j.summary, summaryQuery) ELSE 0 END AS summaryScore,
            CASE WHEN appellantQuery <> "" THEN [(j)-[:HAS_APPELLANT]->(a:Appellant) | 
                apoc.text.sorensenDiceSimilarity(a.name, appellantQuery)] ELSE [0] END AS appellantScores,
            CASE WHEN respondentQuery <> "" THEN [(j)-[:HAS_RESPONDENT]->(r:Respondent) | 
                apoc.text.sorensenDiceSimilarity(r.name, respondentQuery)] ELSE [0] END AS respondentScores

            WITH j, nameScore, problemScore, summaryScore, threshold,
            CASE WHEN size(appellantScores) > 0 THEN reduce(s = 0, x IN appellantScores | s + x) / size(appellantScores) ELSE 0 END AS appellantScore,
            CASE WHEN size(respondentScores) > 0 THEN reduce(s = 0, x IN respondentScores | s + x) / size(respondentScores) ELSE 0 END AS respondentScore

            WITH j, nameScore, problemScore, summaryScore, appellantScore, respondentScore, threshold,
            (CASE WHEN $nameQuery <> "" THEN 1 ELSE 0 END +
            CASE WHEN $problemQuery <> "" THEN 1 ELSE 0 END +
            CASE WHEN $summaryQuery <> "" THEN 1 ELSE 0 END +
            CASE WHEN $appellantQuery <> "" THEN 1 ELSE 0 END +
            CASE WHEN $respondentQuery <> "" THEN 1 ELSE 0 END) AS propertiesProvided,
            (nameScore * 0.4 + problemScore * 0.15 + summaryScore * 0.15 + appellantScore * 0.15 + respondentScore * 0.15) AS weightedScore

            WITH j, nameScore, problemScore, summaryScore, appellantScore, respondentScore, threshold, 
            CASE WHEN propertiesProvided > 0 
                THEN weightedScore * 5 / propertiesProvided 
                ELSE 0 END AS compositeScore 

            // WHERE compositeScore > threshold
            RETURN j, compositeScore,
                nameScore, problemScore, summaryScore, appellantScore, respondentScore
            ORDER BY compositeScore DESC
            LIMIT $top_n
    """
    llm_model = OllamaLLM(model=llm_model_name, temperature=0)  
    chain = extraction_prompt | llm_model
    response = chain.invoke({"user_query": user_query, "graph_schema": schema})
    
    match = re.search(r"\{.*\}", response, re.DOTALL)
 
    if match:
        json_string = match.group(0) 
        try: 
            extracted_params = json.loads(json_string) 
            extracted_params['top_n'] = top_n
            judgements = graph.query(cypher_query, params=extracted_params)

            if judgements:
                filtered_judgements = []
                for judgement in judgements:
                    if judgement['compositeScore'] > 0.5:
                        filtered_judgements.append(judgement)
                
                if not filtered_judgements:
                    filtered_judgements = [judgements[0]]

                for i, judgement in enumerate(filtered_judgements):
                    judgement["rank"] = i + 1
                 
                return filtered_judgements
            else:
                print("No matching judgments found.")
                return []
        
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            print("Response:", response)
            return []