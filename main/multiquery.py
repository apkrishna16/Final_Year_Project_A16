import re
import json
from langchain.prompts import PromptTemplate   
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_ollama import OllamaLLM

def generate_multi_query(query, llm_model_name="llama3.2:3b", n=3):
    """
    Generates alternative query variations for retrieving documents from a vector database.

    Parameters:
    - query: The original query string.
    - llm_model: The language model used for generating variations.
    - n: Number of variations to generate (default is 3).

    Output:
    - Returns a list containing the original query and its variations.
    """
    
    response_schemas = [
        ResponseSchema(name=f"variation_{i+1}", description=f"Variation {i+1} of the query")
        for i in range(n)
    ] 
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
 
    prompt = PromptTemplate(
    input_variables=["question", "n"],
    template="""
            ask: You are an AI assistant. Your task is to **generate {n} alternative variations of the following legal question** to retrieve relevant documents from a vector database. 
            By generating multiple perspectives on the user question, your goal is to help the user overcome some of the limitations of distance-based similarity search.
            
            Instructions:
            - Format the output strictly as JSON with fields: {field_names}.
            - You must always return **single valid JSON fenced by a markdown code block**. Do not return any additional text.
            
            Original Question: {question}
            Format Instuctions: {format_instructions}
        """,
        partial_variables={
            "format_instructions": output_parser.get_format_instructions(),
            "field_names": ", ".join([f"variation_{i+1}" for i in range(n)]),
        },
    )
    
    llm_model = OllamaLLM(model=llm_model_name, temperature=0.95)
    chain = prompt | llm_model 
    response = chain.invoke({"question": query, "n": n})
    
    print(f"Response: {response}") 
    match = re.search(r"\{.*\}", response, re.DOTALL)
 
    if match:
        json_string = match.group(0)
        try:
            parsed_response = json.loads(json_string)
        except json.JSONDecodeError as e:
            print("Error in extracted JSON. Returning {}. Error: ", e) 
            parsed_response = {}
    else:
        print("No JSON found in the response. Returning {}") 
        parsed_response = {}

    print(f"Parsed Response: {parsed_response}")
    variations = [query]
   
    variations.extend([
        parsed_response[f"variation_{i+1}"]
        for i in range(n) if parsed_response 
    ])
 
    return variations