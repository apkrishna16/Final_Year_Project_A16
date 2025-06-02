from django.shortcuts import render
from django.conf import settings
from django.http import HttpResponse
from .retrieval import init_chromadb, init_neo4j, init_ollama
from .response_generation import find_judgements
from markdown import markdown
import os


def home(request):
    responses = ""
    user_query = ""
    # Initialize ChromaDB, Neo4j, and Ollama
    collection = init_chromadb()
    graph = init_neo4j()
    ollama_model = init_ollama("llama3.2:3b", temperature=0.5)   

    if request.method == "POST":
        user_query = request.POST.get('query', '')
        if user_query:
            responses = find_judgements(user_query, collection=collection, graph=graph, llm_model=ollama_model)
            print(f"Responses: {responses}")
            for doc_id, content in responses.items():
                html_content = markdown(content["response"])
                responses[doc_id] = html_content
    return render(request, 'home.html', {"responses": responses, "user_query": user_query})

def serve_file(request, file_name): 
    file_path = os.path.join(settings.BASE_DIR, "main", "data/Judgements/Judgements_515_txt", file_name)
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            content_lines = file.readlines()
        return render(request, "file_detail.html", {"file_name": file_name, "content_lines": content_lines})
    return render(request, "file_detail.html", {"error": "File not found."})
