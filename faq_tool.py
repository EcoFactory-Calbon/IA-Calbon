import os
import pymongo
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_google_genai import GoogleGenerativeAIEmbeddings

MONGO_URL = os.getenv("MONGO_URL")

DB_NAME = "dbInterEco"
COLLECTION_NAME = "faq_embeddings"
INDEX_NAME = "faq_vector_index" 

try:
    client = pymongo.MongoClient(MONGO_URL)
    collection = client[DB_NAME][COLLECTION_NAME]

    embeddings_model = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=os.getenv("GEMINI_API_KEY")
    )

    vector_store = MongoDBAtlasVectorSearch(
        collection=collection,
        embedding=embeddings_model,
        index_name=INDEX_NAME,
        text_key="text", 
        embedding_key="embedding" 
    )
except Exception as e:
    print(f"[faq_tool] ERRO CRÍTICO ao inicializar o Atlas Vector Search: {e}")
    vector_store = None


def get_faq_context(question: str):
    """
    Busca os trechos mais relevantes do MongoDB Atlas com base na pergunta do usuário.
    Retorna uma string contendo os trechos mais parecidos.
    """
    if vector_store is None:
        print("[faq_tool] Erro: vector_store não foi inicializado.")
        return ""
        
    try:
        results = vector_store.similarity_search(question, k=6)

        context_text = "\n\n".join([r.page_content for r in results])
        return context_text
        
    except Exception as e:
        try:
            print(f"[faq_tool] erro ao buscar contexto do Atlas: {type(e).__name__}: {e}")
        except Exception:
            pass
        return ""