import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

PDF_PATH = "GAIA_FAQ.pdf"

def get_faq_context(question: str):
    """
    Busca os trechos mais relevantes do PDF de FAQ com base na pergunta do usuário.
    Retorna uma string contendo os trechos mais parecidos.
    """
    try:
        # Carrega e processa o PDF em trechos
        loader = PyPDFLoader(PDF_PATH)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=700,
            chunk_overlap=150
        )
        chunks = splitter.split_documents(docs)

        # Gera embeddings (usa variável de ambiente GEMINI_API_KEY)
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=os.getenv("GEMINI_API_KEY")
        )

        # Busca com FAISS
        db = FAISS.from_documents(chunks, embeddings)

        results = db.similarity_search(question, k=6)

        context_text = "\n\n".join([r.page_content for r in results])
        return context_text
    except Exception as e:
        # Em caso de qualquer falha (bibliotecas ausentes, PDF não encontrado, etc.),
        # não propagar a exceção — retorne contexto vazio para que o agente FAQ
        # possa responder de forma controlada. Logamos o erro para diagnóstico.
        try:
            # print simples para ajudar em debug local
            print(f"[faq_tool] erro ao gerar contexto do FAQ: {type(e).__name__}: {e}")
        except Exception:
            pass
        return ""  # contexto vazio indica que não há trecho relevante
