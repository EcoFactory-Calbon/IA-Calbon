import os
import pymongo
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch

load_dotenv()

PDF_PATH = "GAIA_FAQ.pdf"
MONGO_URL = os.getenv("MONGO_URL")
DB_NAME = "dbInterEco"
COLLECTION_NAME = "faq_embeddings"
INDEX_NAME = "faq_vector_index" 

loader = PyPDFLoader(PDF_PATH)
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=150
)
chunks = splitter.split_documents(docs)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

client = pymongo.MongoClient(MONGO_URL)
collection = client[DB_NAME][COLLECTION_NAME]

MongoDBAtlasVectorSearch.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection=collection,
    index_name=INDEX_NAME,
    text_key="text",
    embedding_key="embedding"
)