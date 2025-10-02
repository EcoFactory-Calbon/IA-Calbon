import os
import pymongo
from dotenv import load_dotenv
from typing import Optional, List
from langchain.tools import tool
from langchain.pydantic_v1 import BaseModel, Field

load_dotenv()

MONGO_URL = os.getenv("MONGO_URL")

def get_db_connection():
    client = pymongo.MongoClient(MONGO_URL)
    db = client["dbInterEco"]
    return db

