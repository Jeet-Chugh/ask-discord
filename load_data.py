import json
import os

import pandas as pd
import numpy as np
from openai import OpenAI
import tiktoken
from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection
)

from dotenv import load_dotenv
load_dotenv()

class LoadData():

    def __init__(self, configs: dict) -> None:
        self.client = OpenAI(api_key=os.getenv("OPENAI_KEY"))
        self.JSON_DATA_PATH = configs.get("JSON_DATA_PATH", "/data/data.json")
        self.EMBEDDING_MODEL = configs.get("EMBEDDING_MODEL", "text-embedding-3-small")
        self.EMBEDDING_DIMENSIONS = configs.get("EMBEDDING_DIMENSIONS", 512)
        self.MAX_MESSAGE_LENGTH = configs.get("MAX_MESSAGE_LENGTH", 5000)
        self.MIN_MESSAGE_LENGTH = configs.get("MIN_MESSAGE_LENGTH", 50)
        self.COLLECTION_NAME = configs.get("COLLECTION_NAME", "channel")
        self.collection = None
        
        self.load_data()

    def load_data(self) -> None:
        if self.collection_exists():
            return
        self.createCollection()
        df = self.load_json().head(2000)
        self.insert_data(df)

    def collection_exists(self) -> bool:
        return self.COLLECTION_NAME in utility.list_collections()

    def load_json(self) -> pd.DataFrame:
        with open(self.JSON_DATA_PATH, encoding='utf8') as data:
            json_data = json.load(data)
        messages = json_data["messages"]
        df = pd.json_normalize(messages)

        df = df.loc[(df["content"] != "") & (df["content"].str.len() > self.MIN_MESSAGE_LENGTH) & (df["content"].str.len() < self.MAX_MESSAGE_LENGTH) & ~(df["content"].str.startswith("http"))]
        df = df.rename(columns={"author.id" : "authorId", "author.name" : "authorName"})
        RELEVANT_COLUMNS = ["id", "timestamp", "content", "authorId", "authorName"]
        return df[RELEVANT_COLUMNS]
    
    @staticmethod
    def truncate_text(text) -> str:
        tokenizer = tiktoken.get_encoding('cl100k_base')
        tokens = tokenizer.encode(text)
        if len(tokens) > 8192:
            tokens = tokens[:8192]
        return tokenizer.decode(tokens)
    
    def insert_data(self, df: pd.DataFrame) -> None:
        slices = [df[i:i+1000] for i in range(0, len(df), 1000)]
        for s in slices:
            res = self.client.embeddings.create(input=s["content"], model=self.EMBEDDING_MODEL, dimensions=self.EMBEDDING_DIMENSIONS)
            s["embedding"] = [np.array(embedding.embedding, dtype="float32") for embedding in res.data]
            data = s.to_dict(orient="records")
            self.collection.insert(data)
        self.collection.flush()
        self.index_vectors()

    def createCollection(self) -> None:
        # +-+-------------+------------+------------------+------------------------------+
        # | | field name  | field type | other attributes |  field description           |
        # +-+-------------+------------+------------------+------------------------------+
        # |1| "id"        | VarChar    | is_primary=True  |  "primary field"             |
        # | |             |            | auto_id=False    |                              |
        # +-+-------------+------------+------------------+------------------------------+
        # |2| "timestamp" | VarChar    |                  |  "message timestamp"         |
        # +-+-------------+------------+------------------+------------------------------+
        # |3| "content"   | VarChar    |                  |  "message as a string"       |
        # +-+-------------+------------+------------------+------------------------------+
        # |4| "embedding" | FloatVector| dim=512          |  "float vector with dim 512" |
        # +-+-------------+------------+------------------+------------------------------+
        # |5| "authorId"  | VarChar    |                  |  "message author discord id" |
        # +-+-------------+------------+------------------+------------------------------+
        # |6| "authorName"| VarChar    |                  |  "message author name"       |
        # +-+-------------+------------+------------------+------------------------------+
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=50),
            FieldSchema(name="timestamp", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=self.MAX_MESSAGE_LENGTH),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.EMBEDDING_DIMENSIONS),
            FieldSchema(name="authorId", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="authorName", dtype=DataType.VARCHAR, max_length=50)
        ]
        schema = CollectionSchema(fields)
        self.collection = Collection(self.COLLECTION_NAME, schema, consistency_level="Strong")

    def index_vectors(self) -> None:
        index = {
                    "index_type": "IVF_FLAT",
                    "metric_type": "COSINE",
                    "params": {"nlist": self.EMBEDDING_DIMENSIONS},
                }
        self.collection.create_index("embedding", index)
        self.collection.create_index(field_name="id", index_name="id_index")