from pymilvus import Collection
import numpy as np
import pandas as pd
import re

from openai import OpenAI
import os


class Chatbot:

    def __init__(self, configs) -> None:
        self.client = configs["OPENAI_CLIENT"]
        self.MAX_SIMILAR_EXAMPLES = configs.get("MAX_SIMILAR_EXAMPLES", 10)
        self.SIMILARITY_SCORE_CUTOFF = configs.get("SIMILARITY_SCORE_CUTOFF", 0.0)
        self.CHAT_MODEL = configs.get("CHAT_MODEL", "gpt-4o")
        self.COLLECTION_NAME = configs.get("COLLECTION_NAME", "channel")
        self.EMBEDDING_MODEL = configs.get("EMBEDDING_MODEL", "text-embedding-3-small")
        self.EMBEDDING_DIMENSIONS = configs.get("EMBEDDING_DIMENSIONS", 512)

        self.collection = Collection(self.COLLECTION_NAME)
        self.collection.load()

        self.search_params = {"metric_type": "COSINE", "params": {"nprobe": 8}}

    def ask(self, query):
        results = self.find_similar_messages(query)
        prompt = f"""
            Construct a helpful response to the following query:"{query}"
            To form your response, only use the following texts (comma delimited): "{",".join([row["content"] for row in results])}"
            Do not use ANY external information when crafting your response, stick strictly to the text provided.
                  """
        chat = self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}], model=self.CHAT_MODEL
        )
        return chat.choices[0].message.content

    def find(self, query):
        results = self.find_similar_messages(query)
        res = []
        for result in results:
            res.append([
            f"Similarity: {result["distance"] * 100:.2f}%",
            f"Timestamp: {result["timestamp"]}",
            f"Author: {result["authorName"]}",
            f"Message: {result["content"]}",
            f"ID: {result["id"]}"
        ])
        return res

    def find_similar_messages(self, query):
        embed_response = self.client.embeddings.create(
            input=query,
            model=self.EMBEDDING_MODEL,
            dimensions=self.EMBEDDING_DIMENSIONS,
        )
        query_vector = np.array(embed_response.data[0].embedding)
        output_fields = ["id", "timestamp", "content", "authorId", "authorName"]
        top_k = self.collection.search(
            [query_vector],
            "embedding",
            self.search_params,
            limit=self.MAX_SIMILAR_EXAMPLES,
            output_fields=output_fields,
        )
        results = []
        for row in top_k[0]:
            if row.distance > self.SIMILARITY_SCORE_CUTOFF:
                results.append(
                    {
                        "distance": row.distance,
                        "id": row.id,
                        "content": row.content,
                        "timestamp": row.timestamp,
                        "authorName": row.authorName,
                    }
                )
        return results
