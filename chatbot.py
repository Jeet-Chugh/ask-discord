from openai import OpenAI
from pymilvus import (
    connections,
    utility,
    Collection
)

class Chatbot():

    def __init__(self, configs) -> None:
        self.MAX_SIMILAR_EXAMPLES = configs.get("MAX_SIMILAR_EXAMPLES", 10)
        self.SIMILARITY_SCORE_CUTOFF = configs.get("SIMILARITY_SCORE_CUTOFF", 0.80)