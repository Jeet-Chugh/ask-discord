from pymilvus import connections

from load_data import LoadData
from chatbot import Chatbot

data_configs = {
    "JSON_DATA_PATH" : "/Users/sunjeetchugh/Documents/ask-discord/data/data.json",
    "EMBEDDING_MODEL" : "text-embedding-3-small",
    "EMBEDDING_DIMENSIONS" : 512,
    "MAX_MESSAGE_LENGTH" : 5000,
    "MIN_MESSAGE_LENGTH" : 50,
    "COLLECTION_NAME" : "channel2"
}

chatbot_configs = {
    "MAX_SIMILAR_EXAMPLES" : 10,
    "SIMILARITY_SCORE_CUTOFF" : 0.80
}

if __name__ == "__main__":
    connections.connect("default", host="localhost", port="19530")
    db = LoadData(data_configs)
    chatbot = Chatbot(chatbot_configs)
    
