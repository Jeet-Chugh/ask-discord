import streamlit as st
from load_data import LoadData
from chatbot import Chatbot
from pymilvus import connections
from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_KEY"))

configs = {
    "OPENAI_CLIENT": client,
    "CHAT_MODEL": "gpt-4o",
    "EMBEDDING_MODEL": "text-embedding-3-small",
    "JSON_DATA_PATH": "/Users/sunjeetchugh/Documents/ask-discord/data/data.json",
    "EMBEDDING_DIMENSIONS": 512,
    "MAX_MESSAGE_LENGTH": 5000,
    "MIN_MESSAGE_LENGTH": 50,
    "COLLECTION_NAME": "channel",
    "MAX_SIMILAR_EXAMPLES": 10,
    "SIMILARITY_SCORE_CUTOFF": 0.2,
}

connections.connect("default", host="localhost", port="19530")

db = LoadData(configs)
chatbot = Chatbot(configs)

st.title("ask-discord")
mode = st.selectbox("Select mode:", ["Raw", "LLM"])
user_query = st.text_input("Enter your query:")

if st.button("Search"):
    if user_query:
        if mode == "Raw":
            st.write("---")
            RESULT = chatbot.find(user_query)
            for component in RESULT:
                for line in component:
                    st.write(line.strip())
                st.write("---")
        if mode == "LLM":
            st.write("---")
            RESULT = chatbot.ask(user_query)
            st.text(RESULT)
            st.write("---")
    else:
        st.write("Please enter a query to search.")