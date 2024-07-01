# ask-discord

Blazing fast semantic search for discord channels

## Overview

`ask-discord` enables users to semantically search through a dataset of Discord messages. There are two main search modes: 
1. **Raw**: Displays the most similar messages based on ANNS and cosine similarity.
2. **LLM**: Feeds raw results into an LLM to generate a chatbot-like response.

## Getting Started

### Prerequisites

- Docker
- OpenAI API Key
- Milvus 2.4.4
- Python 3.12.3

### Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/ask-discord.git
    cd ask-discord
    ```

2. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Set up environment variables**:
    Create a `.env` file in the root directory and add your OpenAI API key:
    ```env
    OPENAI_KEY=your_openai_api_key
    ```

4. **Start Milvus**:
    Follow the [Milvus installation guide](https://milvus.io/docs/install_standalone-docker.md) to set up and start Milvus. (Requires a recent Docker installation)

### Running the Application

1. **Generate the data**:
      Download your channels of interest using [Discord Chat Exporter](https://github.com/Tyrrrz/DiscordChatExporter). Read [this guide](https://github.com/Tyrrrz/DiscordChatExporter/blob/master/.docs/Token-and-IDs.md) if you have trouble getting your Token and Channel IDs. This is not an endorsement as downloading channels may violate Discord TOS.

2. **Load the data**:
    Ensure the JSON data file is in the correct path specified in `configs`. Modify the path in the main file if needed.

3. **Run the Streamlit application**:
    ```bash
    streamlit run ask-discord.py
    ```

4. **Access the application**:
    Open your web browser and go to `http://localhost:8501`.

## Documentation

### Project Files

- **ask-discord.py**: Main entry point into the Streamlit application. It initializes the configurations, connects to Milvus and OpenAI, and sets up Database and Chatbot.
- **load_data.py**: Handles loading and processing of JSON discord data, and manages the Milvus collection.
- **chatbot.py**: Contains the `Chatbot` class which handles querying Milvus and interacting with Raw/LLM mode.

### Configuration

Configurations are managed through a dictionary in `ask-discord.py`. These include:

- `OPENAI_CLIENT`: OpenAI client instance.
- `CHAT_MODEL`: The model to use for chat (e.g., `gpt-4o`).
- `EMBEDDING_MODEL`: The model to use for generating embeddings.
- `JSON_DATA_PATH`: Path to the JSON data file.
- `EMBEDDING_DIMENSIONS`: vector dimensions.
- `MAX_MESSAGE_LENGTH`: Maximum number of characters in a message to be considered.
- `MIN_MESSAGE_LENGTH`: Minimum number of characters in a message to be considered.
- `COLLECTION_NAME`: Name of the Milvus collection.
- `MAX_SIMILAR_EXAMPLES`: Maximum number of similar messages to retrieve.
- `SIMILARITY_SCORE_CUTOFF`: Cutoff for similarity score.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue to discuss any changes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

## Acknowledgements

- [Discord Chat Exporter](https://github.com/Tyrrrz/DiscordChatExporter)

