# Load, embed and chat away

Load your preferred PDF documents and specify the chunk size and number of documents you wish to retrieve from the vector database to answer your query. To interact with the document embeddings, simply initiate a conversation with the chatbot.
Note: To utilize the vector database, you will require a Pinecone API key, which is available for free for a single index.
Llama3 is used as the LLM through Groq Inference, so please get an API key from Groq

If you planning to deploy this in streamlit cloud, then you need to follow below instructions. If you plan to use this locally, then make sure to set up your environment variables.

https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python -m streamlit run chat.py
```

## License

[MIT](https://choosealicense.com/licenses/mit/)
