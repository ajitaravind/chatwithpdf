# Load, embed and chat away

To get started, users can upload their PDF documents and specify the chunk size and number of documents for retrieval from the vector database. The platform utilizes a Streamlit frontend for a user-friendly experience.
Note: API keys from Groq and Pinecone are required. Llama3 is the LLM used, Groq helps with super fast inference. Pinecone is used as vector database, log in to pinecone website to get an API key(upto one index,
its free for usage)
For deployment, you have two options: Streamlit Cloud or local setup. Follow the provided instructions for Streamlit Cloud deployment, or ensure your environment variables are set up for local use.

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
