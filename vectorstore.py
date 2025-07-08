import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import PGVector
from dotenv import load_dotenv
load_dotenv()

# Initialize embeddings and vectorstore
env_url = os.getenv("DATABASE_URL")
embeddings = OpenAIEmbeddings()
vectordb = PGVector(
    connection_string=env_url,
    embedding_function=embeddings,
)

# Retrieve top-k docs
def get_top_k_docs(query: str, k: int = 5):
    docs = vectordb.similarity_search(query, k=k)
    # Assuming each doc has page_content and metadata
    results = []
    for d in docs:
        results.append({
            "content": d.page_content,
            **(d.metadata or {})
        })
    return results