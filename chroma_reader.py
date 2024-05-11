# Necessary imports
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.chroma import ChromaReader
from IPython.display import Markdown, display
import chromadb
import os
import logging
import sys




# Set up logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# Environment setup for OpenAI and ChromaDB
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
import openai
openai.api_key = os.environ["OPENAI_API_KEY"]


chroma_client = chromadb.HttpClient(host="15.168.140.170", port="8000") #Custom
# Example script to check available collections
try:
    collections = chroma_client.list_collections()  # Assuming there's a method like this
    print("Available collections:", collections)
except Exception as e:
    print("Failed to fetch collections:", e)


try:
    collection = chroma_client.get_collection("gitlab_docs")
    print("Collection fetched successfully:", collection)
except ValueError as e:
    print("Error fetching collection:", e)

# Use ChromaReader to read data from the collection
reader = ChromaReader(
    collection_name="gitlab_docs",
    persist_directory="/home/ec2-user/chroma-storage",  # Correct path confirmed
    host="15.168.140.170",  # Correct host
    port=8000  # Correct port
)

# Load documents using ChromaReader
documents = reader.load_data(limit=10)  # Adjust limit as needed

# Optional: Create an index from documents and query
index = VectorStoreIndex.from_documents(documents, StorageContext.from_defaults())
query_engine = index.as_query_engine()
response = query_engine.query("What is the number of documents in gitlab_docs?")

# Print and display the response
print(response)
display(Markdown(f"**Response:** {response}"))