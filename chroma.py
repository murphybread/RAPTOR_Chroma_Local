# import
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from IPython.display import Markdown, display
import chromadb

# set up OpenAI
import os
import getpass

#os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key:")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY") #Custom
import openai

openai.api_key = os.environ["OPENAI_API_KEY"]

# create client and a new collection
#chroma_client = chromadb.EphemeralClient()
chroma_client = chromadb.HttpClient(host="15.168.140.170", port="8000") #Custom

chroma_collection = chroma_client.get_or_create_collection("gitlab_docs")


# define embedding function
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

# load documents
documents = SimpleDirectoryReader("./").load_data()

# set up ChromaVectorStore and load in data
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

#vector_store = ChromaVectorStore.getCollection(chroma_collection="quickstart") #Custom
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context, embed_model=embed_model
)

# Query Data
query_engine = index.as_query_engine()
response = query_engine.query("What is the number of documents of gitlab?")
print(response)



