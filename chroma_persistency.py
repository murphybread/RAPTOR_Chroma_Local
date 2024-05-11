# import
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import chromadb

# set up OpenAI
import os
import getpass

#os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key:")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY") #Custom
import openai

# openai.api_key = os.environ["OPENAI_API_KEY"]
COLLECTION_NAME = "bge_m3"

# define embedding function
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")


# save to disk
def save_to_disk(documents_path,db_path, embed_model , collection_name):
    documents = SimpleDirectoryReader(documents_path).load_data()
    
    db = chromadb.PersistentClient(path=db_path)
    chroma_collection = db.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context, embed_model=embed_model
    )
    return index

# save_to_disk("./data" ,"./chroma_db", embed_model, COLLECTION_NAME)


def load_from_disk(db_path, embed_model, collection_name):
    db = chromadb.PersistentClient(path=db_path)
    chroma_collection = db.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        embed_model=embed_model,
    )
    return index

index = load_from_disk("./chroma_db", embed_model , COLLECTION_NAME)

# Query Data from the persisted index
query_engine = index.as_query_engine()
response = query_engine.query("What number is about gradio guide? I need more than 3 numbers ")
print(response)

# # Assuming you have a function to get all documents or specific details
# documents = index.get_documents()  # This is a hypothetical function; replace with actual API call
# for doc in documents:
#     print(doc)
