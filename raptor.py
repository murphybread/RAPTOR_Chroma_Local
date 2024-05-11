# =========== data load ===========

from llama_index.core import SimpleDirectoryReader

documents = SimpleDirectoryReader(input_dir="./data").load_data()

# =========== raptor pack configure ===========

from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.packs.raptor import RaptorPack

import os

import chromadb
import openai

client = chromadb.PersistentClient(path="./raptor_paper_db")
collection = client.get_or_create_collection("raptor")
vector_store = ChromaVectorStore(chroma_collection=collection)

openai.api_key = os.environ["OPENAI_API_KEY"]
llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)
embed_model = OpenAIEmbedding(model="text-embedding-3-small")


def create_raptor_pack ():
    raptor_pack = RaptorPack(
        documents,
        embed_model=embed_model,
        llm=llm,  # used for generating summaries
        vector_store=vector_store,  # used for storage
        similarity_top_k=2,  # top k for each layer, or overall top-k for collapsed
        mode="collapsed",  # sets default mode
        transformations=[SentenceSplitter(chunk_size=400, chunk_overlap=50)]
    )
    return raptor_pack

# create_raptor_pack()

# =========== summary module ===========
from llama_index.packs.raptor.base import SummaryModule
from llama_index.packs.raptor import RaptorRetriever

summary_prompt = "As a professional summarizer, create a concise and comprehensive summary of the provided text, be it an article, post, conversation, or passage with as much detail as possible."

def create_summary_raptor ():
    summary_module = SummaryModule(
    llm=llm, summary_prompt=summary_prompt, num_workers=16)
    
    pack = RaptorPack(
    documents, llm=llm, embed_model=embed_model, summary_module=summary_module
)


    return pack
# create_summary_raptor()

# Adding SummaryModule you can configure the summary prompt and number of workers doing summaries.


# ================= retrieve  documents =================
retriever = RaptorRetriever(
    [],
    embed_model=OpenAIEmbedding(
        model="text-embedding-3-small"
    ),  # used for embedding clusters
    llm=OpenAI(model="gpt-3.5-turbo", temperature=0.1),  # used for generating summaries
    vector_store=vector_store,  # used for storage
    similarity_top_k=2,  # top k for each layer, or overall top-k for collapsed
    mode="tree_traversal"# "collapsed",  # sets default mode
)


from llama_index.core.base.response.schema import Response  # Adjust according to the actual module path
from llama_index.core.query_engine import RetrieverQueryEngine
query_engine = RetrieverQueryEngine.from_args(
    retriever, llm=OpenAI(model="gpt-3.5-turbo", temperature=0.1)
)
response = query_engine.query("What is the nubers of gradio guide? with exact paths")

# Accessing and printing response attributes
if isinstance(response, Response):  # Make sure the response is an instance of Response
    print("Response Text:", response.response)
    print("Source Nodes:")
    for node_with_score in response.source_nodes:
        # Assuming 'node_with_score.node' is an object of class 'TextNode' or similar
        node = node_with_score.node  # This is the actual node object
        score = node_with_score.score  # This is the score

        # Access properties of 'node'. You need to know the exact properties/methods of this node.
        # Here we assume 'node' has properties 'id' and 'text' directly accessible.
        # Adjust these lines if 'node' uses different methods to access 'id' and 'text'.
        node_id = getattr(node, 'id', 'Unknown ID')  # Using getattr to safely access attributes
        node_text = getattr(node, 'text', 'No text available')

        print(f"Node ID: {node_id}, Text: {node_text}, Score: {score}")
    
    print("Metadata:", response.metadata)
else:
    print("Response is not of type Response, actual type:", type(response))