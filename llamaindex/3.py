import chromadb
import os
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

os.environ["GOOGLE_API_KEY"] = "AIzaSyBoebxAcA58bYh9n-gKqEdv8trm78wBjpk"
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.llm=GoogleGenAI( model="gemini-2.0-flash")

# load some documents
# excel表格好像不能组合行和列，而是每个单元格有什么信息，就读取什么信息。不理解它是表格格式的。
#documents = SimpleDirectoryReader("./data").load_data()  

# 这篇文章是中文的，感觉对中文的理解不是很好，我问他请假的流程它都不知道
documents = SimpleDirectoryReader(input_files=[r"D:\work\test_code\py\llamaindex\data\博瀚智能员工手册完整版.pdf"]).load_data()  

# initialize client, setting path to save data
db = chromadb.PersistentClient(path="./chroma_db")

# create collection
chroma_collection = db.get_or_create_collection("quickstart")

# assign chroma as the vector_store to the context
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# create your index
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)

# create a query engine and query
query_engine = index.as_query_engine()
response = query_engine.query("summary the documents?")
print(response)