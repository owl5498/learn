
import os
from llama_index.core import Settings
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

#os.environ["OPENAI_API_KEY"] = ""
os.environ["GOOGLE_API_KEY"] = ""
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
#Settings.llm = None  # 添加这一行来禁用LLM
Settings.llm=GoogleGenAI( model="gemini-2.0-flash")  # query_engine.query()的时候，好像还是会调用大模型

documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()
response = query_engine.query("What was the first programming language Paul Graham tried to write?")
print(response)