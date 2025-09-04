import asyncio
import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core.agent.workflow import FunctionAgent

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings



Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.llm=GoogleGenAI( model="gemini-2.0-flash")
# Define a simple calculator tool
def multiply(a: float, b: float) -> float:
    """Useful for multiplying two numbers."""
    return a * b


os.environ["GOOGLE_API_KEY"] = "AIzaSyBoebxAcA58bYh9n-gKqEdv8trm78wBjpk"
	

documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

async def search_documents(query: str) -> str:
    """Useful for answering natural language questions about an personal essay written by Paul Graham."""
    response = await query_engine.aquery(query)
    return str(response)

agent = FunctionAgent(
    tools=[multiply, search_documents],
    llm=Settings.llm,
    system_prompt="""You are a helpful assistant that can perform calculations
    and search through documents to answer questions.""",
)

async def main():
    # Run the agent
    response = await agent.run("What did the author do in college? Also, what's 7 * 8?")
    print(str(response))
    
# Run the agent
if __name__ == "__main__":
    asyncio.run(main())