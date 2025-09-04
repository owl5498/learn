import asyncio
import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.workflow import Context
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings


Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.llm=GoogleGenAI( model="gemini-2.0-flash")
os.environ["GOOGLE_API_KEY"] = "AIzaSyBoebxAcA58bYh9n-gKqEdv8trm78wBjpk"

# messages = [
#     ChatMessage(role="system", content="You are a helpful assistant."),
#     ChatMessage(role="user", content="Tell me a joke."),
# ]
# chat_response = Settings.llm.chat(messages)
# print(str(chat_response))
# 
# 
agent = FunctionAgent(
    llm=Settings.llm,
    system_prompt="""You are a helpful assistant.""",
)
ctx = Context(agent)
async def main():
    response = await agent.run("My name is Logan", ctx=ctx)
    response = await agent.run("What is my name?", ctx=ctx)
    print(str(response))
    
# Run the agent
if __name__ == "__main__":
    asyncio.run(main())