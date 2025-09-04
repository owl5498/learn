from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
import asyncio
from autogen_ext.models.openai import OpenAIChatCompletionClient
from io import BytesIO

import PIL
import requests
from autogen_agentchat.messages import MultiModalMessage
from autogen_core import Image


# Define a model client. You can use other model client that implements
# the `ChatCompletionClient` interface.
model_client = OpenAIChatCompletionClient(
    model="gemini-2.0-flash",
    api_key="AIzaSyBoebxAcA58bYh9n-gKqEdv8trm78wBjpk",
)


# Define a simple function tool that the agent can use.
# For this example, we use a fake weather tool for demonstration purposes.
async def get_weather(city: str) -> str:
    """Get the weather for a given city."""
    return f"The weather in {city} is 73 degrees and Sunny."


# Define an AssistantAgent with the model, tool, system message, and reflection enabled.
# The system message instructs the agent via natural language.
agent = AssistantAgent(
    name="weather_agent",
    model_client=model_client,
    tools=[get_weather],
    system_message="You are a helpful assistant.",
    reflect_on_tool_use=True,
    model_client_stream=True,  # Enable streaming tokens from the model client.
)


async def main() -> None:
    try:
        # 4.1 加载图片（同步操作，提前完成避免阻塞资源释放）
        response = requests.get("https://picsum.photos/300/200")
        response.raise_for_status()  # 避免图片请求失败导致后续报错
        pil_image = PIL.Image.open(BytesIO(response.content))
        img = Image(pil_image)
        
        # 4.2 构建多模态消息并调用代理
        multi_modal_message = MultiModalMessage(
            content=["Can you describe the content of this image?", img], 
            source="user"
        )
        result = await agent.run(task=multi_modal_message)
        
        # 4.3 打印结果（增加类型判断，避免报错）
        if hasattr(result, "messages") and result.messages:
            last_msg = result.messages[-1]
            if hasattr(last_msg, "content"):
                print("Agent Response:\n", last_msg.content)
    finally:
        # 关键：无论是否报错，都主动关闭模型客户端（释放异步连接）
        await model_client.close()



if __name__ == "__main__":
    asyncio.run(main())  