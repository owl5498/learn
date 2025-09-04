# https://langchain-ai.github.io/langgraph/tutorials/get-started/5-customize-state/

from typing import Annotated

from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
import os
from langgraph.types import interrupt, Command  # 导入Command类型
from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool


os.environ["TAVILY_API_KEY"] = "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"
os.environ["GOOGLE_API_KEY"] = "AIzaSyBoebxAcA58bYh9n-gKqEdv8trm78wBjpk"

llm = init_chat_model("google_genai:gemini-2.0-flash")

class State(TypedDict):
    messages: Annotated[list, add_messages]
    name: str
    birthday: str

graph_builder = StateGraph(State)

@tool
# tool_call_id 用来表示哪次工具调用的，关联工具的调用和工具的返回。这样ai大模型或者langchain才不会混乱
def human_assistance(
    name: str, birthday: str, tool_call_id: Annotated[str, InjectedToolCallId]
) -> str:
    """Request assistance from a human."""
    human_response = interrupt(
        {
            "question": "Is this correct?",
            "name": name,
            "birthday": birthday,
        },
    )
    if human_response.get("correct", "").lower().startswith("y"):
        verified_name = name
        verified_birthday = birthday
        response = "Correct"
    else:
        verified_name = human_response.get("name", name)
        verified_birthday = human_response.get("birthday", birthday)
        response = f"Made a correction: {human_response}"

    # ToolMessage 是专门表示工具的返回结果的，然后传递给langchain或者ai 大模型
    # ToolMessage内部的属性有，name，tool_call_id，content等。
    state_update = {
        "name": verified_name,
        "birthday": verified_birthday,
        "messages": [ToolMessage(response, tool_call_id=tool_call_id)],
    }
    # 更新全局状态, Command是专门用来更新全局状态，或者请求让流程停下来/继续运行的
    return Command(update=state_update)

tavily_tool = TavilySearch(max_results=2)
tools = [tavily_tool, human_assistance]
llm_with_tools = llm.bind_tools(tools)
 
# chatbot只是决定要不要调用工具，如果要调用工具，会生成ToolCall结构交给ToolNode去执行。
# ToolNode会将工具结果以ToolMessage结果返回
# 如果不需要调用工具，那么chatbot就直接以AIMessage返回
def chatbot(state: State):
    message = llm_with_tools.invoke(state["messages"])
    return {"messages": [message]}

graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)

graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

memory = InMemorySaver()
graph = graph_builder.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "1"}}

user_input = (
    "Can you look up when LangGraph was released? "
    "When you have the answer, use the human_assistance tool for review."
)
events = graph.stream(
    {"messages": [{"role": "user", "content": user_input}]},
    config,
    stream_mode="values",   # 这种模式会持续输出状态的变化
)
for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()

# 这里的resume是让流程继续的意思。
human_command = Command(
    resume={
        "name": "LangGraph",
        "birthday": "Jan 17, 2024",
    },
)

events = graph.stream(human_command, config, stream_mode="values")
for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()