# 整体上还有问题，可能是到了MaxMessageTermination之后才退出的，但是退款没有正确执行

from typing import Any, Dict, List

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import HandoffTermination, TextMentionTermination,MaxMessageTermination  
from autogen_agentchat.messages import HandoffMessage
from autogen_agentchat.teams import Swarm
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
import asyncio


def refund_flight(flight_id: str) -> str:
    """Refund a flight"""
    return f"Flight {flight_id} refunded"


model_client = OpenAIChatCompletionClient(
    model="gemini-2.0-flash",
    api_key="AIzaSyBoebxAcA58bYh9n-gKqEdv8trm78wBjpk",
)

travel_agent = AssistantAgent(
    "travel_agent",
    model_client=model_client,
    handoffs=["flights_refunder", "user"],
    system_message="""You are a travel agent.
    1. If the user needs to refund a flight, immediately handoff to flights_refunder (do NOT call other tools).
    2. After flights_refunder completes the refund and hands off back to you:
       - Send a confirmation message (e.g., "Your flight has been refunded successfully.").
       - Then immediately output "TERMINATE" in your message to end the process.
    3. If you need user info, send your question first, then handoff to user.
    4. Do NOT handoff back to flights_refunder after they return to you (no repeated switches).""",
)

flights_refunder = AssistantAgent(
    "flights_refunder",
    model_client=model_client,
    handoffs=["travel_agent", "user"],
    tools=[refund_flight],
    system_message="""You are an agent specialized in refunding flights.
    1. First, ask the user for the flight reference number (only if not provided).
    2. Use the refund_flight tool with the flight_id to process the refund.
    3. After refund success:
       - Send a message like "Flight XXXXX has been refunded."
       - Immediately handoff to travel_agent (do NOT send empty messages or extra switches).
    4. Do NOT generate empty messages; only send meaningful content.""",
)


termination = HandoffTermination(target="user") | TextMentionTermination("TERMINATE") | MaxMessageTermination(max_messages=20) 
team = Swarm([travel_agent, flights_refunder], termination_condition=termination)


task = "I need to refund my flight."


async def run_team_stream() -> None:
    task_result = await Console(team.run_stream(task=task))
    last_message = task_result.messages[-1]
   
    while isinstance(last_message, HandoffMessage) and last_message.target == "user":
        user_message = input("User: ")

        task_result = await Console(
            team.run_stream(task=HandoffMessage(source="user", target=last_message.source, content=user_message))
        )
        last_message = task_result.messages[-1]
        
       


async def run_team() -> None:
    try:
        await run_team_stream()
    finally:
        await model_client.close()
        
if __name__ == "__main__":
    asyncio.run(run_team())  
    
    

