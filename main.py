from agents import Agent, Runner,OpenAIChatCompletionsModel, RunConfig
from openai import AsyncOpenAI
import os
import chainlit as cl
from dotenv import load_dotenv

import asyncio
from openai.types.responses import ResponseTextDeltaEvent

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")

external_client = AsyncOpenAI(
    api_key = gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model= OpenAIChatCompletionsModel(
    model = "gemini-1.5-flash",
    openai_client= external_client
)

config = RunConfig(
    model = model,
    model_provider = "gemini",
    tracing_disabled= True
)
agent = Agent(
    name = "Nova",
    instructions = "you are a frontend expert."  
)
 

#chat start
@cl.on_chat_start
async def handle_start(): #for starting assistant 
   
   cl.user_session.set("history",[])#history save karnay kay lyi 

   #assistan's fist message
   await cl.Message(content="Hey there! 😄 I’m Nova, your smart little assistant. how I can help you?"
).send()


# for message
@cl.on_message 
async def handle_message(message:cl.Message):

    history = cl.user_session.get("history") # for save my last message
    history.append({"role": "user", "content": message.content})

    msg = cl.Message(content="")
    await msg.send()

    # Runner
    result =  Runner.run_streamed(
        agent,
        input = history,
        run_config= config
    )

    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            await msg.stream_token(event.data.delta)

    cl.user_session.set("history", history) # for save assistant last message
    history.append({"role": "assistant", "content": result.final_output})



