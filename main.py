from agents import Agent, Runner, OpenAIChatCompletionsModel, RunConfig
from openai import AsyncOpenAI
import os
import chainlit as cl
import random
from dotenv import load_dotenv
from openai.types.responses import ResponseTextDeltaEvent

load_dotenv()  # Load .env file


API_KEYS = [
    "GEMINI_KEY_1",
    "GEMINI_KEY_2",
    "GEMINI_KEY_3",
    "GEMINI_KEY_4",
    "GEMINI_KEY_5",
    "GEMINI_KEY_6",
    "GEMINI_KEY_7",
    "GEMINI_KEY_8",
]


# Load 8 Gemini API keys
gemini_keys = [os.getenv(f"GEMINI_KEY_{i}") for i in range(1, 9)]

# Check if all keys loaded properly
if not all(gemini_keys):
    raise ValueError("Some GEMINI keys are not loaded from .env")

# Function to get a random key
def get_random_key():
    return random.choice(gemini_keys)

# Function to create AsyncOpenAI client with a random key
def create_client():
    return AsyncOpenAI(
        api_key=get_random_key(),
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )

# Initialize model (client will be updated per request)
model = OpenAIChatCompletionsModel(
    model="gemini-1.5-flash",
    openai_client=create_client()
)

config = RunConfig(
    model=model,
    model_provider="gemini",
    tracing_disabled=True
)

agent = Agent(
    name="Nova",
    instructions="you are a frontend expert."
)

# Chat start
@cl.on_chat_start
async def handle_start():
    cl.user_session.set("history", [])
    await cl.Message(
        content="Hey there! 😄 I’m Nova, your smart little assistant. How can I help you?"
    ).send()

# On message
@cl.on_message
async def handle_message(message: cl.Message):
    history = cl.user_session.get("history", [])
    history.append({"role": "user", "content": message.content})

    msg = cl.Message(content="")
    await msg.send()

    # Use a new client with a random key for this request
    model.openai_client = create_client()

    result = Runner.run_streamed(
        agent,
        input=history,
        run_config=config
    )

    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            await msg.stream_token(event.data.delta)

    history.append({"role": "assistant", "content": result.final_output})
    cl.user_session.set("history", history)



