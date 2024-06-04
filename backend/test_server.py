# Import the necessary modules required for the app
from dotenv import load_dotenv  # To load environment variables from .env file
from upstash_redis import Redis  # Interface with Upstash Redis
import uvicorn  # ASGI server for running the app
from fastapi import FastAPI, Query  # Web framework for building APIs
import os  # To interact with the operating system's environment variables
import uuid  # For generating unique identifiers
import json  # To work with JSON data
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage  # Messaging utilities from langchain_core
from pinecone import Pinecone
from pydantic import BaseModel
from test_assistant import EducatorAssistant
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables from a .env file into program's environmentup
load_dotenv()

upstash_redis_url = os.getenv("UPSTASH_REDIS_REST_URL")
upstash_redis_token = os.getenv("UPSTASH_REDIS_REST_TOKEN")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")
openai_api_key=os.getenv("OPENAI_API_KEY")
# Create a Redis client using credentials from environment variables
redis = Redis(
    url=upstash_redis_url, 
    token=upstash_redis_token
)

print(f'================{pinecone_api_key}{pinecone_index_name}')
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index(pinecone_index_name)
print(f'================pinecone brief information{index.describe_index_stats()}')
# Initialize the FastAPI application
app = FastAPI()
# You can allow specific origins or use ["*"] for all origins
origins = [
"*"
]

# Add the middleware to the application
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # List of allowed origins
    allow_credentials=True,  # Allow cookies
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)
sessions = {}
from typing import Dict
from datetime import datetime, timedelta

# Dictionary to hold client_id -> last activity timestamp
client_last_activity: Dict[str, datetime] = {}

# Create an OpenAI client with the API key from environment variables

def generate_unique_uuid():
    return str(uuid.uuid1())
# Function to set (save) messages in Redis under a specific user_id
def set_messages(user_id, messages):
    res = redis.set(user_id, {"data": messages})
    return res

# Function to get (retrieve) messages from Redis associated with a specific user_id
def get_messages(user_id):
    # Fetch data from Upstash Redis
    messages = redis.get(user_id)
    if messages == None:
        return []
    
    print(messages)
    return json.loads(messages)["data"]

# Function to handle incoming messages, update the conversation, and retrieve it

class MessageList(BaseModel):
    session_id: str
    message: str
@app.post("/start")
async def say_hello():
    session_id = generate_unique_uuid()
    assistant = EducatorAssistant(
           index
        )
    sessions[session_id] = assistant
    return {
        "session_id": session_id, 
        "message": "Hello, How can I help you?"
    }
@app.post("/end")
async def end(req: MessageList):
    assistant = None
        # print(f"Received request: {req}")
    if req.session_id in sessions:
        print("Session is found!")
        assistant = sessions[req.session_id]
        
        print(f"Session id: {req.session_id}")
        result = set_messages(req.session_id, assistant.history)

        sessions.pop(req.session_id)
        if result == True:
            print("save success")
            return "save success"
        else:
            print("save failed")
            return "save failed"
    else:
        print("Session id not found!")
        return "not found"

from fastapi import Header, HTTPException, Depends

@app.post("/chat")
async def chat_with_teacher_agent(req: MessageList):
    assistant = None
        # print(f"Received request: {req}")
    if req.session_id in sessions:
        print("Session is found!")
        assistant = sessions[req.session_id]
        
        print(f"Session id: {req.session_id}")
        client_last_activity[req.session_id] = datetime.utcnow()
    
    else:
        print("Creating new session")
        assistant = EducatorAssistant(
           index
        )
        sessions[req.session_id] = assistant
    print(req)
    
    response = assistant.chat(req.message)
    return {"message": response}
from fastapi import BackgroundTasks

async def cleanup_chat_histories():
    current_time = datetime.utcnow()
    print(f'=========current time: {current_time}')
    for session_id, last_activity in list(client_last_activity.items()):

        print(f'=========session_id: {session_id}')

        if current_time - last_activity > timedelta(seconds=1800):  # e.g., 30 minutes timeout
            print(f'=========current time: {current_time}')
            print(f'=========last_activity time: {last_activity}')
            # Delete the client's chat history here ...
            message_list = MessageList(session_id=session_id, message='end')
            await end(message_list)
            del client_last_activity[session_id]
import asyncio
async def repeat_cleanup_task(wait_time: int):
    while True:
        await cleanup_chat_histories()
        await asyncio.sleep(wait_time)

@app.on_event("startup")
async def on_startup():
    asyncio.create_task(repeat_cleanup_task(1800))  # Run the task every 1800 seconds (30 minutes)
# And you would trigger this function using BackgroundTasks

# Main entry point to run the app with Uvicorn when script is executed directly
if __name__ == "__main__":
    uvicorn.run("test_server:app", host="0.0.0.0", port=8001, reload=True)
