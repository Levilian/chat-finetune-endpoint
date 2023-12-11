from fastapi import FastAPI, File, UploadFile, Form
from typing import List, Optional
from pydantic import BaseModel
import json
import random
from openai import OpenAI
import os
from io import BytesIO
from dotenv import load_dotenv
from utils import unicode_converter
from fastapi.middleware.cors import CORSMiddleware
from random import shuffle

class Message(BaseModel):
    sender_name: str
    content: str
    timestamp_ms: int
    share: Optional[dict] = None
    reactions: Optional[List[dict]] = None

class Conversation(BaseModel):
    messages: List[Message]

class SubmitPayload(BaseModel):
    participant: str
    conversations: List[List[Message]]

app = FastAPI()

# Set up CORS middleware
origins = [
    "http://localhost:5173",  # Assuming this is where your SvelteKit is being served
    "http://127.0.0.1:5173",  # Add this if you might access your client using this address
    # "http://localhost:3000",  # You can add more origins as needed
    # "https://your-production-domain.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allows all origins for the sake of example
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.post("/submit-fine-tuning")
async def train_ai(submit_data: SubmitPayload):
    # Process each JSON file and extract messages
    training_data = []

    for conversation in submit_data.conversations:
        # You might want to extract only the message's dictionary for processing
        messages = [message.dict() for message in conversation]
        messages = extract_messages(messages, submit_data.participant)
        random_conversations = generate_random_conversations(messages, submit_data.participant)
        training_data.extend(random_conversations)
    # Train your ML model here with 'training_data'

    # shuffle and split training data into 90% train and 10% validation
    shuffle(training_data)
    split_index = int(len(training_data) * 0.9)
    
    # split training data into train and validation
    train_set = training_data[:split_index]
    validation_set = training_data[split_index:]

    # Convert training and validation data to newline-delimited JSON format
    training_ndjson = "\n".join(json.dumps(conv) for conv in train_set).encode('utf-8')
    validation_ndjson = "\n".join(json.dumps(conv) for conv in validation_set).encode('utf-8')

    # Create an in-memory bytes buffer for each file
    training_buffer = BytesIO(training_ndjson)
    validation_buffer = BytesIO(validation_ndjson)

    load_dotenv()
    client = OpenAI()
    client.api_key = os.getenv("OPENAI_API_KEY")

    # Upload the buffer to OpenAI using the files.create endpoint
    training_file_response = client.files.create(
        file=training_buffer,
        purpose="fine-tune"
    )

    validation_file_response = client.files.create(
        file=validation_buffer,
        purpose="fine-tune"
    )

    # ... (handle response, etc.)
    print(training_file_response.id)
    print(validation_file_response.id)

    response = client.fine_tuning.jobs.create(
    training_file=training_file_response.id,
    validation_file=validation_file_response.id,
    model="gpt-3.5-turbo",
    suffix="test-bot",
    )

    job_id = response.id
    print(job_id)

    job_response = client.fine_tuning.jobs.retrieve(job_id)
    print(job_response.fine_tuned_model)
    return {"message": "Training started successfully", "result": {"training_file_id": training_file_response.id, "validation_file_id": validation_file_response.id, "fine_tuned_model": job_response.fine_tuned_model}}

def extract_messages(data, my_name):
    # Process the messages
    filtered_messages = []
    content_buffer = ""
    last_speaker = None
    for chat in data:
        content = chat.get('content')
        curr_speaker = chat.get('sender_name')

        # If the conversation switched speakers, append the content to the result
        if curr_speaker != last_speaker:
            if content_buffer:
                filtered_messages.append({
                    "role": "assistant" if my_name == last_speaker else "user",
                    "content": content_buffer.strip()
                })
                content_buffer = ""

            content_buffer = content
            last_speaker = curr_speaker
        else:
            # If it's the same speaker, add the content to the content buffer with a new line
            content_buffer += "\n" + content
            
    # Add the last content
    if content_buffer:
        filtered_messages.append({
            "role": "assistant" if last_speaker == my_name else "user",
            "content": content_buffer.strip()
        })
    
    return filtered_messages

def generate_system_message(my_name):
    return f"You are {my_name}. Reply just as {my_name} would. " \
                     "Don't use prior knowledge and don't say offensive or harmful things. " \
                     "Pay attention to the previous conversation."

def generate_random_conversations(messages, my_name, min_conv_length=2, max_conv_length=6):
    conversations = []
    system_message = generate_system_message(my_name)
    conversation = [{"role": "system", "content": system_message}]
    conversation_length = random.randrange(min_conv_length, max_conv_length, 2)

    for message in messages:
        conversation.append(message)
        conversation_length -= 1

        # When the conversation reaches the intended length
        if conversation_length == 0:
            # Only add conversations that end with an assistant message
            if conversation[-1]["role"] == "assistant":
                conversations.append({"messages": conversation})
            
            # Start a new conversation
            conversation = [{"role": "system", "content": system_message}]
            conversation_length = random.randrange(min_conv_length, max_conv_length)

    # Check if there's a pending conversation that wasn't added because the loop ended
    if conversation_length != random.randrange(min_conv_length, max_conv_length, 2) and conversation[-1]["role"] == "assistant":
        conversations.append({"messages": conversation})

    return conversations

@app.post("/process-json")
async def process_json(files: List[UploadFile] = File(...)):
    processed_files = []

    for file in files:
        content = await file.read()
        # Convert latin-1 encoded strings to utf-8 inside the JSON object
        decoded_data = unicode_converter(json.loads(content))

        # Sorting messages by timestamp
        messages = sorted(decoded_data['messages'], key=lambda x: x['timestamp_ms'])

        # Filter out messages that are not strings, including those with unwanted content
        strings_to_remove = ['Liked a message', 'Reacted to your message', 'sent an attachment']
        filtered_messages = [
            message for message in messages
            if isinstance(message.get('content'), str) and not any(s in message['content'] for s in strings_to_remove)
        ]

        # Structuring the data in the same format as your pandas DataFrame
        structured_data = {
            'participants': decoded_data['participants'],
            'messages': filtered_messages
        }

        processed_files.append(structured_data)
    # Return the list of processed files
    return processed_files



@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Optional[str] = None):
    return {"item_id": item_id, "q": q}