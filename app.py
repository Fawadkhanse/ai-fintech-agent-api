from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
from rapidfuzz import fuzz
import re
import json
from dotenv import load_dotenv
import os
from sentence_transformers import SentenceTransformer, util
import faiss
import numpy as np
import logging
import time
import random  
import re
import string

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize FastAPI app
app = FastAPI()
load_dotenv()

# Load data from JSON files
try:
    with open("intents.json", "r") as f:
        intents = json.load(f)
    with open("responses.json", "r") as f:
        responses = json.load(f)
    with open("knowledge_base.json", "r") as f:
        knowledge_base = json.load(f)
except Exception as e:
    logging.error(f"Error loading JSON files: {e}")
    raise ValueError("Failed to load required JSON files.")

# Define request model
class ChatRequest(BaseModel):
    message: str
    user_id: str

# Load the embedding model
try:
    embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', use_auth_token=os.getenv("HF_API_TOKEN"))
except Exception as e:
    logging.error(f"Error loading embedding model: {e}")
    raise ValueError("Failed to load the embedding model.")

# Initialize FAISS index
def initialize_faiss_index():
    try:
        questions = list(knowledge_base.keys())
        if not questions:
            raise ValueError("Knowledge base is empty. Cannot initialize FAISS index.")
        embeddings = embedding_model.encode(questions)
        embeddings = np.array(embeddings).astype('float32')
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        return index, questions
    except Exception as e:
        logging.error(f"Error initializing FAISS index: {e}")
        raise

faiss_index, faiss_questions = initialize_faiss_index()

# Session management
sessions = {}

# Cleanup inactive sessions
def cleanup_inactive_sessions():
    current_time = time.time()
    for user_id, session in list(sessions.items()):
        if "last_active" in session and current_time - session["last_active"] > 600:  # 10 minutes
            del sessions[user_id]
            logging.info(f"Session for user {user_id} expired and was removed.")

def update_session_activity(user_id):
    if user_id in sessions:
        sessions[user_id]["last_active"] = time.time()


def classify_intent(user_input):
    best_match, highest_score = None, 0
    user_input = user_input.lower()
    for intent, keywords in intents.items():
        for keyword in keywords:
            score = fuzz.token_sort_ratio(user_input, keyword)
            logging.info(f"Intent: {intent}, Keyword: {keyword}, Score: {score}")
            if score > highest_score:
                highest_score, best_match = score, intent
    return best_match if highest_score > 70 else "fallback"

# Retrieve from knowledge base
def retrieve_from_knowledge_base(query):
    query_embedding = np.array([embedding_model.encode(query.lower())]).astype('float32')
    distances, indices = faiss_index.search(query_embedding, k=1)
    logging.info(f"FAISS Search - Distances: {distances}, Indices: {indices}")
    if distances[0][0] < 1.0 and indices[0][0] < len(faiss_questions):
        return knowledge_base.get(faiss_questions[indices[0][0]], None)
    return None

# Generate response via API
def generate_response_api(input_text, max_length=100):
  
    HF_API_TOKEN, MODEL_ID = os.getenv("HF_API_TOKEN"), os.getenv("MODEL_ENDPOINT")
    
    if not HF_API_TOKEN or not MODEL_ID:
        logging.error("Missing required environment variables: HF_API_TOKEN or MODEL_ENDPOINT")
        return "I encountered an error while generating a response."
    
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    payload = {"inputs": input_text, "parameters": {"max_length": max_length, "num_return_sequences": 1}}
    
    try:
        response = requests.post(f"https://api-inference.huggingface.co/models/{MODEL_ID}", headers=headers, json=payload)
        result = response.json()
        
        if "error" in result:
            logging.error(f"Hugging Face API error: {result['error']}")
            return "An error occurred while generating a response."
        
        generated_text = result[0].get("generated_text", "").strip()
        extracted_response = re.sub(r"---|###|####", "", generated_text).strip()
        return extracted_response if extracted_response else "I couldn't generate a suitable response."
    except Exception as e:
        logging.error(f"Hugging Face API error: {e}")
        return "An error occurred while generating a response."

# AI agent logic
def ai_agent(user_id, user_input):
    try:
        cleanup_inactive_sessions()
        user_input_lower = user_input.lower().strip()

        # Handle "stop" or "cancel" commands globally
        if user_input_lower in ["stop", "cancel"]:
            if user_id in sessions:
                del sessions[user_id]  # Remove the session
                return "Session canceled. Type 'help' if you need assistance."
            else:
                return "No active session to cancel."

        # Detect specific bill types directly in the input
        bill_types = ["electricity", "water", "internet"]
        detected_bill_type = None
        for bill_type in bill_types:
            if bill_type in user_input_lower:
                detected_bill_type = bill_type
                break

        # If a specific bill type is detected, start the session directly
        if detected_bill_type:
            sessions[user_id] = {
                "state": "bill_number",
                "bill_type": detected_bill_type,
                "last_active": time.time()
            }
            return f"Please enter your {detected_bill_type} bill number."

        # Handle session-based interactions (Bill Payment Flow)
        if user_id in sessions:
            update_session_activity(user_id)
            session = sessions[user_id]
            state = session.get("state")

            if state == "bill_type":
                bill_type = user_input_lower
                if bill_type in bill_types:
                    session["bill_type"], session["state"] = bill_type, "bill_number"
                    return f"Please enter your {bill_type} bill number."
                return "Invalid bill type. Type 'stop' to cancel or choose from electricity, water, or internet."

            elif state == "bill_number":
                user_input_cleaned = user_input_lower.replace(" ", "")
                if user_input_cleaned.isdigit():  # Ensuring it's a valid bill number
                    # Simulate an amount in PKR
                    amount_pkr = random.randint(500, 5000)  # Random amount between 500 and 5000 PKR
                    session["bill_number"], session["amount_pkr"], session["state"] = user_input_lower, amount_pkr, "payment_confirmation"
                    return f"Confirm payment of {amount_pkr} PKR for bill number {user_input_cleaned}? (yes/no)"
                return "Invalid bill number. Type 'stop' to cancel or enter a valid numeric bill number."

            elif state == "payment_confirmation":
                if user_input_lower in ["yes", "y"]:
                    bill_type, bill_number, amount_pkr = session["bill_type"], session["bill_number"], session["amount_pkr"]
                    del sessions[user_id]  # End session after confirmation
                    return f"Payment of {amount_pkr} PKR for {bill_type} bill (Bill No: {bill_number}) has been successfully submitted."
                elif user_input_lower in ["no", "n"]:
                    del sessions[user_id]  # Cancel session
                    return "Payment canceled."
                return "Invalid response. Type 'stop' to cancel or confirm with 'yes' or 'no'."

        # Detect if user wants to start a bill payment session
        if any(keyword in user_input_lower for keyword in ["pay bill", "i want to pay my bill", "pay my bill", "pay my bills"]):
            sessions[user_id] = {"state": "bill_type", "last_active": time.time()}
            return "What type of bill would you like to pay? (electricity, water, internet)? Type 'stop' to cancel."

        # Classify intent
        intent = classify_intent(user_input)
        if intent == "pay_bills":
            sessions[user_id] = {"state": "bill_type", "last_active": time.time()}
            return "What type of bill would you like to pay? (electricity, water, internet)? Type 'stop' to cancel."

        if intent != "fallback":
            return responses.get(intent, "I'm sorry, I didn't understand that.")

        # Knowledge Base Retrieval
        kb_response = retrieve_from_knowledge_base(user_input)
        if kb_response:
            return kb_response

        # AI-generated response (fallback)
        return generate_response_api(user_input)

    except Exception as e:
        logging.error(f"Error processing input: {e}")
        return "An error occurred. Please try again later."
    
# FastAPI endpoints
@app.post("/chat")
async def chat(request: ChatRequest):
    user_id = request.user_id.strip()
    user_input = request.message.strip()
    
    if not user_input or not user_id:
        raise HTTPException(status_code=400, detail="Invalid message or user ID.")
    
    if not user_id.isalnum():
        raise HTTPException(status_code=400, detail="User ID must be alphanumeric.")
    
    try:
        response = ai_agent(user_id, user_input)
        return {"response": response}
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return {"response": "Sorry, something went wrong. Please try again later."}

@app.get("/")
async def root():
    return {"message": "Welcome to the AI Fintech Agent API"}