from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
from rapidfuzz import fuzz
import re
import json

# Initialize FastAPI app
app = FastAPI()

# Load data from JSON files
with open("intents.json", "r") as f:
    intents = json.load(f)

with open("responses.json", "r") as f:
    responses = json.load(f)

with open("knowledge_base.json", "r") as f:
    knowledge_base = json.load(f)

# Define request model
class ChatRequest(BaseModel):
    message: str

# Function to classify intent using fuzzy matching
def classify_intent(user_input):
    user_input = user_input.lower()
    best_match = None
    highest_score = 0
    for intent, keywords in intents.items():
        for keyword in keywords:
            score = fuzz.token_sort_ratio(user_input, keyword)
            if score > highest_score:
                highest_score = score
                best_match = intent
    if highest_score > 70:  # Adjust threshold for fuzzy matching
        return best_match
    return "fallback"

# Function to retrieve from knowledge base using fuzzy matching
def retrieve_from_knowledge_base(query):
    best_match = None
    highest_score = 0
    for key in knowledge_base.keys():
        score = fuzz.token_sort_ratio(query.lower(), key.lower())
        if score > highest_score:
            highest_score = score
            best_match = key
    if highest_score > 70:  # Adjust threshold for fuzzy matching
        return knowledge_base[best_match]
    return None

# Function to generate response using Hugging Face Inference API
def generate_response_api(input_text, max_length=100):
    HF_API_TOKEN = "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"  # Replace with your actual token
    MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"  # Replace with your chosen model
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    api_url = f"https://api-inference.huggingface.co/models/{MODEL_ID}"
    payload = {
        "inputs": input_text,
        "parameters": {"max_length": max_length, "num_return_sequences": 1},
    }

    try:
        response = requests.post(api_url, headers=headers, json=payload)
        result = response.json()

        if "error" in result:
            return "I encountered an error while generating a response."

        generated_text = result[0]["generated_text"].strip()

        # Post-process to remove internal reasoning or extra content
        match = re.search(r"</think>\s*(.+)", generated_text, re.DOTALL)
        if match:
            extracted_response = match.group(1).strip()
        else:
            extracted_response = generated_text

        # Clean up any remaining markers or delimiters
        extracted_response = re.sub(r"---|###|####", "", extracted_response).strip()

        return extracted_response

    except Exception as e:
        return f"An error occurred: {str(e)}"

# Main AI agent function
def ai_agent(user_input):
    try:
        # Step 1: Intent Classification
        intent = classify_intent(user_input)
        if intent != "fallback":
            return responses.get(intent, "fallback")
        # Step 2: Knowledge Base Retrieval
        
        kb_response = retrieve_from_knowledge_base(user_input)
        if kb_response:
            return kb_response
        # Step 3: Generative Response (Fallback) via Hugging Face API
        generated_response = generate_response_api(user_input)
        if generated_response.strip() == "":
            return "I'm sorry, I couldn't find a suitable response for that."
        return generated_response
    except Exception as e:
        return f"An error occurred: {str(e)}"

# FastAPI endpoint
@app.post("/chat")
async def chat(request: ChatRequest):
    user_input = request.message.strip()
    if not user_input:
        raise HTTPException(status_code=400, detail="Please provide a valid message.")
    response = ai_agent(user_input)
    return {"response": response}