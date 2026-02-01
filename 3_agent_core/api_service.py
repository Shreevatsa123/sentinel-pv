import sys
import json
import os
from datetime import datetime
from fastapi import FastAPI
from pydantic import BaseModel
from main import app as agent_graph 

# --- 2026 COMPATIBILITY PATCH ---
import pydantic
import pydantic.v1
sys.modules['pydantic.env_settings'] = pydantic.v1
# ---------------------------------

app = FastAPI()
FEEDBACK_FILE = "../4_frontend/rl_feedback_log.jsonl"

class AnalysisRequest(BaseModel):
    text: str
    drug_name: str 

class FeedbackRequest(BaseModel):
    text: str
    drug_name: str
    predicted_symptoms: list
    user_rating: int
    what_caught: str
    what_missed: str
    timestamp: str

@app.get("/")
def health_check():
    return {"status": "Sentinel Agent Service is Online"}

@app.post("/analyze")
def run_analysis(request: AnalysisRequest):
    print(f"📡 Received request for {request.drug_name}")
    
    result = agent_graph.invoke({
        "post_text": request.text, 
        "target_drug": request.drug_name
    })
    
    return {
        "report": result["final_report"],
        "detected_entities": result.get("detected_entities", []),
        "symptom_map": result.get("symptom_map", {}) # <--- SEND MAPPING TO FRONTEND
    }

@app.post("/feedback")
def save_feedback(request: FeedbackRequest):
    entry = request.dict()
    with open(FEEDBACK_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")
    return {"status": "Feedback Recorded"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)