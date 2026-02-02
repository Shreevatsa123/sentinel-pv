from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import json
import os
from datetime import datetime

# Import the Brain (The graph we built in main.py)
from main import app as agent_graph

app = FastAPI(title="Sentinel PV API", version="6.0")

# --- DATA MODELS ---
class AnalysisRequest(BaseModel):
    text: str
    drug_name: str

class FeedbackRequest(BaseModel):
    text: str
    drug_name: str
    predicted_symptoms: list
    what_caught: str
    what_missed: str
    user_rating: int
    timestamp: str

# --- ENDPOINTS ---

@app.get("/")
def health_check():
    return {"status": "Sentinel V6 Online", "mode": "Async/MCP"}

@app.post("/analyze")
async def run_analysis(request: AnalysisRequest):
    """
    Runs the full Sentinel Agent Graph (Detector -> Mapper -> Investigator -> Analyst).
    Now fully ASYNCHRONOUS to support MCP.
    """
    try:
        # Prepare the input state for the Brain
        initial_state = {
            "post_text": request.text,
            "target_drug": request.drug_name,
            "detected_entities": [],
            "symptom_map": {},
            "rag_evidence": [],
            "final_report": ""
        }
        
        # KEY FIX: Use 'ainvoke' (Async Invoke) instead of 'invoke'
        # This allows the Investigator node to run its async MCP connection code
        result = await agent_graph.ainvoke(initial_state)
        
        return {
            "report": result["final_report"],
            "detected_entities": result["detected_entities"],
            "symptom_map": result["symptom_map"],
            "status": "success"
        }
        
    except Exception as e:
        print(f"❌ API Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback")
def submit_feedback(data: FeedbackRequest):
    """
    Saves user feedback (RLHF) to a JSONL file.
    This remains synchronous as file I/O is fast enough here.
    """
    try:
        log_entry = data.dict()
        
        # Save to the local file in 3_agent_core
        file_path = "rl_feedback_log.jsonl"
        
        with open(file_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
            
        return {"status": "feedback_received"}
        
    except Exception as e:
        print(f"❌ Feedback Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("api_service:app", host="0.0.0.0", port=8000, reload=True)