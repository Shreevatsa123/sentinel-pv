import sys
from fastapi import FastAPI
from pydantic import BaseModel
from main import app as agent_graph # Import the graph we already built

# --- 2026 COMPATIBILITY PATCH ---
import pydantic
import pydantic.v1
sys.modules['pydantic.env_settings'] = pydantic.v1
# ---------------------------------

app = FastAPI()

class AnalysisRequest(BaseModel):
    text: str
    drug_name: str # <--- ADD THIS LINEr

@app.get("/")
def health_check():
    return {"status": "Sentinel Agent Service is Online"}

@app.post("/analyze")
def run_analysis(request: AnalysisRequest):
    print(f"📡 Received request for {request.drug_name}: {request.text}")
    
    # Pass the drug_name into the state so main.py can use it for RAG filtering
    result = agent_graph.invoke({
        "post_text": request.text, 
        "target_drug": request.drug_name
    })
    return {"report": result["final_report"]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)