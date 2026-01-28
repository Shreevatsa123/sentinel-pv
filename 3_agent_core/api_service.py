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

@app.get("/")
def health_check():
    return {"status": "Sentinel Agent Service is Online"}

@app.post("/analyze")
def run_analysis(request: AnalysisRequest):
    print(f"📡 Received request from n8n: {request.text}")
    # Run the LangGraph agent
    result = agent_graph.invoke({"post_text": request.text})
    return {"report": result["final_report"]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)