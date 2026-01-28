import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from huggingface_hub import InferenceClient

app = FastAPI()

MODEL_ID = "Shreevatsa01/sentinel-biobert-hybrid"
HF_TOKEN = os.environ.get("HF_TOKEN")
client = InferenceClient(model=MODEL_ID, token=HF_TOKEN)

class InputData(BaseModel):
    text: str

@app.get("/")
def home():
    return {"status": "Sentinel-PV Agent is Awake"}

@app.post("/analyze")
def analyze_text(data: InputData):
    try:
        # FORCE VISIBILITY:
        # We tell it: "Don't ignore any labels." (ignore_labels=[])
        # We also ask for 'score', 'index', and 'word' explicitly.
        result = client.token_classification(
            data.text, 
            parameters={"ignore_labels": []}
        )
        return {"entities": result}

    except Exception as e:
        print(f"CRASH REPORT: {e}")
        return {"error": str(e)}
