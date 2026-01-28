import os
import sys
from typing import TypedDict, List
from langgraph.graph import StateGraph, END

# --- 2026 COMPATIBILITY PATCH ---
import pydantic
from pydantic_settings import BaseSettings
import pydantic.v1
sys.modules['pydantic.env_settings'] = pydantic.v1
# ---------------------------------

from transformers import pipeline
from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# 1. CONFIGURATION
DB_PATH = "./chroma_db_storage"
BIOBERT_MODEL = "Shreevatsa01/biobert-v2-hybrid-slang" 

# 2. DEFINE THE STATE
class AgentState(TypedDict):
    post_text: str
    detected_entities: List[str]
    rag_evidence: List[str]
    final_report: str

# 3. DEFINE THE NODES
def detector_node(state: AgentState):
    print(f"\n🔍 [Detector]: Scanning text for symptoms...")
    ner_pipeline = pipeline("token-classification", model=BIOBERT_MODEL, aggregation_strategy="simple")
    results = ner_pipeline(state['post_text'])
    symptoms = [res['word'] for res in results]
    print(f"   ✅ Found: {symptoms}")
    return {"detected_entities": symptoms}

def investigator_node(state: AgentState):
    symptoms = state.get("detected_entities", [])
    if not symptoms: return {"rag_evidence": []}
    
    print(f"🕵️ [Investigator]: Searching FDA Knowledge Base for {symptoms}...")
    embedding_fn = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embedding_fn, collection_name="fda_drug_labels")
    
    evidence = []
    for symptom in symptoms:
        docs = vectorstore.similarity_search(symptom, k=2)
        evidence.extend([doc.page_content for doc in docs])
    return {"rag_evidence": evidence}

def analyst_node(state: AgentState):
    print("🧠 [Analyst]: Generating Report with Llama 3...")
    llm = ChatOllama(model="llama3.2:1b", temperature=0)
    
    prompt = ChatPromptTemplate.from_template("""
    You are a Pharmacovigilance Expert. 
    USER POST: "{post}"
    DETECTED SYMPTOMS: {symptoms}
    FDA EVIDENCE: {evidence}
    
    Determine if these are KNOWN side effects or POTENTIAL NEW SIGNALS.
    """)
    
    chain = prompt | llm
    response = chain.invoke({
        "post": state['post_text'],
        "symptoms": ", ".join(state['detected_entities']),
        "evidence": "\n".join(state['rag_evidence'][:3])
    })
    return {"final_report": response.content}

# 4. BUILD THE GRAPH
workflow = StateGraph(AgentState)
workflow.add_node("detector", detector_node)
workflow.add_node("investigator", investigator_node)
workflow.add_node("analyst", analyst_node)

workflow.set_entry_point("detector")
workflow.add_edge("detector", "investigator")
workflow.add_edge("investigator", "analyst")
workflow.add_edge("analyst", END)

app = workflow.compile()

# 5. EXECUTION
if __name__ == "__main__":
    user_input = "I've been on Ozempic and I'm vomiting and have a weird blurry spot in my eye."
    print(f"🚀 Processing: {user_input}")
    result = app.invoke({"post_text": user_input})
    print("\n" + "="*30 + " FINAL REPORT " + "="*30)
    print(result["final_report"])
    print("="*74)