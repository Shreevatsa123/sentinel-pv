import os
import sys
import torch
from typing import TypedDict, List
from langgraph.graph import StateGraph, END

# --- 2026 COMPATIBILITY PATCH ---
import pydantic
from pydantic_settings import BaseSettings
import pydantic.v1
sys.modules['pydantic.env_settings'] = pydantic.v1
# ---------------------------------

from transformers import AutoTokenizer, AutoModelForTokenClassification
from sentence_transformers import SentenceTransformer
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# ==========================================
# 1. CONFIGURATION
# ==========================================
DB_PATH = "./chroma_db_local" 
BIOBERT_MODEL = "Shreevatsa01/biobert-v2-hybrid-slang" 
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# MASTER DRUG LIST 
KNOWN_DRUGS = [
    "Ozempic", "Wegovy", "Mounjaro", "Zepbound", "Trulicity", 
    "Victoza", "Rybelsus", "Jardiance", "Farxiga", "Januvia", 
    "Eliquis", "Xarelto", "Entresto", "Humira", "Keytruda", 
    "Opdivo", "Enbrel", "Stelara", "Biktarvy", "Dupixent"
]

# ==========================================
# 2. GLOBAL MODEL LOADING (Safe CPU Mode)
# ==========================================
print("\n⏳ 1. Loading BioBERT (Detector)...")
tokenizer = AutoTokenizer.from_pretrained(BIOBERT_MODEL)
model = AutoModelForTokenClassification.from_pretrained(BIOBERT_MODEL)
model.to("cpu")
print("   ✅ BioBERT Ready.")

print("⏳ 2. Loading Embeddings (RAG Reader)...")
embedding_model_cpu = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
print("   ✅ Embeddings Ready.")

class ManualEmbeddings:
    def embed_documents(self, texts):
        return embedding_model_cpu.encode(texts).tolist()
    def embed_query(self, text):
        return embedding_model_cpu.encode(text).tolist()

# ==========================================
# 3. AGENT LOGIC
# ==========================================
class AgentState(TypedDict):
    post_text: str
    target_drug: str
    detected_entities: List[str]
    rag_evidence: List[str]
    final_report: str

def manual_predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=2)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    
    found_symptoms = []
    found_drugs = []
    current_entity = ""
    current_label = None
    
    for token, prediction in zip(tokens, predictions[0].numpy()):
        label = model.config.id2label[prediction]
        if label != "O": 
            clean_token = token[2:] if token.startswith("##") else token
            if label.startswith("B-"):
                if current_entity:
                    if "Chemical" in current_label: found_drugs.append(current_entity)
                    else: found_symptoms.append(current_entity)
                current_entity = clean_token
                current_label = label
            elif label.startswith("I-") and current_entity:
                current_entity += clean_token
        else:
            if current_entity:
                if "Chemical" in current_label: found_drugs.append(current_entity)
                else: found_symptoms.append(current_entity)
                current_entity = ""
                current_label = None
    if current_entity:
        if "Chemical" in current_label: found_drugs.append(current_entity)
        else: found_symptoms.append(current_entity)
        
    return list(set(found_symptoms)), list(set(found_drugs))

# --- NODE 1: DETECTOR ---
def detector_node(state: AgentState):
    print(f"\n🔍 [Detector]: Scanning text...")
    try:
        symptoms, drugs = manual_predict(state['post_text'])
        
        # Start with a safe default
        current_target = state.get('target_drug', 'Ozempic')
        
        # --- SMART CONTEXT SWITCHING ---
        # Instead of picking the first match in the LIST, we pick the first match in the TEXT.
        text_lower = state['post_text'].lower()
        found_matches = []
        
        for drug in KNOWN_DRUGS:
            index = text_lower.find(drug.lower())
            if index != -1:
                # Store (position, drug_name) so we can sort by position
                found_matches.append((index, drug))
        
        if found_matches:
            # Sort by index (earliest mention wins)
            found_matches.sort() 
            current_target = found_matches[0][1]
            print(f"   🎯 Priority Drug Detected: {current_target} (at char {found_matches[0][0]})")
        
        # Fallback: If no known drug keywords found, use BioBERT detected drugs
        elif drugs:
            current_target = drugs[0]

        print(f"   ✅ Context Set to: {current_target}")
        print(f"   ✅ Symptoms: {symptoms}")
        
        return {
            "detected_entities": symptoms,
            "target_drug": current_target 
        }
    except Exception as e:
        print(f"   ❌ Detection Error: {e}")
        return {"detected_entities": []}

# --- NODE 2: INVESTIGATOR (REAL RAG) ---
def investigator_node(state: AgentState):
    symptoms = state.get("detected_entities", [])
    target_drug = state.get("target_drug") 
    
    if not symptoms:
        return {"rag_evidence": []}

    print(f"🕵️ [Investigator]: Searching Knowledge Base for {target_drug}...")
    
    if not os.path.exists(DB_PATH):
        print(f"   ❌ ERROR: Database missing at {DB_PATH}.")
        return {"rag_evidence": ["Error: Database missing."]}

    try:
        embedder = ManualEmbeddings()
        vectorstore = Chroma(
            persist_directory=DB_PATH, 
            embedding_function=embedder, 
            collection_name="fda_drug_labels"
        )
        
        evidence = []
        # Filter strictly by the detected drug
        docs = vectorstore.similarity_search(
            f"{target_drug} side effects: {' '.join(symptoms)}", 
            k=3,
            filter={"drug_name": target_drug} 
        )
        
        for doc in docs:
            source = doc.metadata.get('source', 'Unknown File')
            evidence.append(f"[{source}]: {doc.page_content}")
        
        print(f"   ✅ Found {len(evidence)} records for {target_drug}.")
        return {"rag_evidence": evidence}
    except Exception as e:
        print(f"   ⚠️ RAG Error: {e}")
        return {"rag_evidence": []}

# --- NODE 3: ANALYST ---
def analyst_node(state: AgentState):
    print(f"🧠 [Analyst]: Drafting Report for {state['target_drug']}...")
    llm = ChatOllama(model="llama3.2:1b", temperature=0)
    
    prompt = ChatPromptTemplate.from_template("""
    You are a Pharmacovigilance Expert. Write a CONCISE Slack report.
    
    STRICT FORMATTING:
    - Use single asterisks for bold headers (e.g. *HEADER*).
    - No markdown #. 

    DATA:
    - Target Drug: {drug}
    - Original Post: "{post}"
    - Detected Symptoms: {symptoms}
    - Real FDA Knowledge Base: {evidence}
    
    OUTPUT TEMPLATE:
    
    *ORIGINAL REPORT*
    "{post}"
    
    *DETECTED SYMPTOMS*
    {symptoms}
    
    *FDA EVIDENCE CHECK*
    - *Drug Checked:* {drug}
    - *Findings:* (Summarize the RAG data in 1 sentence. Cite source file.)
    
    *VERDICT*
    (Classify: Known Side Effect vs Potential New Signal)
    
    *RECOMMENDATION*
    (One sentence advice)
    """)
    
    chain = prompt | llm
    response = chain.invoke({
        "drug": state['target_drug'],
        "post": state['post_text'],
        "symptoms": ", ".join(state['detected_entities']),
        "evidence": "\n".join(state['rag_evidence'])
    })
    return {"final_report": response.content}

# 5. BUILD GRAPH
workflow = StateGraph(AgentState)
workflow.add_node("detector", detector_node)
workflow.add_node("investigator", investigator_node)
workflow.add_node("analyst", analyst_node)

workflow.set_entry_point("detector")
workflow.add_edge("detector", "investigator")
workflow.add_edge("investigator", "analyst")
workflow.add_edge("analyst", END)

app = workflow.compile()

# 6. EXECUTION TEST
if __name__ == "__main__":
    print("🚀 Sentinel API Ready (Smart Context Mode)")