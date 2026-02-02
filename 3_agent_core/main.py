import asyncio
import os
import sys
import json
import time
import torch
from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, END

import pydantic
import pydantic.v1
sys.modules['pydantic.env_settings'] = pydantic.v1

from transformers import AutoTokenizer, AutoModelForTokenClassification
from sentence_transformers import SentenceTransformer
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# ==========================================
# 1. CONFIGURATION
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "chroma_db_local")
BIOBERT_MODEL = "Shreevatsa01/sentinel-v3-final"
EMBEDDING_MODEL = "all-MiniLM-L6-v2" 

KNOWN_DRUGS = [
    "Ozempic", "Wegovy", "Mounjaro", "Zepbound", "Trulicity", 
    "Victoza", "Rybelsus", "Jardiance", "Farxiga", "Januvia", 
    "Eliquis", "Xarelto", "Entresto", "Humira", "Keytruda", 
    "Opdivo", "Enbrel", "Stelara", "Biktarvy", "Dupixent"
]

MEDDRA_TARGETS = [
    "Vomiting", "Nausea", "Dizziness", "Headache", "Severe Pain", "Fatigue", "Insomnia",
    "Abdominal Pain", "Diarrhea", "Constipation", "Rash", "Pruritus",
    "Dyspnea", "Palpitations", "Anxiety", "Depression", "Vertigo",
    "Tremor", "Alopecia", "Hyperhidrosis", "Pyrexia", "Myalgia",
    "Blurred Vision", "Tinnitus", "Dry Mouth", "Weight Loss", "Allergic Reaction",
    "Testicular Pain", "Erectile Dysfunction", "Libido Decreased",
    "Skin Discoloration", "Cyanosis", "Erythema", "Ecchymosis",
    "Suicidal Ideation", "Aggression", "Confusion",
    "Chest Pain", "Edema", "Chills"
]

# ==========================================
# 2. LOAD MODELS
# ==========================================
print("\n⏳ Loading BioBERT...")
tokenizer = AutoTokenizer.from_pretrained(BIOBERT_MODEL)
model = AutoModelForTokenClassification.from_pretrained(BIOBERT_MODEL)
model.to("cpu")
print("   ✅ BioBERT Ready.")

print(f"⏳ Loading Semantic Brain ({EMBEDDING_MODEL})...")
embedding_model_cpu = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
print(f"   ✅ Vector Space Ready.")

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
def log_for_ragas(state: dict, final_report: str):
    log_entry = {
        "user_input": state['post_text'],
        "retrieved_contexts": state['rag_evidence'],
        "generated_answer": final_report,
        "timestamp": time.time()
    }
    with open("ragas_logs.jsonl", "a", encoding='utf-8') as f:
        f.write(json.dumps(log_entry) + "\n")

def manual_predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad(): outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=2)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    
    found_symptoms = []
    current_entity_words = []
    
    for token, prediction in zip(tokens, predictions[0].numpy()):
        label = model.config.id2label[prediction]
        if label == "O":
            if current_entity_words:
                found_symptoms.append(" ".join(current_entity_words))
                current_entity_words = []
            continue
        clean_token = token
        if token.startswith("##"):
            clean_token = token[2:]
            if current_entity_words:
                current_entity_words[-1] += clean_token 
        else:
            if label.startswith("B-"):
                if current_entity_words:
                    found_symptoms.append(" ".join(current_entity_words))
                    current_entity_words = []
                current_entity_words.append(clean_token)
            elif label.startswith("I-"):
                current_entity_words.append(clean_token)

    if current_entity_words:
        found_symptoms.append(" ".join(current_entity_words))
    return list(set(found_symptoms)), [] 

# ==========================================
# 4. AGENT NODES
# ==========================================
class AgentState(TypedDict):
    post_text: str
    target_drug: str
    detected_entities: List[str]
    symptom_map: Dict[str, str]
    standardized_symptoms: List[str]
    rag_evidence: List[str]
    final_report: str

def detector_node(state: AgentState):
    print(f"\n🔍 [Detector]: Scanning text...")
    symptoms, _ = manual_predict(state['post_text'])
    current_target = state.get('target_drug', 'Unknown')
    text_lower = state['post_text'].lower()
    for drug in KNOWN_DRUGS:
        if drug.lower() in text_lower:
            current_target = drug
            break
    
    print(f"   ✅ Identified: {current_target} | Symptoms: {symptoms}")
    return {"detected_entities": symptoms, "target_drug": current_target}

def mapper_node(state: AgentState):
    raw_symptoms = state.get("detected_entities", [])
    if not raw_symptoms: 
        return {"symptom_map": {}, "standardized_symptoms": []}
    
    print(f"🔄 [Mapper]: Normalizing {len(raw_symptoms)} terms...")
    llm = ChatOllama(model="llama3.2:1b", temperature=0.0)
    
    prompt = ChatPromptTemplate.from_template("""
    You are a strict data converter. 
    Task: Map the input terms to the standard list.
    
    Standard List:
    {meddra_list}
    
    Input Terms:
    {user_terms}
    
    INSTRUCTIONS:
    1. Output VALID JSON ONLY.
    2. NO conversational text.
    3. Map ONLY the Input Terms.
    
    Example Input: ["puking", "spinning"]
    Example JSON: {{ "puking": "Vomiting", "spinning": "Dizziness" }}
    """)
    
    chain = prompt | llm | StrOutputParser()
    
    try:
        raw_output = chain.invoke({
            "meddra_list": ", ".join(MEDDRA_TARGETS),
            "user_terms": json.dumps(raw_symptoms)
        })
        
        # Surgical Extraction of JSON
        start_index = raw_output.find('{')
        end_index = raw_output.rfind('}')
        
        if start_index != -1 and end_index != -1:
            clean_json = raw_output[start_index : end_index + 1]
            mapping_result = json.loads(clean_json)
            standardized = list(set(mapping_result.values()))
            print(f"   ✅ Mapped: {mapping_result}")
            return {"symptom_map": mapping_result, "standardized_symptoms": standardized}
        else:
            raise ValueError("No JSON found")
            
    except Exception:
        print(f"   ⚠️ Mapper failed, using raw terms.")
        return {"symptom_map": {k: k for k in raw_symptoms}, "standardized_symptoms": raw_symptoms}

async def investigator_node(state: AgentState):
    terms = state.get("standardized_symptoms", [])
    target = state.get("target_drug")
    if not terms: return {"rag_evidence": []}
    
    print(f"🕵️ [Investigator]: Querying FDA Database for {len(terms)} symptoms...")
    server_script = os.path.join(BASE_DIR, "fda_mcp_server.py")
    server_params = StdioServerParameters(command=sys.executable, args=[server_script], env=None)
    
    collected_evidence = []
    
    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                
                # LOOP THROUGH ALL SYMPTOMS (The Fix)
                for symptom in terms:
                    print(f"   [DEBUG] Searching for: {symptom}")
                    result = await session.call_tool(
                        "search_fda_labels", 
                        arguments={"drug_name": target, "symptom": symptom}
                    )
                    
                    if hasattr(result, 'content') and isinstance(result.content, list):
                        for item in result.content:
                            if hasattr(item, 'text'):
                                collected_evidence.append(f"[{symptom} Match]: {item.text}")
                    else:
                        collected_evidence.append(str(result))
                
                # De-duplicate to keep the prompt clean
                unique_evidence = list(set(collected_evidence))
                return {"rag_evidence": unique_evidence[:5]} # Limit to top 5 to fit in context

    except Exception as e:
        print(f"   ⚠️ MCP Connection Error: {e}")
        return {"rag_evidence": [f"Error: {str(e)}"]}
    
def analyst_node(state: AgentState):
    print(f"🧠 [Analyst]: Generating Clinical Report...")
    llm = ChatOllama(model="llama3.2:1b", temperature=0.1) 
    
    # --- UPGRADED PROMPT FOR BETTER REPORTING ---
    prompt = ChatPromptTemplate.from_template("""
    You are Sentinel AI, a Clinical Safety Analyst.
    
    INPUT DATA:
    - Drug: {drug}
    - Patient Narrative: "{post}"
    - Detected Symptoms: {mapping_str}
    - FDA Database Matches: {evidence_str}
    
    TASK: Write a professional Clinical Safety Report.
    
    OUTPUT FORMAT (Markdown):
    
    ### SENTINEL SAFETY SIGNAL
    
    **1. SIGNAL DETECTION**
    > "{post}"
    
    **2. SYMPTOM ANALYSIS**
    *Table of identified terms:*
    | Patient Term | MedDRA Term |
    |--------------|-------------|
    (Insert rows here based on 'Detected Symptoms')
    
    **3. FDA LABEL CORRELATION**
    *Evidence Summary:*
    (Summarize the FDA Database Matches in 1-2 clear sentences. If no data, state "No explicit match found in current label data.")
    
    **4. CLINICAL ASSESSMENT**
    (Provide a 2-3 sentence conclusion. Do NOT just say "Yes". Explain: "The reported symptoms [X, Y] are consistent with known side effects listed in the FDA label..." OR "The reported symptoms appear to be novel/unlisted...")
    """)
    
    # Format the mapping table rows for the prompt
    map_rows = "\n".join([f"| {k} | {v} |" for k, v in state['symptom_map'].items()])
    if not map_rows: map_rows = "| None | None |"
    
    ev_str = "\n".join([f"- {e[:200]}..." for e in state['rag_evidence']])
    
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({
        "drug": state['target_drug'],
        "post": state['post_text'],
        "mapping_str": map_rows,
        "evidence_str": ev_str
    })
    
    log_for_ragas(state, response)
    return {"final_report": response}

# 5. BUILD GRAPH
workflow = StateGraph(AgentState)
workflow.add_node("detector", detector_node)
workflow.add_node("mapper", mapper_node)
workflow.add_node("investigator", investigator_node)
workflow.add_node("analyst", analyst_node)

workflow.set_entry_point("detector")
workflow.add_edge("detector", "mapper")
workflow.add_edge("mapper", "investigator")
workflow.add_edge("investigator", "analyst")
workflow.add_edge("analyst", END)

app = workflow.compile()

if __name__ == "__main__":
    print(f"🚀 Sentinel V6 (Clean Production Build) Online")