import os
import sys
import torch
import json
from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, END

# --- 2026 COMPATIBILITY PATCH ---
import pydantic
from pydantic_settings import BaseSettings
import pydantic.v1
sys.modules['pydantic.env_settings'] = pydantic.v1
# ---------------------------------

from transformers import AutoTokenizer, AutoModelForTokenClassification
from sentence_transformers import SentenceTransformer, util
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser

# ==========================================
# 1. CONFIGURATION
# ==========================================
DB_PATH = "./chroma_db_local" 
BIOBERT_MODEL = "Shreevatsa01/sentinel-v3-final"

# UPGRADE: Using a smarter model for "slang understanding"
EMBEDDING_MODEL = "all-mpnet-base-v2" 

KNOWN_DRUGS = [
    "Ozempic", "Wegovy", "Mounjaro", "Zepbound", "Trulicity", 
    "Victoza", "Rybelsus", "Jardiance", "Farxiga", "Januvia", 
    "Eliquis", "Xarelto", "Entresto", "Humira", "Keytruda", 
    "Opdivo", "Enbrel", "Stelara", "Biktarvy", "Dupixent"
]

# TARGET ONTOLOGY (Expanded for better semantic anchor points)
MEDDRA_TARGETS = [
    "Vomiting", "Nausea", "Dizziness", "Headache", "Severe Pain", "Fatigue", "Insomnia",
    "Abdominal Pain", "Diarrhea", "Constipation", "Rash", "Pruritus",
    "Dyspnea", "Palpitations", "Anxiety", "Depression", "Vertigo",
    "Tremor", "Alopecia", "Hyperhidrosis", "Pyrexia", "Myalgia",
    "Blurred Vision", "Tinnitus", "Dry Mouth", "Weight Loss", "Allergic Reaction"
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
target_embeddings = embedding_model_cpu.encode(MEDDRA_TARGETS, convert_to_tensor=True)
print(f"   ✅ Vector Space Ready.")

class ManualEmbeddings:
    def embed_documents(self, texts): return embedding_model_cpu.encode(texts).tolist()
    def embed_query(self, text): return embedding_model_cpu.encode(text).tolist()

# ==========================================
# 3. AGENT NODES
# ==========================================
class AgentState(TypedDict):
    post_text: str
    target_drug: str
    detected_entities: List[str]
    symptom_map: Dict[str, str]
    standardized_symptoms: List[str]
    rag_evidence: List[str]
    final_report: str

def manual_predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad(): outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=2)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    
    found_symptoms = []
    current_entity_words = []
    
    # FIX: Better logic to reconstruct words with spaces
    for token, prediction in zip(tokens, predictions[0].numpy()):
        label = model.config.id2label[prediction]
        
        if label == "O":
            # If we were tracking an entity, save it now
            if current_entity_words:
                found_symptoms.append(" ".join(current_entity_words))
                current_entity_words = []
            continue

        # Handle Subwords (##ing)
        clean_token = token
        if token.startswith("##"):
            clean_token = token[2:]
            if current_entity_words:
                current_entity_words[-1] += clean_token # Attach to previous word
        else:
            # It's a new word token
            if label.startswith("B-"):
                # If we were already tracking, save the previous one
                if current_entity_words:
                    found_symptoms.append(" ".join(current_entity_words))
                    current_entity_words = []
                current_entity_words.append(clean_token)
            elif label.startswith("I-"):
                current_entity_words.append(clean_token)

    # Catch the last one
    if current_entity_words:
        found_symptoms.append(" ".join(current_entity_words))
        
    return list(set(found_symptoms)), [] # We ignore drugs for now to focus on symptoms

def detector_node(state: AgentState):
    print(f"\n🔍 [Detector]: Scanning...")
    symptoms, _ = manual_predict(state['post_text'])
    
    # Priority Context Switching
    current_target = state.get('target_drug', 'Ozempic')
    text_lower = state['post_text'].lower()
    for drug in KNOWN_DRUGS:
        if drug.lower() in text_lower:
            current_target = drug
            break
            
    print(f"   ✅ Detected: {symptoms}")
    return {"detected_entities": symptoms, "target_drug": current_target}

def mapper_node(state: AgentState):
    raw = state.get("detected_entities", [])
    if not raw: return {"symptom_map": {}, "standardized_symptoms": []}
    
    mapping = {}
    print(f"🔄 [Mapper]: Vectorizing {raw}...")
    
    slang_embeddings = embedding_model_cpu.encode(raw, convert_to_tensor=True)
    hits = util.semantic_search(slang_embeddings, target_embeddings, top_k=1)
    
    for i, hit_list in enumerate(hits):
        score = hit_list[0]['score']
        slang_term = raw[i]
        std_term = MEDDRA_TARGETS[hit_list[0]['corpus_id']]
        
        # LOWER THRESHOLD to catch idioms like "killing me"
        if score > 0.4: 
            mapping[slang_term] = std_term
            print(f"   ✅ '{slang_term}' ➝ '{std_term}' ({score:.2f})")
        else:
            print(f"   ⚠️ No match for '{slang_term}' ({score:.2f})")
            mapping[slang_term] = slang_term # Keep raw

    return {"symptom_map": mapping, "standardized_symptoms": list(set(mapping.values()))}

def investigator_node(state: AgentState):
    terms = state.get("standardized_symptoms", [])
    target = state.get("target_drug")
    if not terms: return {"rag_evidence": []}
    
    print(f"🕵️ [Investigator]: Searching for {terms}...")
    if not os.path.exists(DB_PATH): return {"rag_evidence": ["DB Missing"]}
    
    try:
        vec = Chroma(persist_directory=DB_PATH, embedding_function=ManualEmbeddings(), collection_name="fda_drug_labels")
        all_docs = []
        for term in terms:
            # Better Search Query: "Ozempic side effect vomiting" instead of just "vomiting"
            docs = vec.similarity_search(f"{target} side effect {term}", k=2, filter={"drug_name": target})
            all_docs.extend(docs)
            
        unique_evidence = list(set([d.page_content for d in all_docs]))
        return {"rag_evidence": unique_evidence[:3]} # Keep top 3 to reduce noise
    except: return {"rag_evidence": []}

def analyst_node(state: AgentState):
    print(f"🧠 [Analyst]: Drafting Report...")
    llm = ChatOllama(model="llama3.2:1b", temperature=0.1) 
    
    # UPDATED PROMPT: Forces summary instead of raw dump
    prompt = ChatPromptTemplate.from_template("""
    You are Sentinel AI. Analyze this potential Adverse Drug Event.

    INPUT DATA:
    - Drug: {drug}
    - Patient Report: "{post}"
    - Symptom Analysis: {mapping_str}
    - FDA Database Matches: {evidence_str}
    
    TASK: Write a clean Markdown report.
    
    ### 🚨 SENTINEL ALERT: {drug}
    
    **1. SIGNAL EXTRACTED**
    > "{post}"
    
    **2. SYMPTOM MAPPING**
    {mapping_str}
    
    **3. FDA CORRELATION**
    *Summarize the FDA data below into 1-2 sentences. Do not copy-paste.*
    {evidence_str}
    
    **4. VERDICT**
    (Based on FDA data, is this a KNOWN side effect? Yes/No)
    """)
    
    map_str = "\n".join([f"- **{k}** ➝ {v}" for k, v in state['symptom_map'].items()])
    
    # Truncate evidence to prevent huge context
    ev_str = "\n".join([f"- {e[:200]}..." for e in state['rag_evidence']])
    
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({
        "drug": state['target_drug'],
        "post": state['post_text'],
        "mapping_str": map_str,
        "evidence_str": ev_str
    })
    return {"final_report": response}

# 4. BUILD GRAPH
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
    print(f"🚀 Sentinel V5 Online (Smart Mapper: {EMBEDDING_MODEL})")