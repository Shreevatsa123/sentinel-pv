import streamlit as st
import requests
import json
import subprocess
import sys
import time
from datetime import datetime
import pandas as pd

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="SENTINEL-PV", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# CLEAN & SIMPLE CSS (System Default Backgrounds)
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
    }
    h1, h2, h3 {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        font-weight: 600;
        color: #2c3e50;
    }
    .stButton>button {
        background-color: #2c3e50;
        color: white;
        border-radius: 4px;
        height: 3em;
        font-weight: bold;
    }
    
    /* 1. SIMPLE REPORT STYLE (Transparent Background) */
    .report-box {
        border: 1px solid #cccccc;
        padding: 20px;
        border-radius: 5px;
        margin-bottom: 20px;
        font-family: 'Helvetica Neue', sans-serif;
        background-color: transparent; /* Forces system default */
    }
    
    /* 2. PROJECT DESCRIPTION BOX */
    .desc-box {
        border: 1px solid #e0e0e0;
        padding: 20px;
        border-radius: 8px;
        margin-bottom: 20px;
    }
    
    /* 3. RL DESCRIPTION */
    .rl-desc {
        border-left: 4px solid #2c3e50;
        padding-left: 15px;
        margin-bottom: 15px;
        color: #555;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. CONFIGURATION & DATA
# ==========================================
API_URL = "http://localhost:8000"

DRUG_LIST = [
    "Ozempic", "Wegovy", "Mounjaro", "Zepbound", "Trulicity", 
    "Victoza", "Rybelsus", "Jardiance", "Farxiga", "Januvia", 
    "Eliquis", "Xarelto", "Entresto", "Humira", "Keytruda", 
    "Opdivo", "Enbrel", "Stelara", "Biktarvy", "Dupixent"
]

DRUG_DESCRIPTIONS = {
    "Ozempic": "Semaglutide injection for type 2 diabetes.",
    "Wegovy": "Semaglutide injection for chronic weight management.",
    "Mounjaro": "Tirzepatide injection for type 2 diabetes.",
    "Humira": "Adalimumab for arthritis and Crohn's disease.",
    "Dupixent": "Dupilumab for eczema and asthma.",
    "Enbrel": "Etanercept for autoimmune diseases.",
    "default": "FDA-approved therapeutic agent monitored for adverse events."
}

# ==========================================
# 3. BACKEND CONNECTION
# ==========================================
@st.cache_resource
def ensure_api_running():
    try:
        requests.get(f"{API_URL}/", timeout=1)
        return True
    except:
        subprocess.Popen(
            [sys.executable, "api_service.py"], 
            cwd="../3_agent_core" 
        )
        time.sleep(5)
        return False

ensure_api_running()

# ==========================================
# 4. SIDEBAR CONFIGURATION
# ==========================================
with st.sidebar:
    st.title("SENTINEL-PV")
    st.caption("Pharmacovigilance Surveillance System")
    
    st.markdown("### System Status")
    try:
        requests.get(f"{API_URL}/", timeout=0.5)
        st.caption("Status: Connected")
    except:
        st.caption("Status: Disconnected (Check Backend)")

    st.divider()
    
    st.markdown("### Configuration")
    target_drug = st.selectbox(
        "FDA Monitored Therapeutics", 
        DRUG_LIST
    )
    
    desc = DRUG_DESCRIPTIONS.get(target_drug, DRUG_DESCRIPTIONS["default"])
    st.info(desc)

# ==========================================
# 5. MAIN INTRODUCTION
# ==========================================
st.markdown("## Automated Adverse Event Detection")

# DESCRIPTION BOX
st.markdown("""
<div class="desc-box">
    <strong>The Mission:</strong> Modern pharmacovigilance relies on manual reporting, which is slow and often misses the "patient voice" on social platforms. 
    Sentinel is an AI-driven system designed to ingest unstructured patient narratives (like social media posts), 
    detect potential side effects using specialized Medical Entity Recognition (NER), and map them to standardized MedDRA terminology.
    <br><br>
    <strong>How it works:</strong> This dashboard simulates the ingestion pipeline. When you submit a report, a multi-agent system (Detector, Mapper, Investigator, Analyst) 
    processes the text to generate a clinical safety signal report.
</div>
""", unsafe_allow_html=True)

# ==========================================
# 6. INPUT SECTION
# ==========================================
col1, col2 = st.columns([3, 1])

with col1:
    default_text = f"Started {target_drug} yesterday. Experienced severe nausea and dizziness throughout the night."
    input_text = st.text_area("Patient Narrative / Raw Data Stream", value=default_text, height=200)

with col2:
    st.write("##") # Spacing
    analyze_btn = st.button("EXECUTE SURVEILLANCE PROTOCOL", use_container_width=True)

# ==========================================
# 7. EXECUTION & LOGGING
# ==========================================
if "result" not in st.session_state: st.session_state.result = None

if analyze_btn:
    with st.status("Initializing Sentinel Agents...", expanded=True) as status:
        st.write("System: Importing core libraries...")
        time.sleep(0.5)
        st.write("System: Loading BioBERT Medical Entity Recognition weights...")
        time.sleep(0.5)
        
        st.write("Agent [Detector]: Scanning text for physiological anomalies...")
        time.sleep(0.5)
        
        st.write("Agent [Mapper]: Vectorizing entities to MedDRA Ontology...")
        
        try:
            res = requests.post(f"{API_URL}/analyze", json={"text": input_text, "drug_name": target_drug})
            
            st.write("Agent [Investigator]: Querying FDA Label Database (RAG)...")
            time.sleep(0.3)
            
            st.write("Agent [Analyst]: Synthesizing clinical safety report...")
            
            if res.status_code == 200:
                st.session_state.result = res.json()
                st.session_state.last_text = input_text
                status.update(label="Analysis Complete", state="complete", expanded=False)
            else:
                status.update(label="Analysis Failed", state="error")
                st.error("Backend Error: Could not process request.")
        except Exception as e:
            status.update(label="Connection Failed", state="error")
            st.error(f"Connection Error: {e}")

# ==========================================
# 8. REPORT DISPLAY
# ==========================================
if st.session_state.result:
    data = st.session_state.result
    
    # --- CLEANING THE TEXT ---
    # Removes specifically the "### Sentinel Alert 🚨" header if the backend sends it
    report_text = data["report"].replace("### Sentinel Alert 🚨", "").replace("###", "").strip()
    
    st.divider()
    r_col1, r_col2 = st.columns([2, 1])
    
    with r_col1:
        # SIMPLE REPORT BOX (Transparent)
        st.markdown(f"""
        <div class="report-box">
            {report_text}
        </div>
        """, unsafe_allow_html=True)
        
    with r_col2:
        st.subheader("Ontology Mapping")
        mapping = data.get("symptom_map", {})
        if mapping:
            df = pd.DataFrame(list(mapping.items()), columns=["Raw Term", "MedDRA Term"])
            st.dataframe(df, hide_index=True, use_container_width=True)
        else:
            st.markdown("*No specific symptoms mapped.*")

    # ==========================================
    # 9. RLHF FEEDBACK LOOP
    # ==========================================
    st.divider()
    st.subheader("Reinforcement Learning (RLHF) Calibration")
    
    # SIMPLE RL DESCRIPTION
    st.markdown("""
    <div class="rl-desc">
    <strong>Why this matters:</strong> This system uses Reinforcement Learning from Human Feedback. 
    By auditing the agents' output below, you are directly modifying the model's reward function. 
    Your corrections help the system distinguish between colloquial slang and genuine medical symptoms, 
    improving patient safety monitoring over time.
    </div>
    """, unsafe_allow_html=True)
    
    with st.form("rl_form"):
        c1, c2 = st.columns(2)
        with c1: 
            caught = st.text_input("Validation: What did the model correctly identify?")
        with c2: 
            missed = st.text_input("Correction: What did the model miss?")
        
        rating = st.slider("Quality Score (Reward Function)", 0, 100, 80)
        
        if st.form_submit_button("SUBMIT AUDIT DATA"):
            try:
                requests.post(f"{API_URL}/feedback", json={
                    "text": st.session_state.last_text,
                    "drug_name": target_drug,
                    "predicted_symptoms": data.get("detected_entities", []),
                    "what_caught": caught,
                    "what_missed": missed,
                    "user_rating": rating,
                    "timestamp": str(datetime.now())
                })
                st.success("Audit data successfully ingested into RL training set.")
            except: 
                st.error("Failed to submit audit data.")