import streamlit as st
import requests
import json
import subprocess
import sys
import time
from datetime import datetime
import pandas as pd

# ==========================================
# 1. PAGE CONFIG
# ==========================================
st.set_page_config(page_title="Sentinel Command", page_icon="🛡️", layout="wide")

# ==========================================
# 2. YELLOW-ON-BLACK CSS
# ==========================================
st.markdown("""
<style>
    /* Target the container for the report */
    [data-testid="stMarkdownContainer"] p {
        font-family: 'Courier New', monospace;
    }
    
    /* The Cool Yellow Box Styling */
    .sentinel-box {
        background-color: #000000;
        color: #FFFF00;
        padding: 20px;
        border: 2px solid #FFFF00;
        border-radius: 8px;
        box-shadow: 0 0 15px rgba(255, 255, 0, 0.15);
        font-family: 'Courier New', monospace;
        margin-bottom: 20px;
    }
    
    /* Force headings inside the box to be yellow */
    .sentinel-box h1, .sentinel-box h2, .sentinel-box h3, .sentinel-box strong {
        color: #FFFF00 !important;
    }
    
    /* Blockquotes in Yellow */
    .sentinel-box blockquote {
        color: #e6e600;
        border-left: 2px solid #FFFF00;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. AUTO-START SYSTEM
# ==========================================
API_URL = "http://localhost:8000"

@st.cache_resource
def ensure_api_running():
    try:
        requests.get(f"{API_URL}/", timeout=1)
        return True
    except:
        print("⚠️ Starting Brain...")
        subprocess.Popen([sys.executable, "api_service.py"])
        time.sleep(5)
        return False

ensure_api_running()

# ==========================================
# 4. DASHBOARD UI
# ==========================================
with st.sidebar:
    st.title("🛡️ SENTINEL V4")
    st.caption("Pharmacovigilance Core")
    
    try:
        requests.get(f"{API_URL}/", timeout=0.5)
        st.success("SYSTEM: ONLINE 🟢")
    except:
        st.error("SYSTEM: OFFLINE 🔴")

    st.divider()
    target_drug = st.selectbox("Protocol", [
        "Ozempic", "Wegovy", "Mounjaro", "Zepbound", "Humira", "Dupixent", "Enbrel"
    ])

st.markdown(f"## 🚨 Patient Report Simulation: **{target_drug}**")

col1, col2 = st.columns([3, 1])
with col1:
    ex = f"Started {target_drug} yesterday. Kept barfing all night and felt super dizzy."
    input_text = st.text_area("📄 Raw Stream", value=ex, height=150)
with col2:
    st.write("##")
    analyze_btn = st.button("⚡ ANALYZE", type="primary", use_container_width=True)

if "result" not in st.session_state: st.session_state.result = None

if analyze_btn:
    with st.spinner("🤖 Detector -> Mapper -> Investigator -> Analyst..."):
        try:
            res = requests.post(f"{API_URL}/analyze", json={"text": input_text, "drug_name": target_drug})
            if res.status_code == 200:
                st.session_state.result = res.json()
                st.session_state.last_text = input_text
            else:
                st.error("Analysis Failed")
        except:
            st.error("Connection Error.")

# RESULTS DISPLAY
if st.session_state.result:
    data = st.session_state.result
    
    st.divider()
    r_col1, r_col2 = st.columns([2, 1])
    
    with r_col1:
        st.subheader("📋 Official Sentinel Report")
        
        # --- THE MAGIC VISUAL FIX ---
        # We start the div, render markdown INSIDE it, then close div.
        # Note: We construct the raw HTML string but pass it to markdown with unsafe_allow_html
        
        # 1. Open the Yellow Box
        st.markdown('<div class="sentinel-box">', unsafe_allow_html=True)
        
        # 2. Render the actual Report using Markdown (so **bold** works)
        # We use a trick: standard st.markdown won't sit inside the div easily.
        # Instead, we render the markdown content directly.
        st.markdown(data["report"]) 
        
        # 3. Close the Yellow Box
        st.markdown('</div>', unsafe_allow_html=True)
        
    with r_col2:
        st.subheader("🔄 Normalization")
        mapping = data.get("symptom_map", {})
        if mapping:
            df = pd.DataFrame(list(mapping.items()), columns=["Slang", "MedDRA Term"])
            st.dataframe(df, hide_index=True, use_container_width=True)
        else:
            st.info("No slang detected.")

    st.divider()
    st.subheader("🧠 RL Loop")
    with st.form("rl_form"):
        c1, c2 = st.columns(2)
        with c1: caught = st.text_input("✅ Caught?")
        with c2: missed = st.text_input("❌ Missed?")
        rating = st.slider("Grade", 0, 100, 80)
        if st.form_submit_button("💾 Submit Audit"):
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
                st.success("Audit Sent.")
            except: st.error("Failed.")