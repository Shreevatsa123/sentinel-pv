import streamlit as st
import json
import os
import time

# CONFIGURATION
FEEDBACK_FILE = "rl_feedback_log.jsonl"
GOLD_FILE = "gold_standard.jsonl"

st.set_page_config(page_title="Sentinel Admin", page_icon="👨‍⚖️", layout="wide")

# HELPER FUNCTIONS
def load_feedback():
    data = []
    if os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, "r") as f:
            for line in f:
                try:
                    data.append(json.loads(line))
                except: continue
    return data

def save_feedback(entries):
    """Rewrites the feedback file with remaining entries."""
    with open(FEEDBACK_FILE, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

def append_to_gold(entries):
    """Appends approved entries to Gold Standard."""
    with open(GOLD_FILE, "a") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

# MAIN INTERFACE
st.title("👨‍⚖️ Data Mediation Console")
st.markdown("Review raw feedback. **Approve** good data for training, **Reject** spam/errors.")

# Load Data
if "logs" not in st.session_state:
    st.session_state.logs = load_feedback()

logs = st.session_state.logs

# SIDEBAR STATS
gold_count = 0
if os.path.exists(GOLD_FILE):
    gold_count = sum(1 for line in open(GOLD_FILE))

with st.sidebar:
    st.header("🏆 Gold Standard")
    st.metric("Total Samples", gold_count)
    if os.path.exists(GOLD_FILE):
        with open(GOLD_FILE, "rb") as f:
            st.download_button("⬇️ Download Training Data", f, file_name="gold_standard.jsonl")
    
    st.divider()
    if st.button("🔄 Refresh Logs"):
        st.session_state.logs = load_feedback()
        st.rerun()

# REVIEW QUEUE
if not logs:
    st.success("✅ Review Queue is Empty.")
else:
    st.info(f"⚡ {len(logs)} Pending Reviews found.")
    
    # We use a form so we can process all decisions at once
    with st.form("mediation_form"):
        decisions = {}
        
        for i, entry in enumerate(logs):
            rating = entry.get('user_rating', entry.get('rating', 0))
            drug = entry.get('drug_name', 'Unknown')
            
            st.markdown(f"### Review #{i+1}: **{drug}** (Grade: {rating}/100)")
            
            # 1. RESIZABLE TEXT AREA (Your Request)
            # Users can drag the bottom-right corner to resize this individually
            st.text_area(
                label="Raw Text",
                value=entry.get('text', ''),
                height=100, 
                key=f"text_{i}"
            )
            
            c1, c2, c3 = st.columns([1, 1, 2])
            with c1:
                st.success(f"✅ Caught: {entry.get('what_caught', '-')}")
            with c2:
                st.error(f"❌ Missed: {entry.get('what_missed', '-')}")
            
            # 2. INDIVIDUAL SELECTION (Your Request)
            with c3:
                decisions[i] = st.radio(
                    "Decision:",
                    ["Skip", "Approve (Gold)", "Reject (Delete)"],
                    key=f"dec_{i}",
                    horizontal=True,
                    index=0 # Default to Skip (Safety)
                )
            
            st.divider()
        
        # 3. EXECUTE BUTTON
        submitted = st.form_submit_button("🚀 EXECUTE DECISIONS", type="primary")
        
        if submitted:
            to_approve = []
            to_keep = []
            
            processed_count = 0
            
            for i, entry in enumerate(logs):
                decision = decisions[i]
                
                if decision == "Approve (Gold)":
                    to_approve.append(entry)
                    processed_count += 1
                elif decision == "Reject (Delete)":
                    processed_count += 1
                    # Do not append to anything, effectively deleting it
                else:
                    # Keep "Skip" entries in the queue
                    to_keep.append(entry)
            
            # Perform File Operations
            if to_approve:
                append_to_gold(to_approve)
            
            save_feedback(to_keep)
            
            # Update Session State
            st.session_state.logs = to_keep
            
            st.success(f"Processed {processed_count} reviews. ({len(to_approve)} Approved, {processed_count - len(to_approve)} Deleted)")
            time.sleep(1)
            st.rerun()