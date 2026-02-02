# Sentinel-PV: Autonomous Pharmacovigilance Agent

Sentinel PV is an end-to-end pharmacovigilance project I built to capture Adverse Drug Events (ADEs) hidden in informal patient conversations on social media. While learning about drug safety systems, I noticed that most traditional pipelines struggle with patient-generated text—people describe side effects using slang, typos, and casual language that keyword-based methods simply miss. This project is my attempt to close that gap by turning messy, real-world narratives into structured, medically meaningful safety signals that can be validated against official clinical data.

The data science core of Sentinel PV (Part A) is centered around a BioBERT-based Named Entity Recognition (NER) model that I fine-tuned using a hybrid dataset. I trained the model on the CADEC gold-standard corpus to ground it in proper medical reasoning, and then adapted it further using ~8,000 scraped Reddit comments so it could learn how patients actually talk online. This combination helps the model handle both formal clinical language and highly informal descriptions. I focused on 20 high-profile, high-patient-volume drugs—including GLP-1 agonists like Ozempic and Wegovy, and biologics such as Humira—because these drugs are frequently discussed online and generate a high volume of side-effect reports.

In production (Part B), I designed Sentinel PV as a multi-agent system orchestrated with LangGraph, where each agent has a clear responsibility. A Detector Agent (BioBERT) extracts drugs and symptoms from raw Reddit posts, a Mapper Agent (Llama 3.2) normalizes slang and misspellings into standardized medical terms (e.g., mapping informal phrases to MedDRA concepts), and an Investigator Agent queries official FDA labeling data through the Model Context Protocol (MCP). Finally, an Analyst Agent pulls everything together into a concise, clinical-style safety report. The entire pipeline is automated using n8n, allowing the system to continuously ingest real Reddit data and convert unstructured social chatter into structured, verifiable pharmacovigilance insights.

---

## Table of Contents

1. [Tech Stack](#tech-stack)
2. [Project Architecture Overview](#project-architecture-overview)
3. [Part A: The Data Science Core](#part-a-the-data-science-core-model-training)
4. [Part B: The Intelligent Production System](#part-b-the-intelligent-production-system)
  * [B1: The Agentic Brain (Simulation)](#b1-the-agentic-brain-frontend--backend)
  * [B2: The Data Pipeline (n8n)](#b2-the-data-pipeline-n8n-workflow)
5. [Key Features & Innovations](#key-features--innovations)
6. [Installation & Setup](#installation--setup)

---

## Tech Stack

### **Core AI & Machine Learning**

* **LLMs:** `Llama 3.2 (1B & 3B)` via **Ollama** (Local Inference).
* **NER Model:** Fine-tuned **BioBERT** (`Shreevatsa01/sentinel-v3-final`).
* **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`.
* **Vector Database:** **ChromaDB** (Local persistence).
* **Frameworks:** **PyTorch**, **Hugging Face Transformers**.

### **Agentic Orchestration & Architecture**

* **Orchestration:** **LangGraph** (Stateful Multi-Agent Workflows).
* **Protocol:** **MCP (Model Context Protocol)** by Anthropic (Client/Server Architecture for FDA Data).
* **Evaluation:** **Ragas** (RAG Assessment Framework for Quality Control).
* **API:** **FastAPI** (Asynchronous Backend).

### **Frontend & Data Pipeline**

* **UI/Dashboard:** **Streamlit** (Python-based interactive dashboard).
* **Automation:** **n8n** (Low-code workflow automation for Reddit Scraping).

---

## Project Architecture Overview

The project is divided into two distinct phases:

* **Part A (Research):** The "Lab" where the custom BioBERT model was trained to recognize medical entities.
* **Part B (Production):** The "Factory" where the models are deployed into a live system with Agents, APIs, and a User Interface.

---

## Part A: The Data Science Core (Model Training)

*Located in: `/1_data_science*`

This section focuses on creating the specialized "eyes" of the system—the **Named Entity Recognition (NER)** model.

* **Objective:** Teach a BERT model to read informal social media text and identify drug names (`DRUG`) and adverse events (`AE`).
* **Base Model:** `dmis-lab/biobert-v1.1`.
* **Dataset:** CADEC (CSIRO Adverse Drug Event Corpus) / SMM4H.
* **Process:**
1. Data Preprocessing & Tokenization.
2. Fine-tuning with Hugging Face `Trainer`.
3. Evaluation (F1-Score, Precision, Recall).
4. **Result:** The `sentinel-v3-final` model, which can extract "dizziness", "threw up", and "feeling weird" from text.



---

## Part B: The Intelligent Production System

*Located in: `/3_agent_core` & `/4_frontend*`

This is the fully functional application that orchestrates multiple AI agents to process data in real-time.

### **B1: The Agentic Brain (Frontend + Backend)**

This system uses **LangGraph** to manage a team of 4 specialized AI Agents working in a relay:

#### **1. The Workflow (The "Brain")**

1. **Detector Agent (BioBERT):** Scans raw text to extract entities.
* *Input:* "My head is spinning after taking Ozempic."
* *Output:* `['spinning']`


2. **Mapper Agent (Llama 3.2):** Normalizes slang and typos into standard MedDRA medical terms. Uses a "Surgical JSON Extraction" technique to ensure valid output.
* *Input:* `['spinning', 'hurtz']`
* *Output:* `{'spinning': 'Vertigo', 'hurtz': 'Pain'}`


3. **Investigator Agent (MCP Client):** Connects to a separate **FDA Data Server** via the **Model Context Protocol (MCP)**. It performs a multi-symptom search to find clinical evidence.
* *Action:* Queries `fda_mcp_server.py` for "Vertigo" warnings on the Ozempic label.


4. **Analyst Agent (Llama 3.2 CoT):** The final decision maker. It reviews the User Text, Mapped Symptoms, and FDA Evidence to write a professional Clinical Safety Report.

#### **Architecture Highlights:**

* **Asynchronous API:** The backend (`api_service.py`) handles requests asynchronously to allow non-blocking MCP connections.
* **Quality Control (Ragas):** A separate module (`evaluate_quality.py`) grades the Analyst's reports on **Faithfulness** and **Answer Relevancy**.

### **B2: The Data Pipeline (n8n Workflow)**

To fuel the simulation with real-world data, an **n8n** workflow is used.

* **Function:** Scrapes Reddit comments from subreddits like `r/Ozempic`, `r/Mounjaro`, etc.
* **Trigger:** Scheduled or Manual.
* **Process:**
1. Fetch Reddit API / RSS feeds.
2. Filter for keywords (e.g., "side effect", "sick", "pain").
3. Clean and structure the JSON.
4. Send to the Sentinel API for analysis.



---

## Key Features & Innovations

1. **MCP (Model Context Protocol):**
* Decouples the Database from the Brain. The FDA data lives on a "Server" (`fda_mcp_server.py`), and the Brain connects to it like a tool. This makes the system modular and future-proof.


2. **Self-Healing Mapper:**
* Unlike strict dictionary lookups, the LLM-based Mapper can understand context. It knows that "pain in my balls"  "Testicular Pain" without hard-coding.


3. **Chain of Thought (CoT):**
* The Analyst doesn't just guess; it follows a reasoning path to generate "Clinical Assessments" rather than simple Yes/No answers.


4. **Robust Error Handling:**
* Includes fallback mechanisms for JSON parsing errors (surgical extraction) and connection timeouts.



---

## Installation & Setup

### **Prerequisites**

* Python 3.10+
* Ollama (running `llama3.2:1b`)
* Node.js (for n8n, optional)

### **1. Backend Setup**

```bash
cd 3_agent_core
pip install -r requirements.txt
# Start the API (The Brain)
uvicorn api_service:app --reload --port 8000

```

### **2. Frontend Setup**

```bash
cd 4_frontend
pip install streamlit
# Launch the Dashboard
streamlit run dashboard.py

```

### **3. Evaluation (Optional)**

To grade the AI's performance:

```bash
cd 3_agent_core
python evaluate_quality.py

```

---

*Sentinel PV represents a leap forward in automated drug safety monitoring, combining the precision of BioBERT with the reasoning capabilities of modern LLMs.*

### Configuration (Required)
To keep sensitive details private, this project uses a `terraform.tfvars` file which is excluded from version control. You must create this file manually.

1.  Navigate to `2_infrastructure/terraform/`.
2.  Create a file named `terraform.tfvars`.
3.  Add your specific Google Cloud Project ID inside:

```hcl
project_id = "your-gcp-project-id-here"
```
