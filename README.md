# Sentinel-PV: Autonomous Pharmacovigilance Agent

## Table of Contents

* [1. Project Overview](#1-project-overview)
* [2. Tech Stack](#2-tech-stack)
* [3. Part A: The Data Science Core (Model Training)](#3-part-a-the-data-science-core-model-training)
   * [BioBERT Fine-Tuning Strategy](#biobert-fine-tuning-strategy)
   * [Hybrid Dataset Approach (CADEC + Scraped Reddit Data)](#hybrid-dataset-approach-cadec--scraped-reddit-data)
   * [Model Evaluation Metrics](#model-evaluation-metrics)
* [4. Part B: The Intelligent Production System](#4-part-b-the-intelligent-production-system)
   * [B1: The Agentic Brain (Simulation & Core)](#b1-the-agentic-brain-simulation--core)
      * [Multi-Agent Architecture (LangGraph)](#multi-agent-architecture-langgraph)
      * [The Agent Workflow: Detector → Mapper → Investigator → Analyst](#the-agent-workflow-detector--mapper--investigator--analyst)
      * [FDA Data Verification via Model Context Protocol (MCP)](#fda-data-verification-via-model-context-protocol-mcp)
      * [Interactive Dashboard (Streamlit)](#interactive-dashboard-streamlit)
   * [B2: The Data Pipeline (Automation)](#b2-the-data-pipeline-automation)
      * [n8n Workflow Integration](#n8n-workflow-integration)
      * [Real-Time Reddit Scraping & Ingestion](#real-time-reddit-scraping--ingestion)
* [5. Code structure](#5-quality-assurance--evaluation-ragas)
* [6. Quality Assurance & Evaluation (Ragas)](#6-quality-assurance--evaluation-ragas)
* [7. Infrastructure & Deployment (Terraform)](#7-infrastructure--deployment-terraform)
* [8. Installation & Setup](#8-installation--setup)
   * [Prerequisites](#prerequisites)
   * [Backend Setup](#backend-setup)
   * [Frontend Setup](#frontend-setup)
* [9. Usage Guide](#9-usage-guide)

---

## 1. Project Overview

Sentinel PV is an end-to-end pharmacovigilance project I built to capture Adverse Drug Events (ADEs) hidden in informal patient conversations on social media. While learning about drug safety systems, I noticed that most traditional pipelines struggle with patient-generated text—people describe side effects using slang, typos, and casual language that keyword-based methods simply miss. This project is my attempt to close that gap by turning messy, real-world narratives into structured, medically meaningful safety signals that can be validated against official clinical data.

The data science core of Sentinel PV (Part A) is centered around a BioBERT-based Named Entity Recognition (NER) model that I fine-tuned using a hybrid dataset. I trained the model on the CADEC gold-standard corpus to ground it in proper medical reasoning, and then adapted it further using ~8,000 scraped Reddit comments so it could learn how patients actually talk online. This combination helps the model handle both formal clinical language and highly informal descriptions. I focused on 20 high-profile, high-patient-volume drugs—including GLP-1 agonists like Ozempic and Wegovy, and biologics such as Humira—because these drugs are frequently discussed online and generate a high volume of side-effect reports.

In production (Part B), I designed Sentinel PV as a multi-agent system orchestrated with LangGraph, where each agent has a clear responsibility. A Detector Agent (BioBERT) extracts drugs and symptoms from raw Reddit posts, a Mapper Agent (Llama 3.2) normalizes slang and misspellings into standardized medical terms (e.g., mapping informal phrases to MedDRA concepts), and an Investigator Agent queries official FDA labeling data through the Model Context Protocol (MCP). Finally, an Analyst Agent pulls everything together into a concise, clinical-style safety report. The entire pipeline is automated using n8n, allowing the system to continuously ingest real Reddit data and convert unstructured social chatter into structured, verifiable pharmacovigilance insights.


## 2. Tech Stack

* **LLMs:** Llama 3.2 (1B & 3B) via **Ollama**.
* **NER Model:** Custom Fine-tuned **BioBERT** (`dmis-lab/biobert-v1.1`).
* **Orchestration:** **LangGraph** (Stateful Multi-Agent Workflows).
* **Protocol:** **MCP (Model Context Protocol)** by Anthropic (Client/Server Architecture).
* **Vector DB:** **ChromaDB** with `sentence-transformers/all-MiniLM-L6-v2`.
* **Backend:** **FastAPI** (Asynchronous).
* **Frontend:** **Streamlit**.
* **Automation:** **n8n** (Workflow Automation).
* **Quality Control:** **Ragas** (RAG Assessment Framework).


## 3. Part A: The Data Science Core (Model Training)

*Location:* `1_data_science/notebooks/sentinel_pv_v3.ipynb`

### **BioBERT Fine-Tuning Strategy**
I fine-tuned the `dmis-lab/biobert-v1.1` model specifically for Token Classification to detect `DRUG` and `ADVERSE_EVENT` entities. This specialized training enables the system to parse unstructured social media text and identify symptoms even when hidden amidst slang or informal grammar.

### **Hybrid Dataset Approach (CADEC + Scraped Reddit Data)**
To ensure the model understands both clinical and casual language, I utilized a dual-source corpus:

* **CADEC:** A gold-standard dataset for establishing medical precision.
* **Scraped Reddit Data:** A custom dataset of ~8,000 comments from drug-specific subreddits (e.g., r/Ozempic) to expose the model to real-world typos and slang.
    * *Tooling:* The scraper used to build this dataset is located at `3_agent_core/ingest_reddit_history.py`.
    * *Usage:* Run `python ingest_reddit_history.py` to fetch fresh data. This generates a `raw_reddit_data.jsonl` file.
    * *Sample:* A structure example is available in `demo-files-generated/raw_reddit_data-DEMO.jsonl`.

### **Model Evaluation Metrics**
The model was evaluated using standard **Precision**, **Recall**, and **F1-Score** metrics. The training process prioritized maximizing the **Recall** for `ADVERSE_EVENT` tags to ensure the system catches every potential safety signal, minimizing the risk of false negatives in a safety-critical context.


*Sentinel PV represents a leap forward in automated drug safety monitoring, combining the precision of BioBERT with the reasoning capabilities of modern LLMs.*

### Configuration (Required)
To keep sensitive details private, this project uses a `terraform.tfvars` file which is excluded from version control. You must create this file manually.

1.  Navigate to `2_infrastructure/terraform/`.
2.  Create a file named `terraform.tfvars`.
3.  Add your specific Google Cloud Project ID inside:

```hcl
project_id = "your-gcp-project-id-here"
```
