import json
import os
import asyncio
from datasets import Dataset
from ragas import evaluate
# FIX: Use specific classes to avoid deprecation warnings
from ragas.metrics import Faithfulness, AnswerRelevancy
from langchain_ollama import ChatOllama
from sentence_transformers import SentenceTransformer

# 1. SETUP
# We use the smaller model to match your DB
print("⏳ Loading Evaluation Models...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
judge_llm = ChatOllama(model="llama3.2:1b") # The teacher grading the student

# Ragas expects this wrapper class
class RagasEmbeddings:
    def embed_documents(self, texts): return embedding_model.encode(texts).tolist()
    def embed_query(self, text): return embedding_model.encode(text).tolist()

def run_evaluation():
    log_file = "ragas_logs.jsonl"
    
    if not os.path.exists(log_file):
        print(f"❌ Error: {log_file} not found. Run the dashboard first!")
        return

    print("📊 Reading Logs...")
    data_samples = {
        'question': [],
        'answer': [],
        'contexts': [],
        # Note: We do NOT have 'ground_truth' (reference), so we cannot use Context Precision
    }
    
    # Read the JSONL file
    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                log = json.loads(line)
                data_samples['question'].append(log['user_input'])
                data_samples['answer'].append(log['generated_answer'])
                
                # RAGAS requires 'contexts' to be a list of strings.
                # If your log saved it as a string, wrap it in a list.
                ctx = log['retrieved_contexts']
                if isinstance(ctx, str):
                    ctx = [ctx]
                data_samples['contexts'].append(ctx) 
            except:
                continue

    if not data_samples['question']:
        print("⚠️ Log file is empty.")
        return

    # Convert to Dataset format
    dataset = Dataset.from_dict(data_samples)
    
    print(f"🧪 Evaluating {len(dataset)} interactions...")
    
    # FIX: Instantiate metrics manually
    faithfulness_metric = Faithfulness()
    answer_relevancy_metric = AnswerRelevancy()
    
    results = evaluate(
        dataset,
        metrics=[
            faithfulness_metric,      # Did the model hallucinate outside the context?
            answer_relevancy_metric,  # Did the model actually answer the user's question?
            # context_precision       # REMOVED: Requires 'ground_truth' which we don't have.
        ],
        llm=judge_llm,
        embeddings=RagasEmbeddings()
    )
    
    print("\n📝 === QUALITY REPORT CARD ===")
    print(results)
    
    # Save to CSV
    df = results.to_pandas()
    df.to_csv("evaluation_report.csv", index=False)
    print("✅ Detailed report saved to 'evaluation_report.csv'")

if __name__ == "__main__":
    run_evaluation()