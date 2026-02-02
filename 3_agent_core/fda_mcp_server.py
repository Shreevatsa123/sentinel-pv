# pip install mcp
import sys
import os
from mcp.server.fastmcp import FastMCP
from langchain_chroma import Chroma
from sentence_transformers import SentenceTransformer

# 1. Define the Server
mcp = FastMCP("Sentinel FDA Service")

# ⚠️ CRITICAL: Log to stderr, NEVER stdout (stdout is for JSON only)
def log(msg):
    sys.stderr.write(f"[MCP LOG] {msg}\n")
    sys.stderr.flush()

# 2. Load Resources (The DB)
log("Loading FDA Database...")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "chroma_db_local")

# Use the smaller model to match your existing DB (384 dimensions)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

class ManualEmbeddings:
    def embed_documents(self, texts): return embedding_model.encode(texts).tolist()
    def embed_query(self, text): return embedding_model.encode(text).tolist()

try:
    db = Chroma(
        persist_directory=DB_PATH, 
        embedding_function=ManualEmbeddings(), 
        collection_name="fda_drug_labels"
    )
    log("Database loaded successfully.")
except Exception as e:
    log(f"CRITICAL DB ERROR: {e}")

# 3. Expose a Tool
@mcp.tool()
def search_fda_labels(drug_name: str, symptom: str) -> list[str]:
    """Searches official FDA labels for side effects."""
    log(f"Request: {drug_name} + {symptom}")
    
    try:
        results = db.similarity_search(f"{drug_name} side effect {symptom}", k=3)
        return [doc.page_content for doc in results]
    except Exception as e:
        log(f"Search Error: {e}")
        return [f"Error searching database: {e}"]

# 4. Run
if __name__ == "__main__":
    mcp.run()