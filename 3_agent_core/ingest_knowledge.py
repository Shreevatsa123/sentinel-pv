import os
import requests
import sys

# --- 2026 PYDANTIC COMPATIBILITY ---
import pydantic
from pydantic_settings import BaseSettings
import pydantic.v1
sys.modules['pydantic.env_settings'] = pydantic.v1
# -----------------------------------

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

PDF_URL = "https://www.accessdata.fda.gov/drugsatfda_docs/label/2017/209637lbl.pdf"
PDF_PATH = "Ozempic_FDA_Label.pdf"
DB_PATH = "./chroma_db_storage"

def download_pdf():
    if not os.path.exists(PDF_PATH):
        print(f"⬇️ Downloading FDA Label...")
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(PDF_URL, headers=headers)
        with open(PDF_PATH, "wb") as f:
            f.write(response.content)
        print("✅ Download Complete.")

def ingest_data():
    download_pdf()
    print("📖 Reading PDF...")
    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()
    
    print("✂️ Splitting text...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)

    print("🧠 Vectorizing and Storing in ChromaDB...")
    embedding_fn = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # This creates the physical folder './chroma_db_storage'
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=embedding_fn, 
        persist_directory=DB_PATH,
        collection_name="fda_drug_labels"
    )
    print(f"🎉 Success! Knowledge Base created at {DB_PATH}")

if __name__ == "__main__":
    ingest_data()