import os
import requests
import sys
import shutil

# --- 2026 PYDANTIC COMPATIBILITY ---
import pydantic
from pydantic_settings import BaseSettings
import pydantic.v1
sys.modules['pydantic.env_settings'] = pydantic.v1
# -----------------------------------

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# 1. CONFIGURATION
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "../5_FDA_drug_data") 
DB_PATH = os.path.join(BASE_DIR, "chroma_db_local")     

# --- COMPLETE LIST OF DRUGS (No more manual adding) ---
DRUG_LABELS = {
    "Ozempic": "https://www.accessdata.fda.gov/drugsatfda_docs/label/2017/209637lbl.pdf",
    "Wegovy": "https://www.accessdata.fda.gov/drugsatfda_docs/label/2021/215256s000lbl.pdf",
    "Mounjaro": "https://www.accessdata.fda.gov/drugsatfda_docs/label/2022/215866s000lbl.pdf",
    "Zepbound": "https://www.accessdata.fda.gov/drugsatfda_docs/label/2023/217806s000lbl.pdf",
    "Trulicity": "https://www.accessdata.fda.gov/drugsatfda_docs/label/2022/125469Orig1s051ltr.pdf",
    "Victoza": "https://www.accessdata.fda.gov/drugsatfda_docs/label/2019/022341s031lbl.pdf",
    "Rybelsus": "https://www.accessdata.fda.gov/drugsatfda_docs/label/2025/213051s024,s028s029lbl.pdf",
    "Jardiance": "https://www.accessdata.fda.gov/drugsatfda_docs/label/2022/204629s033lbl.pdf",
    "Farxiga": "https://www.accessdata.fda.gov/drugsatfda_docs/label/2020/202293Orig1s022lbl.pdf",
    "Januvia": "https://www.accessdata.fda.gov/drugsatfda_docs/label/2008/021995s007lbl.pdf",
    "Eliquis": "https://www.accessdata.fda.gov/drugsatfda_docs/label/2014/202155s006lbl.pdf",
    "Xarelto": "https://www.accessdata.fda.gov/drugsatfda_docs/label/2016/202439s017lbl.pdf",
    "Entresto": "https://www.accessdata.fda.gov/drugsatfda_docs/label/2021/207620s018lbl.pdf",
    "Humira": "https://www.accessdata.fda.gov/drugsatfda_docs/label/2012/125057s232lbl.pdf",
    "Keytruda": "https://www.accessdata.fda.gov/drugsatfda_docs/label/2021/125514s096lbl.pdf",
    "Opdivo": "https://www.accessdata.fda.gov/drugsatfda_docs/label/2017/125554s055lbl.pdf",
    "Enbrel": "https://www.accessdata.fda.gov/drugsatfda_docs/label/2024/103795s5597s5598s5599lbl.pdf",
    "Stelara": "https://www.accessdata.fda.gov/drugsatfda_docs/label/2024/125261s166,761044s014lbl.pdf",
    "Biktarvy": "https://www.accessdata.fda.gov/drugsatfda_docs/label/2018/210251s000lbl.pdf",
    "Dupixent": "https://www.accessdata.fda.gov/drugsatfda_docs/label/2020/761055s015s017lbl.pdf"
}

def setup_directories():
    if not os.path.exists(DATA_DIR):
        print(f"📁 Creating folder: {DATA_DIR}")
        os.makedirs(DATA_DIR)

def download_labels():
    print(f"\n⬇️ Starting Downloads to {DATA_DIR}...")
    headers = {"User-Agent": "Mozilla/5.0"}
    
    for drug, url in DRUG_LABELS.items():
        file_path = os.path.join(DATA_DIR, f"{drug}_FDA_Label.pdf")
        
        # Check if exists and is valid PDF
        if os.path.exists(file_path):
            try:
                with open(file_path, "rb") as f:
                    if f.read(5) == b"%PDF-":
                        print(f"   ⏩ {drug} exists. Skipping.")
                        continue
            except:
                pass # If error, just re-download
            
        print(f"   ⬇️ Downloading {drug}...")
        try:
            response = requests.get(url, headers=headers, timeout=30)
            if response.status_code == 200:
                with open(file_path, "wb") as f:
                    f.write(response.content)
            else:
                print(f"   ❌ Failed {drug}: HTTP {response.status_code}")
        except Exception as e:
            print(f"   ❌ Error {drug}: {e}")

def ingest_data():
    setup_directories()
    download_labels()
    
    # --- CRITICAL FIX: FORCE CPU FOR EMBEDDINGS ---
    print(f"\n⏳ Loading Embedding Model (Safe CPU Mode)...")
    embedding_fn = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'} # <--- PREVENTS CRASH
    )
    
    all_chunks = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    
    print("\n📖 Processing PDFs...")
    for drug in DRUG_LABELS.keys():
        file_path = os.path.join(DATA_DIR, f"{drug}_FDA_Label.pdf")
        if not os.path.exists(file_path): continue
        
        try:
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            # Tag metadata so the Agent knows which drug this is
            for doc in docs:
                doc.metadata["drug_name"] = drug
                doc.metadata["source"] = f"{drug} FDA Label"
            
            chunks = text_splitter.split_documents(docs)
            all_chunks.extend(chunks)
            print(f"   📄 Processed {drug}: {len(chunks)} chunks.")
        except Exception as e:
            print(f"   ⚠️ Error reading {drug}: {e}")

    if all_chunks:
        print(f"\n🧠 Indexing {len(all_chunks)} chunks into ChromaDB...")
        
        # Clear old DB to prevent duplicates
        if os.path.exists(DB_PATH):
            shutil.rmtree(DB_PATH)
            
        vectorstore = Chroma.from_documents(
            documents=all_chunks, 
            embedding=embedding_fn, 
            persist_directory=DB_PATH,
            collection_name="fda_drug_labels"
        )
        print(f"🎉 Success! Real FDA Knowledge Base built at: {DB_PATH}")
    else:
        print("❌ No data found to ingest.")

if __name__ == "__main__":
    ingest_data()