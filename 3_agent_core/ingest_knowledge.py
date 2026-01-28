import os
import requests
import sys

# --- 2026 PYDANTIC COMPATIBILITY ---
import pydantic
import pydantic.v1
sys.modules['pydantic.env_settings'] = pydantic.v1
# -----------------------------------

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# 1. CONFIGURATION - Path set to project root (one level up from 3_agent_core)
# This creates the folder in Sentinel-PV/5_FDA_drug_data
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "5_FDA_drug_data")
DB_PATH = os.path.join(BASE_DIR, "chroma_db_storage")

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
    for drug, url in DRUG_LABELS.items():
        file_path = os.path.join(DATA_DIR, f"{drug}_FDA_Label.pdf")
        
        if os.path.exists(file_path):
            # Check if file is valid (not an HTML error page)
            with open(file_path, "rb") as f:
                header = f.read(5)
                if header == b"%PDF-":
                    print(f"⏩ {drug} label already exists and is valid. Skipping.")
                    continue
                else:
                    print(f"⚠️ {drug} file is corrupted/HTML. Redownloading...")
                    os.remove(file_path)
            
        print(f"⬇️ Downloading {drug} label...")
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(url, headers=headers, timeout=20)
            if response.status_code == 200:
                with open(file_path, "wb") as f:
                    f.write(response.content)
                print(f"✅ {drug} downloaded.")
            else:
                print(f"❌ HTTP {response.status_code} for {drug}. Skipping.")
        except Exception as e:
            print(f"❌ Failed to download {drug}: {e}")

def ingest_data():
    setup_directories()
    download_labels()
    
    all_chunks = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    
    print("\n📖 Processing PDF files...")
    for drug in DRUG_LABELS.keys():
        file_path = os.path.join(DATA_DIR, f"{drug}_FDA_Label.pdf")
        if not os.path.exists(file_path): continue
        
        try:
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            for doc in docs:
                doc.metadata["drug_name"] = drug
            chunks = text_splitter.split_documents(docs)
            all_chunks.extend(chunks)
        except Exception as e:
            print(f"⚠️ Skipping {drug} due to read error: {e}")

    if all_chunks:
        print(f"🧠 Indexing {len(all_chunks)} chunks into ChromaDB...")
        embedding_fn = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = Chroma.from_documents(
            documents=all_chunks, 
            embedding=embedding_fn, 
            persist_directory=DB_PATH,
            collection_name="fda_drug_labels"
        )
        print(f"🎉 Success! Multi-drug base populated at {DB_PATH}")

if __name__ == "__main__":
    ingest_data()