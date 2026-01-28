# Updated ingest_knowledge.py
import os
import requests
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Define your Top 10 Target Labels
DRUG_LABELS = {
    "Ozempic": "https://www.accessdata.fda.gov/drugsatfda_docs/label/2017/209637lbl.pdf",
    "Wegovy": "https://www.accessdata.fda.gov/drugsatfda_docs/label/2021/215256s000lbl.pdf",
    "Mounjaro": "https://www.accessdata.fda.gov/drugsatfda_docs/label/2022/215866s000lbl.pdf",
    # ... add the rest of the Top 10 links here
}

DB_PATH = "./chroma_db_storage"

def ingest_all_labels():
    all_docs = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    
    for drug_name, url in DRUG_LABELS.items():
        print(f"⬇️ Downloading {drug_name}...")
        pdf_path = f"{drug_name}_label.pdf"
        
        # Download and Save
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        with open(pdf_path, "wb") as f:
            f.write(response.content)
            
        # Load and add Metadata Tag
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        for doc in docs:
            doc.metadata["drug_name"] = drug_name # This is crucial for filtering later
            
        chunks = text_splitter.split_documents(docs)
        all_docs.extend(chunks)

    # Store everything in one persistent collection
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(
        documents=all_docs,
        embedding=embeddings,
        persist_directory=DB_PATH,
        collection_name="fda_drug_labels"
    )
    print(f"🎉 Success! Brain populated with {len(all_docs)} medical chunks.")
    

if __name__ == "__main__":
    ingest_all_labels()