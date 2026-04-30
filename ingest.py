import os 
import fitz  # PyMuPDF
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

# -------- PATH (your folder) --------
data_dir = "F:/data"

# -------- INIT DB --------
client = chromadb.PersistentClient(path="./chromadb")
collection = client.get_or_create_collection(name="banking_bot")

# -------- MODELS --------
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=64
)

# -------- HELPER: detect bank from filename --------
def detect_bank(filename):
    name = filename.lower()
    if "hdfc" in name:
        return "HDFC"
    elif "sbi" in name:
        return "SBI"
    elif "jpmorgan" in name:
        return "JPMorgan"
    elif "goldman" in name:
        return "Goldman Sachs"
    else:
        return "Unknown"

# -------- PDF PROCESSING --------
def extract_pdf_text(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# -------- MAIN LOOP --------
doc_id = 0

for filename in os.listdir(data_dir):
    file_path = os.path.join(data_dir, filename)

    # ---------- PDF ----------
    if filename.endswith(".pdf"):
        print(f"Processing PDF: {filename}")

        bank = detect_bank(filename)

        raw_text = extract_pdf_text(file_path)
        cleaned_text = " ".join(raw_text.split())
        chunks = text_splitter.split_text(cleaned_text)

        # 👉 LIMIT for testing (remove later)
        #  

        for chunk in chunks:
            vector = embed_model.encode(chunk).tolist()

            collection.add(
                ids=[f"doc_{doc_id}"],
                embeddings=[vector],
                documents=[chunk],
                metadatas=[{
                    "source": filename,
                    "type": "pdf",
                    "bank": bank
                }]
            )
            doc_id += 1

    # ---------- CSV ----------
    elif filename.endswith(".csv"):
        print(f"Processing CSV: {filename}")

        bank = detect_bank(filename)

        df = pd.read_csv(file_path)

        for _, row in df.iterrows():
            text = " ".join([str(val) for val in row.values])
            vector = embed_model.encode(text).tolist()

            collection.add(
                ids=[f"doc_{doc_id}"],
                embeddings=[vector],
                documents=[text],
                metadatas=[{
                    "source": filename,
                    "type": "csv",
                    "bank": bank
                }]
            )
            doc_id += 1

print("✅ Ingestion complete!")