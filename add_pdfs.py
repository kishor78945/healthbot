# add_pdfs.py
import os, json
from pypdf import PdfReader

BASE_DIR = os.environ.get("HEALTHBOT_BASE_DIR", "/content/healthbot")
PDF_DIR = os.path.join(BASE_DIR, "pdfs")
CHUNKS_PATH = os.path.join(BASE_DIR, "all_chunks.json")
SOURCES_PATH = os.path.join(BASE_DIR, "doc_sources.json")
EMBATCH_DIR = os.path.join(BASE_DIR, "emb_batches")
os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(EMBATCH_DIR, exist_ok=True)
os.makedirs(BASE_DIR, exist_ok=True)

def extract_pdfs_to_chunks(pdf_dir=PDF_DIR):
    all_chunks = []
    doc_sources = []
    files = [f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")]
    files.sort()
    for fname in files:
        path = os.path.join(pdf_dir, fname)
        try:
            reader = PdfReader(path)
            for p in reader.pages:
                text = p.extract_text() or ""
                for para in text.split("\n"):
                    if len(para.strip()) > 50:
                        all_chunks.append(para.strip())
                        doc_sources.append(fname)
            print(f"Extracted from {fname}")
        except Exception as e:
            print("Failed:", fname, e)
    json.dump(all_chunks, open(CHUNKS_PATH, "w", encoding="utf-8"), ensure_ascii=False)
    json.dump(doc_sources, open(SOURCES_PATH, "w", encoding="utf-8"), ensure_ascii=False)
    print("Saved chunks:", len(all_chunks))

if __name__ == "__main__":
    extract_pdfs_to_chunks()
    print("Now run your main script (it will call compute_batches on startup).")
