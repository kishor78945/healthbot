import os, json, numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss

BASE_DIR = os.path.dirname(__file__)
PDF_DIR = os.path.join(BASE_DIR, "pdfs")

EMBATCH_DIR = os.path.join(BASE_DIR, "emb_batches"); os.makedirs(EMBATCH_DIR, exist_ok=True)
INDEX_PATH = os.path.join(BASE_DIR, "faiss_index.bin")
META_PATH = os.path.join(BASE_DIR, "emb_meta.json")
CHUNKS_PATH = os.path.join(BASE_DIR, "all_chunks.json")
SOURCES_PATH = os.path.join(BASE_DIR, "doc_sources.json")

# ---------- Load previous state ----------
def load_json(path, default):
    if os.path.exists(path):
        return json.load(open(path, "r", encoding="utf-8"))
    return default

all_chunks = load_json(CHUNKS_PATH, [])
doc_sources = load_json(SOURCES_PATH, [])

# ---------- Extract text ----------
def extract_text_from_pdfs():
    new_chunks, new_sources = [], []
    for fname in os.listdir(PDF_DIR):
        if not fname.lower().endswith(".pdf"):
            continue
        fpath = os.path.join(PDF_DIR, fname)
        print("📄 Reading", fname)
        try:
            reader = PdfReader(fpath)
            for p in reader.pages:
                txt = p.extract_text() or ""
                for para in txt.split("\n"):
                    if len(para.strip()) > 50:
                        new_chunks.append(para.strip())
                        new_sources.append(fname)
        except Exception as e:
            print("⚠️ Skipping", fname, e)
    return new_chunks, new_sources

new_chunks, new_sources = extract_text_from_pdfs()
all_chunks.extend(new_chunks)
doc_sources.extend(new_sources)

json.dump(all_chunks, open(CHUNKS_PATH, "w", encoding="utf-8"), ensure_ascii=False)
json.dump(doc_sources, open(SOURCES_PATH, "w", encoding="utf-8"), ensure_ascii=False)
print(f"✅ Extracted {len(new_chunks)} new chunks, total={len(all_chunks)}")

# ---------- Embeddings ----------
embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

def compute_batches(chunks, batch_size=256):
    processed = 0
    for f in os.listdir(EMBATCH_DIR):
        if f.endswith(".npy"):
            try: processed += np.load(os.path.join(EMBATCH_DIR,f)).shape[0]
            except: pass
    for i in range(processed, len(chunks), batch_size):
        emb = embedder.encode(chunks[i:i+batch_size], convert_to_numpy=True).astype(np.float32)
        np.save(os.path.join(EMBATCH_DIR, f"batch_{i:06d}.npy"), emb)
        json.dump({"dim": emb.shape[1]}, open(META_PATH,"w"))
    print("✅ Embeddings updated")

compute_batches(all_chunks)

# ---------- FAISS index ----------
def build_index():
    meta = json.load(open(META_PATH, "r", encoding="utf-8"))
    dim = meta["dim"]
    idx = faiss.IndexFlatL2(dim)
    for f in sorted(os.listdir(EMBATCH_DIR)):
        if f.endswith(".npy"):
            idx.add(np.load(os.path.join(EMBATCH_DIR,f)))
    faiss.write_index(idx, INDEX_PATH)
    print("✅ FAISS index built with", idx.ntotal, "vectors")

build_index()
