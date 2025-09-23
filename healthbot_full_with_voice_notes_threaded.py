# healthbot_full_with_voice_notes_threaded.py
# Rewritten to avoid heavy startup work and be Render-friendly.
# Requirements (same as original): ffmpeg, sentence-transformers, faiss, pypdf, pydub, gTTS, twilio, feedparser, apscheduler, langdetect, whisper (optional)

import os
import time
import json
import logging
import re
import calendar
import uuid
import requests
import subprocess
import numpy as np
import threading
from datetime import datetime, timedelta
from flask import Flask, request, send_file
from twilio.twiml.messaging_response import MessagingResponse
from twilio.rest import Client as TwilioClient
from twilio.request_validator import RequestValidator
import atexit

# Optional Colab imports (safe if not present)
try:
    from google.colab import files, drive  # type: ignore
except Exception:
    files = None
    drive = None

# PDF reader
from pypdf import PdfReader

# SentenceTransformer and faiss will be imported lazily to avoid heavy work at import time.
# langdetect
from langdetect import detect as _langdetect, DetectorFactory
DetectorFactory.seed = 0

# Other libs used in functions (pydub, gTTS) are imported lazily as needed.

# ----------------- Config & environment -----------------
LOG_PATH = os.environ.get("HEALTHBOT_LOG_PATH", "/tmp/healthbot.log")
logging.basicConfig(filename=LOG_PATH, level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger().addHandler(logging.StreamHandler())  # also log to stdout
logging.info("HealthBot starting (log -> %s)", LOG_PATH)

# Twilio and ngrok / public tunnel
TWILIO_WHATSAPP_FROM = os.environ.get("TWILIO_WHATSAPP_FROM", "whatsapp:+14155238886")
NGROK_AUTHTOKEN = os.environ.get("NGROK_AUTHTOKEN", "")
SKIP_TWILIO_VALIDATION = os.environ.get("SKIP_TWILIO_VALIDATION", "1") == "1"  # default to True for dev
SKIP_PERSISTENT_SEND = os.environ.get("SKIP_PERSISTENT_SEND", "0") == "1"

# OpenAI/NVIDIA client environment variables (kept as in original)
OPENAI_API_KEY = os.environ.get("NVIDIA_API_KEY")  # keep variable name same as original to minimize changes
OPENAI_BASE_URL = os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")

# Twilio client (create lazily)
_tw_client = None
def get_twilio_client():
    global _tw_client
    if _tw_client is None:
        sid = os.environ.get("TWILIO_ACCOUNT_SID")
        token = os.environ.get("TWILIO_AUTH_TOKEN")
        if sid and token:
            _tw_client = TwilioClient(sid, token)
        else:
            _tw_client = None
    return _tw_client

# ----------------- Storage paths -----------------
BASE_DIR = os.environ.get("HEALTHBOT_BASE_DIR", "/tmp/healthbot")
os.makedirs(BASE_DIR, exist_ok=True)
EMBATCH_DIR = os.path.join(BASE_DIR, "emb_batches"); os.makedirs(EMBATCH_DIR, exist_ok=True)
INDEX_PATH = os.path.join(BASE_DIR, "faiss_index.bin")
META_PATH = os.path.join(BASE_DIR, "emb_meta.json")
CHUNKS_PATH = os.path.join(BASE_DIR, "all_chunks.json")
SOURCES_PATH = os.path.join(BASE_DIR, "doc_sources.json")
PREFS_PATH = os.path.join(BASE_DIR, "user_lang_prefs.json")
ALERTS_QUEUE = os.path.join(BASE_DIR, "alerts_queue.json")
SUBSCRIBERS_PATH = os.path.join(BASE_DIR, "subscribers.json")
LAST_SEEN_PATH = os.path.join(BASE_DIR, "last_seen.json")
SEEN_PATH = os.path.join(BASE_DIR, "seen_users.json")
MEDIA_STORE = os.path.join(BASE_DIR, "media_files"); os.makedirs(MEDIA_STORE, exist_ok=True)

logging.info("Base dir: %s", BASE_DIR)

# ----------------- Feeds -----------------
FEED_URLS = [
    "https://ncdc.gov.in/outbreaks.rss",
    "https://www.who.int/feeds/entity/csr/don/en/rss.xml",
    "https://www.emro.who.int/rss-feeds/disease-outbreaks.rss",
    "https://www.ecdc.europa.eu/en/rss-feeds",
    "https://tools.cdc.gov/api/v2/resources/media/132608.rss",
    "http://emergency.cdc.gov/rss/"
]

# ----------------- PDF ingestion helpers -----------------
def load_chunks():
    try:
        if os.path.exists(CHUNKS_PATH) and os.path.exists(SOURCES_PATH):
            return json.load(open(CHUNKS_PATH, "r", encoding="utf-8")), json.load(open(SOURCES_PATH, "r", encoding="utf-8"))
    except Exception:
        logging.exception("Failed loading chunks/sources")
    return [], []

def save_chunks(all_chunks, doc_sources):
    try:
        json.dump(all_chunks, open(CHUNKS_PATH, "w", encoding="utf-8"), ensure_ascii=False)
        json.dump(doc_sources, open(SOURCES_PATH, "w", encoding="utf-8"), ensure_ascii=False)
    except Exception:
        logging.exception("Failed saving chunks/sources")

all_chunks, doc_sources = load_chunks()
if not all_chunks:
    logging.info("No pre-extracted PDF chunks found. If running in Colab with 'files' available, PDF upload will be used.")
    if files:
        try:
            uploaded = files.upload()
            for pdf in uploaded.keys():
                try:
                    r = PdfReader(pdf)
                    for p in r.pages:
                        txt = p.extract_text() or ""
                        for para in txt.split("\n"):
                            if len(para.strip()) > 50:
                                all_chunks.append(para.strip())
                                doc_sources.append(pdf)
                except Exception:
                    logging.exception("Skipping PDF %s", pdf)
            save_chunks(all_chunks, doc_sources)
            logging.info("Extracted %d PDF chunks from uploaded files", len(all_chunks))
        except Exception:
            logging.exception("Colab upload failed or not available")

# ----------------- Embeddings & FAISS (lazy) -----------------
# Environment controls:
#   HEALTHBOT_PRECOMPUTE_EMBS=1  -> run compute_batches on startup (only set when you want to precompute)
#   HEALTHBOT_FORCE_LOAD_INDEX=1 -> attempt to load index at startup (only if index fits memory)
EMBEDDER = None
_INDEX = None
_embed_lock = threading.Lock()
_index_lock = threading.Lock()

def get_embedder():
    """Lazily initialize SentenceTransformer embedder."""
    global EMBEDDER
    if EMBEDDER is None:
        with _embed_lock:
            if EMBEDDER is None:
                logging.info("Initializing sentence-transformers embedder (this may download model files)...")
                try:
                    from sentence_transformers import SentenceTransformer as _ST  # lazy import
                    EMBEDDER = _ST("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
                    logging.info("Embedder ready.")
                except Exception:
                    logging.exception("Failed to initialize embedder")
                    EMBEDDER = None
    return EMBEDDER

def compute_batches(chunks, batch_size=256):
    """Compute and store embeddings in EMBATCH_DIR. Run this explicitly (e.g., in CI or a build step)."""
    e = get_embedder()
    if e is None:
        logging.warning("Embedder not available; cannot compute batches.")
        return
    processed = 0
    try:
        for f in os.listdir(EMBATCH_DIR):
            if f.endswith(".npy"):
                try: processed += np.load(os.path.join(EMBATCH_DIR, f)).shape[0]
                except Exception: pass
    except Exception:
        logging.exception("Failed to inspect EMBATCH_DIR")
    for i in range(processed, len(chunks), batch_size):
        try:
            emb = e.encode(chunks[i:i+batch_size], convert_to_numpy=True).astype(np.float32)
            np.save(os.path.join(EMBATCH_DIR, f"batch_{i:06d}.npy"), emb)
            json.dump({"dim": emb.shape[1]}, open(META_PATH, "w"))
        except Exception:
            logging.exception("Failed computing/saving batch starting at %d", i)
    logging.info("compute_batches finished (processed from %d)", processed)

def compute_batches_if_requested():
    if os.environ.get("HEALTHBOT_PRECOMPUTE_EMBS", "") == "1":
        logging.info("HEALTHBOT_PRECOMPUTE_EMBS=1 -> running compute_batches now")
        compute_batches(all_chunks)
    else:
        logging.info("Skipping compute_batches at startup (HEALTHBOT_PRECOMPUTE_EMBS not set)")

def load_index():
    """Load FAISS index lazily. Returns None if cannot load."""
    global _INDEX
    if _INDEX is None:
        with _index_lock:
            if _INDEX is None:
                try:
                    if not os.path.exists(META_PATH):
                        logging.warning("META_PATH missing; cannot load FAISS index (no precomputed embeddings).")
                        return None
                    meta = json.load(open(META_PATH, "r", encoding="utf-8"))
                    dim = meta["dim"]
                    import faiss  # lazy import
                    idx = faiss.IndexFlatL2(dim)
                    for f in sorted(os.listdir(EMBATCH_DIR)):
                        if f.endswith(".npy"):
                            try: idx.add(np.load(os.path.join(EMBATCH_DIR, f)))
                            except Exception:
                                logging.exception("Failed loading batch %s", f)
                    try:
                        faiss.write_index(idx, INDEX_PATH)
                    except Exception:
                        logging.exception("faiss write_index failed")
                    _INDEX = idx
                    logging.info("FAISS index loaded with %d vectors", idx.ntotal)
                except Exception:
                    logging.exception("load_index failed")
                    _INDEX = None
    return _INDEX

# Optionally run compute/load only if requested (safe default: do nothing)
compute_batches_if_requested()
if os.environ.get("HEALTHBOT_FORCE_LOAD_INDEX", "") == "1":
    load_index()

def retrieve_with_sources(query, top_k=8):
    """Return retrieved chunks, sources, indices, and confidences for a query."""
    if not query:
        return [], [], [], []
    idx = load_index()
    if idx is None or idx.ntotal == 0:
        logging.info("No FAISS index available; returning empty retrievals.")
        return [], [], [], []
    try:
        emb = get_embedder()
        if not emb:
            logging.warning("Embedder not available for retrieval")
            return [], [], [], []
        q_emb = emb.encode([query], convert_to_numpy=True).astype(np.float32)
        D, I = idx.search(q_emb, top_k)
        dists = D[0].tolist(); idxs = I[0].tolist()
        retrieved = []; sources = []; confs = []
        for dist, ix in zip(dists, idxs):
            if ix < 0 or ix >= len(all_chunks): continue
            retrieved.append(all_chunks[ix])
            sources.append(doc_sources[ix] if ix < len(doc_sources) else None)
            confs.append(1.0 / (1.0 + float(dist)))
        return retrieved, sources, idxs, confs
    except Exception:
        logging.exception("retrieve_with_sources failed")
        return [], [], [], []

# ----------------- Language utils -----------------
LANG_MAP = {"ta": "Reply in Tamil. Use local simple words.", "hi": "Reply in Hindi. Use local simple words.", "en": "Reply in English. Use clear simple words."}
TRANSLATIONS = {"fallback": {"en": "Sorry, no clear answer. Please ask a health worker.", "ta": "மன்னிக்கவும், தெளிவான பதில் இல்லை. உள்ளூர் சுகாதார பணியாளரை அணுகவும்.", "hi": "क्षमा करें, स्पष्ट उत्तर उपलब्ध नहीं है। स्वास्थ्य कार्यकर्ता से पूछें।"}}

def detect_lang(txt):
    txt = txt or ""
    if any('\u0B80' <= ch <= '\u0BFF' for ch in txt): return "ta"
    if any('\u0900' <= ch <= '\u097F' for ch in txt): return "hi"
    try:
        l = _langdetect(txt)
    except Exception:
        return "en"
    if l.startswith("ta"): return "ta"
    if l.startswith("hi"): return "hi"
    return "en"

def load_prefs():
    try:
        if os.path.exists(PREFS_PATH):
            return json.load(open(PREFS_PATH, "r", encoding="utf-8"))
    except Exception:
        logging.exception("Failed loading prefs")
    return {}

def save_prefs(p):
    try:
        json.dump(p, open(PREFS_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    except Exception:
        logging.exception("Failed saving prefs")

user_lang_prefs = load_prefs()

# ----------------- Misc helpers & state -----------------
try:
    seen_users = set(json.load(open(SEEN_PATH, "r", encoding="utf-8")))
except Exception:
    seen_users = set()

def persist_seen_users():
    try:
        json.dump(list(seen_users), open(SEEN_PATH, "w", encoding="utf-8"), ensure_ascii=False)
    except Exception:
        logging.exception("Failed saving seen_users")

_prefs_lock = threading.Lock()
def set_pref_if_script_detected(sender, incoming_msg):
    if not sender or not incoming_msg: return
    try:
        txt = incoming_msg.strip().lower()
        if any(w in txt for w in ("reply in tamil", "in tamil", "tamil please", "தமிழ்")): code = "ta"
        elif any(w in txt for w in ("reply in hindi", "in hindi", "hindi please", "हिंदी", "हिन्दी")): code = "hi"
        elif any(w in txt for w in ("reply in english", "in english", "english please")): code = "en"
        else: return
        with _prefs_lock:
            user_lang_prefs[sender] = code
            try: save_prefs(user_lang_prefs)
            except Exception: logging.exception("Failed saving prefs in set_pref_if_script_detected")
    except Exception:
        logging.exception("set_pref_if_script_detected failed")

# ----------------- Audio preprocess & ASR helpers -----------------
def _preprocess_audio(in_path, out_path, sample_rate=16000):
    try:
        cmd = ["ffmpeg", "-y", "-i", in_path, "-ac", "1", "-ar", str(sample_rate), "-af", "highpass=f=80,dynaudnorm", out_path]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return out_path
    except Exception:
        logging.exception("ffmpeg preprocess failed, falling back to pydub")
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_file(in_path)
            audio = audio.set_channels(1).set_frame_rate(sample_rate)
            audio.export(out_path, format="wav")
            return out_path
        except Exception:
            logging.exception("fallback preprocess failed")
            return in_path

def _download_media(media_url, save_path):
    try:
        auth = None
        tw_sid = os.environ.get("TWILIO_ACCOUNT_SID")
        tw_token = os.environ.get("TWILIO_AUTH_TOKEN")
        if "api.twilio.com" in (media_url or "") and tw_sid and tw_token:
            auth = (tw_sid, tw_token)
        r = requests.get(media_url, stream=True, timeout=30, auth=auth)
        r.raise_for_status()
        with open(save_path, "wb") as fh:
            for chunk in r.iter_content(chunk_size=8192):
                if not chunk: continue
                fh.write(chunk)
        return save_path
    except Exception:
        logging.exception("Failed to download media %s", media_url)
        raise

def _convert_to_wav(in_path, out_path):
    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_file(in_path)
        audio.export(out_path, format="wav")
        return out_path
    except Exception:
        logging.exception("convert_to_wav failed; returning original path")
        return in_path

def _detect_lang_from_text(txt):
    try:
        d = _langdetect(txt or "")
        if d.startswith("ta"): return "ta"
        if d.startswith("hi"): return "hi"
        return "en"
    except Exception:
        return "en"

def transcribe_with_local_whisper(wav_path, lang_hint=None):
    try:
        import whisper  # type: ignore
        model_name = os.environ.get("WHISPER_MODEL", "small")
        model = whisper.load_model(model_name)
        opts = {"task": "transcribe"}
        if lang_hint and lang_hint in ("ta", "hi", "en"): opts["language"] = lang_hint
        result = model.transcribe(wav_path, **opts)
        text = (result.get("text") or "").strip()
        detected = (result.get("language") or "").lower() if isinstance(result.get("language"), str) else ""
        if detected.startswith("ta"): lang_code = "ta"
        elif detected.startswith("hi") or detected.startswith("hi-in"): lang_code = "hi"
        else:
            try: lang_code = _detect_lang_from_text(text)
            except Exception: lang_code = "en"
        return text, lang_code, None
    except Exception:
        logging.exception("local whisper transcription failed")
        return None, None, None

def generate_tts(text, lang_code="en"):
    try:
        from gtts import gTTS
        if not text: return None, None
        safe_code = "en" if lang_code not in ("en", "hi", "ta") else lang_code
        parts = []
        i = 0
        limit_chars = 1400
        while i < len(text):
            parts.append(text[i:i+limit_chars]); i += limit_chars
        fname = f"tts_{uuid.uuid4().hex[:10]}.mp3"
        fpath = os.path.join(MEDIA_STORE, fname)
        gTTS(text=parts[0], lang=safe_code).save(fpath)
        public_base = getattr(globals().get("public_tunnel", None), "public_url", None) or os.environ.get("PUBLIC_BASE_URL")
        public_url = (public_base.rstrip("/") + "/media/" + fname) if public_base else None
        return public_url, fpath
    except Exception:
        logging.exception("gTTS generation failed")
        return None, None

# ----------------- Feeds / outbreak helpers -----------------
MONTH_NAME_TO_NUM = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}
SHORT_MONTHS = {m.lower(): i for i, m in enumerate(calendar.month_abbr) if m}

def _entry_timestamp(entry):
    for key in ("published_parsed", "updated_parsed"):
        t = entry.get(key)
        if t:
            try: return int(time.mktime(t))
            except Exception: pass
    return 0

def _month_year_from_text(text):
    text_l = (text or "").lower()
    now = datetime.utcnow()
    if "last month" in text_l:
        first = datetime(now.year, now.month, 1) - timedelta(days=1)
        start = datetime(first.year, first.month, 1)
        last_day = calendar.monthrange(start.year, start.month)[1]
        end = datetime(start.year, start.month, last_day, 23, 59, 59)
        return int(start.timestamp()), int(end.timestamp())
    if "this month" in text_l:
        start = datetime(now.year, now.month, 1)
        last_day = calendar.monthrange(start.year, start.month)[1]
        end = datetime(start.year, start.month, last_day, 23, 59, 59)
        return int(start.timestamp()), int(end.timestamp())
    m = re.search(r'\b(' + '|'.join(list(MONTH_NAME_TO_NUM.keys()) + list(SHORT_MONTHS.keys())) + r')\s+(\d{4})\b', text_l)
    if m:
        mon = m.group(1); year = int(m.group(2))
        mon_num = MONTH_NAME_TO_NUM.get(mon) or SHORT_MONTHS.get(mon)
        if mon_num:
            start = datetime(year, mon_num, 1)
            last_day = calendar.monthrange(year, mon_num)[1]
            end = datetime(year, mon_num, last_day, 23, 59, 59)
            return int(start.timestamp()), int(end.timestamp())
    m2 = re.search(r'\b(\d{4})[-/](\d{1,2})\b', text_l) or re.search(r'\b(\d{1,2})[-/](\d{4})\b', text_l)
    if m2:
        g1, g2 = m2.group(1), m2.group(2)
        try:
            if len(g1) == 4:
                year = int(g1); mon = int(g2)
            else:
                year = int(g2); mon = int(g1)
            if 1 <= mon <= 12:
                start = datetime(year, mon, 1)
                last_day = calendar.monthrange(year, mon)[1]
                end = datetime(year, mon, last_day, 23, 59, 59)
                return int(start.timestamp()), int(end.timestamp())
        except Exception:
            pass
    return None

import feedparser
def fetch_and_detect_events():
    events = []
    for url in FEED_URLS:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries:
                events.append({'id': entry.get('id', entry.get('link')), 'title': entry.get('title'),
                               'summary': entry.get('summary', '') or entry.get('description', ''), 'link': entry.get('link'),
                               'published': entry.get('published', '') or entry.get('updated', ''), 'source': url})
        except Exception:
            logging.exception("Failed feed %s", url)
    return events

try:
    alerts_queue = json.load(open(ALERTS_QUEUE, "r", encoding="utf-8"))
except Exception:
    alerts_queue = []

def dedupe_and_queue(events):
    existing_ids = set(a['id'] for a in alerts_queue if a.get('id'))
    new = []
    for e in events:
        if not e.get('id'): continue
        if e['id'] in existing_ids: continue
        text_body = (e.get('title', '') or '') + ' ' + (e.get('summary', '') or '')
        if len(text_body.strip()) < 30: continue
        alerts_queue.append({'id': e['id'], 'title': e.get('title'), 'summary': e.get('summary'), 'link': e.get('link'),
                             'published': e.get('published'), 'source': e.get('source'), 'status': 'pending', 'queued_at': time.time()})
        new.append(e)
    try:
        json.dump(alerts_queue, open(ALERTS_QUEUE, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    except Exception:
        logging.exception("Failed to persist alerts_queue")
    return new

def fetch_recent_outbreaks(disease=None, region=None, limit=5, lookback=80, time_window=None):
    matches = []
    qd = (disease or "").strip().lower(); qr = (region or "").strip().lower()
    for url in FEED_URLS:
        try:
            feed = feedparser.parse(url)
        except Exception:
            logging.exception("Failed parse feed %s", url); continue
        for entry in feed.entries[:lookback]:
            title = (entry.get("title") or "").strip()
            summary = (entry.get("summary") or entry.get("description") or "").strip()
            link = entry.get("link") or ""
            pub = entry.get("published") or entry.get("updated") or ""
            ts = _entry_timestamp(entry)
            if time_window:
                start_ts, end_ts = time_window
                if not ts or ts < start_ts or ts > end_ts: continue
            combined = (title + " " + summary).lower()
            if qd and qd not in combined: continue
            if qr and qr not in combined: continue
            matches.append({"title": title, "summary": summary, "link": link, "published": pub, "source": url, "ts": ts})
    matches = sorted(matches, key=lambda e: e.get("ts", 0), reverse=True)
    return matches[:limit]

def _extract_region_from_text(user_text):
    m = re.search(r'\b(?:in|near|around|at)\s+([A-Za-z\u0900-\u097F\u0B80-\u0BFF\s\-,]+)', user_text, re.IGNORECASE)
    if m:
        region = m.group(1).strip().split(",")[0]; return region[:60].strip()
    return None

def _extract_disease_from_text(user_text):
    kws = ["dengue", "chikungunya", "malaria", "cholera", "measles", "influenza", "covid", "ebola"]
    t = (user_text or "").lower()
    for k in kws:
        if k in t: return k
    return None

def _extract_counts_from_text(s):
    nums = re.findall(r'(\d{1,7})\s*(?:cases|case|confirmed|suspected|patients)', (s or "").lower())
    return [int(n) for n in nums]

# ----------------- Language ensemble for ASR/text -----------------
def detect_language_ensemble(transcript, whisper_lang=None):
    if not transcript: return whisper_lang or "en"
    if any('\u0B80' <= ch <= '\u0BFF' for ch in transcript): return "ta"
    if any('\u0900' <= ch <= '\u097F' for ch in transcript): return "hi"
    votes = {}
    def vote(c): votes[c] = votes.get(c, 0) + 1
    if whisper_lang in ("ta", "hi", "en"): vote(whisper_lang)
    try:
        ft = _langdetect(transcript or "")
        if ft.startswith("ta"): vote("ta")
        elif ft.startswith("hi"): vote("hi")
        else: vote("en")
    except Exception:
        vote("en")
    roman_tamil_kws = ["enna", "sari", "yenn", "unga", "thaan", "vaa", "ungalukku", "romba", "ippo", "vera"]
    roman_hindi_kws = ["kaise", "kya", "hai", "nahi", "mera", "meri", "tum", "thik", "kaam", "karke"]
    low = (transcript or "").lower()
    for w in roman_tamil_kws:
        if w in low: vote("ta"); break
    for w in roman_hindi_kws:
        if w in low: vote("hi"); break
    sorted_votes = sorted(votes.items(), key=lambda x: x[1], reverse=True)
    if not sorted_votes: return "en"
    return sorted_votes[0][0]

# ----------------- Answer / LLM pipeline -----------------
# NOTE: This code calls your OpenAI/NVIDIA client lazily. Keep as-is if you have the client set up.
def answer_with_sources(user_text, user_lang=None, top_k=8, doc_threshold=0.22):
    text_l = (user_text or "").strip().lower()
    DISEASE_KEYWORDS = ["dengue", "chikungunya", "malaria", "cholera", "measles", "influenza", "covid", "ebola"]
    OUTBREAK_INDICATORS = ["outbreak", "outbreaks", "cases", "case", "suspected", "confirmed", "how many", "count", "counts", "reported", "reports", "cluster"]
    TIME_PLACE_INDICATORS = ["last month", "this month", "in ", "near ", "during ", "between ", "202", "jan", "feb", "mar", "april", "may", "june", "july", "aug", "sep", "oct", "nov", "dec"]

    has_disease = any(d in text_l for d in DISEASE_KEYWORDS)
    has_outbreak_word = any(w in text_l for w in OUTBREAK_INDICATORS)
    has_time_place = bool(_month_year_from_text(user_text)) or any(tp in text_l for tp in TIME_PLACE_INDICATORS)
    asks_numeric = bool(re.search(r'\b(how many|how much|what(?:\s+number)?|count|counts|number of|total)\b', text_l))

    treat_as_outbreak = False
    if has_disease and (has_outbreak_word or has_time_place or asks_numeric):
        treat_as_outbreak = True
    if re.search(r'\b(symptom|symptoms|signs|treatment|treat|vaccine|vaccination|prevent|prevention|what is|how to|when to|dose)\b', text_l):
        treat_as_outbreak = False

    if treat_as_outbreak:
        disease = _extract_disease_from_text(user_text)
        region = _extract_region_from_text(user_text)
        time_w = _month_year_from_text(user_text) or None
        events = fetch_recent_outbreaks(disease=disease, region=region, limit=8, lookback=120, time_window=time_w)
        if not events and (disease or region or time_w):
            events = fetch_recent_outbreaks(disease=None, region=None, limit=8, lookback=160, time_window=time_w)
        if not events:
            events = fetch_recent_outbreaks(disease=disease, region=region, limit=8, lookback=200, time_window=None)
        if events:
            counts_found = []
            for e in events:
                counts_found.extend(_extract_counts_from_text((e.get("summary", "") or "") + " " + (e.get("title", "") or "")))
            if counts_found:
                total_counts = sum(counts_found)
                time_str = ""
                if time_w:
                    start_dt = datetime.utcfromtimestamp(time_w[0]).strftime("%Y-%m-%d")
                    end_dt = datetime.utcfromtimestamp(time_w[1]).strftime("%Y-%m-%d")
                    time_str = f" between {start_dt} and {end_dt}"
                disease_str = f" {disease}" if disease else ""
                region_str = f" in {region}" if region else ""
                reply = f"Reported{disease_str}{region_str}{time_str}:\n• ~{total_counts} cases referenced across {len(events)} recent reports.\n\n"
                for e in events[:3]:
                    reply += f"• {e.get('title')}\n  {(e.get('summary') or '')[:300]}\n  Published: {e.get('published')}\n  Source: {e.get('link') or e.get('source')}\n\n"
                reply += "What to do (brief):\n1) Seek healthcare for high fever or danger signs.\n2) Remove stagnant water, use repellents, use nets.\n"
                srcs = []; seen = set()
                for e in events:
                    u = e.get('link') or e.get('source')
                    if u and u not in seen: seen.add(u); srcs.append(u)
                return reply, srcs, 1.0
            # If numeric counts not found — produce concise summary using LLM if configured
            try:
                # Use OpenAI/NVIDIA client if available; otherwise fallback to deterministic summary
                from openai import OpenAI  # lazy import; original code used writer/palmyra etc.
                client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL) if OPENAI_API_KEY else None
                if client:
                    event_texts = []
                    for e in events[:6]:
                        event_texts.append(f"Title: {e.get('title')}\nDate: {e.get('published')}\nLink: {e.get('link') or e.get('source')}\nSummary: {e.get('summary')}\n")
                    context_blob = "\n\n".join(event_texts)
                    system_prompt = ("You are a factual summarizer. Given the feed items below, extract concise verifiable findings (cases, locations, dates) where possible and provide 2 short actionable bullets. If numbers aren't in sources, say 'no numeric counts in sources'. Keep short and factual. Provide source URLs.")
                    user_msg = f"Context:\n{context_blob}\n\nQuestion: Summarize outbreak statistics and immediate actions for: {user_text}"
                    resp = client.chat.completions.create(model="writer/palmyra-med-70b-32k", messages=[{"role":"system","content":system_prompt},{"role":"user","content":user_msg}], temperature=0.0, max_tokens=400)
                    l_ans = resp.choices[0].message.content.strip()
                    srcs = []; seen = set()
                    for e in events:
                        u = e.get('link') or e.get('source')
                        if u and u not in seen: seen.add(u); srcs.append(u)
                    return l_ans, srcs, 1.0
            except Exception:
                logging.exception("LLM synth failed for feed items")
            # fallback to simple summary
            lines = []
            for e in events[:5]:
                lines.append(f"{e.get('title')} ({e.get('published')})\n{(e.get('summary') or '')[:240]}\nSource: {e.get('link') or e.get('source')}")
            reply = "Recent outbreak items:\n\n" + "\n\n".join(lines) + "\n\nWhat to do: remove stagnant water, seek care for high fever."
            srcs = [e.get('link') or e.get('source') for e in events]
            return reply, srcs, 0.7
        return ("I could not find recent outbreak feed reports matching your query. Try a different place/time or check these sources: " + ", ".join(FEED_URLS)), FEED_URLS, 0.1

    # Normal knowledge retrieval / LLM answer path
    retrieved, sources, idxs, confs = retrieve_with_sources(user_text, top_k=top_k)
    best_sim = confs[0] if confs else 0.0
    if user_lang is None: user_lang = detect_lang(user_text)
    if user_lang not in LANG_MAP: user_lang = "en"
    lang_instruction = LANG_MAP.get(user_lang)
    lang_prefix_map = {"ta": "Reply ONLY in Tamil. ", "hi": "Reply ONLY in Hindi. ", "en": "Reply ONLY in English. "}
    lang_prefix = lang_prefix_map.get(user_lang, "Reply ONLY in English. ")
    system_base = ("You are a multilingual health assistant. Answers: ~80% specific/evidence-based and ~20% short awareness. Structure: 1) Clinical specifics 2) Preventive/awareness (short). Be conservative if docs lack info.")
    system_prompt = lang_prefix + (lang_instruction or "") + " " + system_base

    # Compose prompt depending on whether context is present
    if best_sim >= doc_threshold and len(retrieved) > 0:
        context = "\n\n".join(retrieved[:6])
        prompt = [{"role": "system", "content": system_prompt + " Use ONLY the provided context when it contains the answer."},
                  {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {user_text}\nDeliver ~80% clinical specifics then 20% awareness."}]
    else:
        prompt = [{"role": "system", "content": system_prompt + " Use reliable public health knowledge cautiously when docs don't contain info."},
                  {"role": "user", "content": f"Question: {user_text}\nProvide a structured answer (clinical specifics then short awareness)."}]
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL) if OPENAI_API_KEY else None
        if client:
            resp = client.chat.completions.create(model="writer/palmyra-med-70b-32k", messages=prompt, temperature=0.15, max_tokens=700)
            answer = resp.choices[0].message.content.strip()
            logging.info("LLM answer received (len=%d)", len(answer or ""))
        else:
            logging.warning("No OpenAI/NVIDIA client configured; using fallback translation text")
            answer = TRANSLATIONS["fallback"]["en"]
    except Exception:
        logging.exception("LLM call failed")
        answer = TRANSLATIONS["fallback"]["en"]
    srcs_clean = []; seen = set()
    for s in sources or []:
        if s and s not in seen: seen.add(s); srcs_clean.append(s)
    return answer, srcs_clean if srcs_clean else None, best_sim

# ----------------- Quick local tests (safe) -----------------
def _quick_local_tests():
    logging.info("Quick tests: Tamil detect: %s", detect_lang("இந்த வாக்கியம் தமிழ் மொழியில் உள்ளது"))
    logging.info("Quick tests: Hindi detect: %s", detect_lang("यह वाक्य हिन्दी में है"))
    txt = "Are there dengue cases in Chennai last month?"
    try:
        ans, srcs, conf = answer_with_sources(txt)
        logging.info("ANS preview: %s", (ans or "")[:400])
        logging.info("Sources: %s conf: %s", srcs, conf)
    except Exception:
        logging.exception("Quick local test failed")

# Run quick tests only when explicitly requested via env (avoid at import)
if os.environ.get("HEALTHBOT_RUN_QUICK_TESTS", "0") == "1":
    _quick_local_tests()

# ----------------- Flask app & Twilio helpers -----------------
app = Flask(__name__)
validator = RequestValidator(os.environ.get("TWILIO_AUTH_TOKEN", ""))

def _twilio_send_and_log(create_func, *args, **kwargs):
    try:
        msg = create_func(*args, **kwargs)
        try: logging.info("Twilio sent sid=%s to=%s status=%s", getattr(msg, "sid", None), kwargs.get("to"), getattr(msg, "status", None))
        except Exception: logging.info("Twilio sent message (no sid)")
        return msg
    except Exception:
        logging.exception("Twilio API call failed")
        return None

@app.route("/media/<path:fname>", methods=["GET"])
def serve_media(fname):
    fpath = os.path.join(MEDIA_STORE, fname)
    if not os.path.exists(fpath): return ("Not found", 404)
    return send_file(fpath, as_attachment=False)

# load state files gracefully
try: subscribers = json.load(open(SUBSCRIBERS_PATH, "r", encoding="utf-8"))
except Exception: subscribers = []
try: last_seen = json.load(open(LAST_SEEN_PATH, "r", encoding="utf-8"))
except Exception: last_seen = {}

rate_map = {}
def allowed_rate(sender, limit=6, per_seconds=60):
    now = time.time()
    lst = rate_map.get(sender, [])
    lst = [t for t in lst if now - t < per_seconds]
    if len(lst) >= limit: return False
    lst.append(now); rate_map[sender] = lst; return True

# ----------------- Voice processing (no sending) -----------------
def handle_incoming_voice_media_process_only(sender, media_url, content_type):
    try:
        tmpname = f"in_{uuid.uuid4().hex[:8]}"
        in_path = os.path.join(MEDIA_STORE, tmpname)
        _download_media(media_url, in_path)

        wav_path = in_path + ".wav"
        try: _convert_to_wav(in_path, wav_path)
        except Exception: wav_path = in_path

        preproc_path = wav_path + ".pre.wav"
        preproc_path = _preprocess_audio(wav_path, preproc_path)

        pref_lang = user_lang_prefs.get(sender)
        lang_hint_for_asr = pref_lang if pref_lang in ("ta", "hi", "en") else None

        transcript, detected_lang, whisper_conf = transcribe_with_local_whisper(preproc_path, lang_hint=lang_hint_for_asr)
        if not transcript:
            transcript, detected_lang, whisper_conf = (None, None, None)
        if not transcript:
            return {"success": False, "error": "transcription_failed"}

        if pref_lang:
            final_lang = pref_lang
        else:
            final_lang = detect_language_ensemble(transcript, whisper_lang=detected_lang)

        logging.info("VOICE_NOTE | sender=%s detected=%s pref=%s final=%s", sender, detected_lang, pref_lang, final_lang)

        try:
            answer, sources, conf = answer_with_sources(transcript, user_lang=final_lang, top_k=6)
        except Exception:
            logging.exception("answer pipeline failed for voice note")
            answer = TRANSLATIONS["fallback"].get(final_lang, TRANSLATIONS["fallback"]["en"])
            sources = None; conf = 0.0

        tts_answer = answer if len(answer) <= 1500 else (answer[:1400] + "...")
        tts_url, tts_path = generate_tts(tts_answer, lang_code=final_lang)
        return {"success": True, "transcript": transcript, "answer": answer, "sources": sources, "conf": conf, "lang": final_lang, "tts_url": tts_url}
    except Exception:
        logging.exception("handle_incoming_voice_media_process_only failed")
        return {"success": False, "error": "exception"}

# ----------------- Background worker to process voice note and send via REST -----------------
# ----------------- Background worker to process text and send via REST -----------------
def _process_text_and_send_async(sender, incoming_msg):
    try:
        logging.info("BG: starting text processing for %s", sender)

        # detect user language
        set_pref_if_script_detected(sender, incoming_msg)
        user_lang = user_lang_prefs.get(sender) or detect_lang(incoming_msg)

        # run pipeline
        try:
            answer, sources, conf = answer_with_sources(
                incoming_msg,
                user_lang=user_lang,
                top_k=8,
                doc_threshold=0.22
            )
        except Exception:
            logging.exception("answer pipeline failed for text message")
            answer = TRANSLATIONS["fallback"].get(user_lang, TRANSLATIONS["fallback"]["en"])
            sources, conf = None, 0.0

        # choose reply
        if (("don't know" in (answer or "").lower() or "i do not know" in (answer or "").lower()) 
            or (conf < 0.05 and len((answer or "").split()) < 6)):
            reply = TRANSLATIONS["fallback"].get(user_lang, TRANSLATIONS["fallback"]["en"])
        else:
            reply = answer

        # send persistent reply via Twilio
        if not SKIP_PERSISTENT_SEND:
            tw = get_twilio_client()
            if tw:
                body_for_send = reply if len(reply) <= 1600 else reply[:1600] + "..."
                _twilio_send_and_log(
                    tw.messages.create,
                    body=body_for_send,
                    from_=TWILIO_WHATSAPP_FROM,
                    to=sender
                )

        logging.info("BG: finished text processing for %s", sender)

    except Exception:
        logging.exception("BG: unexpected failure processing text for %s", sender)



# ----------------- /whatsapp webhook -----------------
@app.route("/whatsapp", methods=["POST"])
def whatsapp_webhook():
    try:
        logging.info("Headers: %s", dict(request.headers))
        logging.info("Form: %s", request.form.to_dict())
    except Exception:
        pass

    public_url = None
    try:
        pt = globals().get("public_tunnel")
        if pt and getattr(pt, "public_url", None): public_url = pt.public_url
    except Exception:
        public_url = None
    full_url = (public_url.rstrip("/") + request.full_path) if public_url else request.url
    params = request.form.to_dict()
    signature = request.headers.get("X-Twilio-Signature", "")
    if not SKIP_TWILIO_VALIDATION:
        try:
            if not validator.validate(full_url, params, signature):
                logging.warning("Invalid Twilio signature")
                return ("", 403)
        except Exception:
            logging.exception("Validation raised")
            return ("", 403)
    else:
        logging.info("Skipping Twilio signature validation (dev mode)")

    incoming_msg = request.form.get("Body", "") or ""
    incoming_msg = incoming_msg.strip()
    sender = request.form.get("From")
    if not sender: return ("", 400)
    logging.info("IN | %s | %s", sender, incoming_msg)

    try: num_media = int(request.form.get("NumMedia", "0"))
    except Exception: num_media = 0

    # If voice note present -> acknowledge immediately & spawn background thread
    if num_media > 0:
        mtype = request.form.get("MediaContentType0", "")
        murl = request.form.get("MediaUrl0", "")
        if mtype and mtype.startswith("audio") and murl:
            logging.info("Incoming audio media from %s type=%s url=%s", sender, mtype, murl)
            resp = MessagingResponse()
            resp.message("Got your voice note — processing it now. I'll reply here when ready.")
            try:
                t = threading.Thread(target=_process_voice_and_send_async, args=(sender, murl, mtype), daemon=True)
                t.start()
            except Exception:
                logging.exception("Failed to spawn background thread; processing synchronously")
                result = handle_incoming_voice_media_process_only(sender, murl, mtype)
                if result.get("success"):
                    transcript = result.get("transcript", "")
                    answer = result.get("answer") or TRANSLATIONS["fallback"].get(result.get("lang") or "en")
                    short_ans = (answer[:1500] + "...") if len(answer) > 1500 else answer
                    resp = MessagingResponse(); resp.message(f"Q: {transcript}\n\nA: {short_ans}")
            last_seen[sender] = int(time.time())
            try: json.dump(last_seen, open(LAST_SEEN_PATH, "w"), ensure_ascii=False)
            except Exception: logging.exception("failed to persist last_seen after voice note")
            logging.info("Returning TwiML acknowledgement for audio webhook")
            return str(resp)

    # Rate limiting
    if not allowed_rate(sender):
        resp = MessagingResponse(); resp.message("Rate limit — try again shortly."); return str(resp)

    cmd = incoming_msg.lower().strip()

    # subscribe/unsubscribe
    if cmd in ("subscribe", "start", "join"):
        if sender not in subscribers:
            subscribers.append(sender); json.dump(subscribers, open(SUBSCRIBERS_PATH, "w"), ensure_ascii=False)
        last_seen[sender] = int(time.time()); json.dump(last_seen, open(LAST_SEEN_PATH, "w"), ensure_ascii=False)
        resp = MessagingResponse(); resp.message("✅ Subscribed to outbreak alerts. Send UNSUBSCRIBE to stop."); return str(resp)
    if cmd in ("unsubscribe", "stop", "leave"):
        if sender in subscribers:
            subscribers.remove(sender); json.dump(subscribers, open(SUBSCRIBERS_PATH, "w"), ensure_ascii=False)
        resp = MessagingResponse(); resp.message("You have been unsubscribed."); return str(resp)

    # check outbreaks commands
    if cmd in ("check outbreaks", "latest outbreaks", "check alerts", "outbreaks"):
        try: q = json.load(open(ALERTS_QUEUE, "r", encoding="utf-8"))
        except Exception: q = alerts_queue
        approved = [a for a in q if a.get("status") == "approved"]; pending = [a for a in q if a.get("status") == "pending"]
        items = (approved or pending)[:3]
        resp = MessagingResponse()
        if not items:
            resp.message("No recent outbreak alerts at the moment."); return str(resp)
        texts = []
        for a in items:
            snip = (a.get("summary") or "")[:240] + ("..." if len((a.get("summary") or "")) > 240 else "")
            texts.append(f"*{a.get('title')}*\n{snip}\n{a.get('published')}\n{a.get('link')}")
        resp.message("\n\n---\n\n".join(texts))
        last_seen[sender] = int(time.time()); json.dump(last_seen, open(LAST_SEEN_PATH, "w"), ensure_ascii=False)
        return str(resp)

    # language change via "LANGUAGE: X"
    if incoming_msg.lower().startswith("language:"):
        try:
            code = _lang_code_from_text(incoming_msg.split(":", 1)[1].strip())
            user_lang_prefs[sender] = code; save_prefs(user_lang_prefs)
            seen_users.add(sender); persist_seen_users()
            if code == "ta": reply = "✅ மொழி தமிழ் தேர்ந்தெடுக்கப்பட்டது."
            elif code == "hi": reply = "✅ भाषा हिंदी चुन ली गई है।"
            else: reply = "✅ Language set to English."
        except Exception:
            reply = "Couldn't set language. Send LANGUAGE: TAMIL / HINDI / ENGLISH."
        resp = MessagingResponse(); resp.message(reply)
        last_seen[sender] = int(time.time()); json.dump(last_seen, open(LAST_SEEN_PATH, "w"), ensure_ascii=False)
        return str(resp)

    # Normal text Q->A
      # ---------- Normal text: quick ack + background processing ----------
    # persist last_seen quickly
    try:
        last_seen[sender] = int(time.time())
        json.dump(last_seen, open(LAST_SEEN_PATH, "w"), ensure_ascii=False)
    except Exception:
        logging.exception("Failed updating last_seen for %s", sender)

    # quick ack to Twilio (so Twilio receives HTTP 200 fast)
    resp = MessagingResponse()
    resp.message("✅ Got your message — processing it now. I'll reply here when ready.")
    try:
        t = threading.Thread(target=_process_text_and_send_async, args=(sender, incoming_msg), daemon=True)
        t.start()
    except Exception:
        logging.exception("Failed to spawn background thread for text processing; attempting synchronous processing now")
        # fallback synchronous (slow) processing so the user still gets an answer
        try:
            answer, sources, conf = answer_with_sources(incoming_msg, user_lang=user_lang_prefs.get(sender) or detect_lang(incoming_msg), top_k=8, doc_threshold=0.22)
            short_ans = (answer[:1600] + "...") if answer and len(answer) > 1600 else (answer or TRANSLATIONS["fallback"]["en"])
            resp = MessagingResponse(); resp.message(short_ans)
        except Exception:
            logging.exception("Fallback synchronous processing failed")
            resp = MessagingResponse(); resp.message(TRANSLATIONS["fallback"]["en"])

    # Finally return the quick ack to Twilio
    logging.info("Webhook: acknowledged %s with quick reply", sender)
    return str(resp)


# ----------------- Admin endpoints -----------------
@app.route('/admin/list_alerts', methods=['GET'])
def admin_list_alerts():
    return json.dumps([a for a in alerts_queue if a.get('status') == 'pending'][:50], ensure_ascii=False)

@app.route('/admin/approve_alert', methods=['POST'])
def admin_approve_alert():
    data = request.get_json()
    aid = data.get('id'); msg = data.get('message')
    if not aid: return ("missing id", 400)
    for a in alerts_queue:
        if a['id'] == aid:
            a['status'] = 'approved'; a['approved_at'] = time.time(); a['approved_message'] = msg or (a.get('title') + '\n' + a.get('link', ''))
            json.dump(alerts_queue, open(ALERTS_QUEUE, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
            return ("approved", 200)
    return ("not found", 404)

def send_whatsapp_freeform(to, body):
    try:
        tw = get_twilio_client()
        if not tw:
            logging.warning("Twilio client not configured; cannot send freeform")
            return False
        msg = _twilio_send_and_log(tw.messages.create, body=body, from_=TWILIO_WHATSAPP_FROM, to=to)
        return bool(msg)
    except Exception:
        logging.exception("send freeform failed")
        return False

# ----------------- Scheduler -----------------
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

scheduler = BackgroundScheduler()
AUTO_BROADCAST = os.environ.get("HEALTHBOT_AUTO_BROADCAST", "0") == "1"

def job_fetch_and_queue():
    try:
        events = fetch_and_detect_events()
        new = dedupe_and_queue(events)
        logging.info("Scheduler queued %s new events", len(new))
    except Exception:
        logging.exception("fetch_and_queue failed")

def job_broadcast():
    try:
        if AUTO_BROADCAST:
            # implement broadcast logic if desired
            logging.info("Auto broadcast done")
        else:
            logging.info("AUTO_BROADCAST disabled")
    except Exception:
        logging.exception("broadcast job failed")

scheduler.add_job(job_fetch_and_queue, trigger=IntervalTrigger(hours=1), id="fetch_and_queue", replace_existing=True)
scheduler.add_job(job_broadcast, trigger=IntervalTrigger(minutes=30), id="broadcast_approved", replace_existing=True)
scheduler.start()
atexit.register(lambda: scheduler.shutdown(wait=False))
logging.info("Scheduler started (background)")

# ----------------- ngrok helper (local/Colab only) -----------------
def _maybe_start_ngrok():
    port_env = os.environ.get("PORT")
    if port_env:
        return None
    try:
        from pyngrok import ngrok, conf as ngrok_conf  # lazy import
        if NGROK_AUTHTOKEN and NGROK_AUTHTOKEN.strip() and NGROK_AUTHTOKEN != "<YOUR_NGROK_AUTHTOKEN_OR_LEAVE_EMPTY>":
            try:
                ngrok_conf.get_default().auth_token = NGROK_AUTHTOKEN
            except Exception:
                os.system(f"ngrok config add-authtoken {NGROK_AUTHTOKEN}")
        try:
            ngrok.kill()
        except Exception:
            pass
        public_tunnel = ngrok.connect(5000)
        globals()["public_tunnel"] = public_tunnel
        print("ngrok public url:", public_tunnel.public_url)
        print("Set Twilio webhook to:", public_tunnel.public_url + "/whatsapp")
        return public_tunnel
    except Exception:
        logging.exception("ngrok not started")
        return None

# ----------------- Helper: parse language codes from "LANGUAGE:" -----------------
def _lang_code_from_text(txt):
    if not txt: return "en"
    t = txt.strip().lower()
    if "tamil" in t or t.startswith("ta") or "தமிழ்" in t: return "ta"
    if "hindi" in t or t.startswith("hi") or "हिन्दी" in t or "हिंदी" in t: return "hi"
    if "english" in t or t.startswith("en") or "eng" in t: return "en"
    try: d = detect_lang(t); return d if d in LANG_MAP else "en"
    except Exception: return "en"

# ----------------- Start (only if executed directly) -----------------
if __name__ == '__main__':
    # Start ngrok for local dev if desired
    _maybe_start_ngrok()

    # Use PORT from environment if present (platforms like Render provide $PORT)
    port = int(os.environ.get("PORT", 5000))
    host = "0.0.0.0"

    logging.info("Starting Flask app on %s:%d", host, port)
    try:
        # If running directly (not via gunicorn), use Flask's built-in server.
        app.run(host=host, port=port)
    except Exception:
        logging.exception("Failed to start Flask")
        raise
