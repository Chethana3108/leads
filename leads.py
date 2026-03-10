from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import requests
from bs4 import BeautifulSoup
import json, time, os, base64, io
from datetime import datetime, date
from gtts import gTTS
from threading import Threada
import re
import torch
import numpy as np
import faiss
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import pdfplumber
import uvicorn
from dotenv import load_dotenv
 

load_dotenv()
 

BASE_URL     = os.getenv("BASE_URL", "https://central.crm-doctor.com/crmsites/vaishnavitandelcrm510/")
LOGIN_URL    = BASE_URL + "index.php"
LIST_API_URL = BASE_URL + "modules/Mobile/v1/listModuleRecords"
LEAD_DETAIL_URL = (
    BASE_URL
    + "index.php?module=Leads&view=Detail&record={}"
    + "&mode=showDetailViewByMode&requestMode=full"
    + "&tab_label=Lead%20Details&app=MARKETING"
)
 
USERNAME     = os.getenv("CRM_USERNAME", "vaishnavitandelcrm510")
PASSWORD     = os.getenv("CRM_PASSWORD", "123456")
ACCESS_TOKEN = os.getenv("ACCESS_TOKEN", "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyaWQiOiI0NiJ9.-qFr9dbFAekc_h4cGMAYpddOXM_W_0P8uqpK8HbBS_s")
USER_ID      = os.getenv("USER_ID", "46")
MODULE       = os.getenv("MODULE", "Leads")
 
CACHE_FILE            = "crm_leads_cache.json"
SYNC_INTERVAL_SECONDS = 60
HOST                  = "0.0.0.0"
PORT                  = int(os.getenv("PORT", 8000))
 

CHUNK_SIZE    = 400
CHUNK_OVERLAP = 80
TOP_K_CHUNKS  = 4
EMBED_MODEL   = "all-MiniLM-L6-v2"
 

app = FastAPI(title="CRM + PDF Bot API", version="2.0.0")
 

LEADS: list = []
model        = None
tokenizer    = None
embed_model  = None
faiss_index  = None
pdf_chunks: list[str] = []
pdf_name: str = ""
 
 
class AskRequest(BaseModel):
    question: str = ""
 

def load_llm():
    global model, tokenizer
    MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
    print("Loading Qwen LLM...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        model.eval()
        print("Qwen LLM Loaded")
    except Exception as e:
        print(f"LLM load failed: {e}")
        model, tokenizer = None, None
 
 
def load_embed_model():
    global embed_model
    print("Loading Embedding Model...")
    try:
        embed_model = SentenceTransformer(EMBED_MODEL)
        print(f"Embedding Model Loaded: {EMBED_MODEL}")
    except Exception as e:
        print(f"Embedding model load failed: {e}")
 
 
def qwen_generate(prompt: str, max_new_tokens: int = 300) -> str:
    if not model or not tokenizer:
        return ""
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        out    = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    except Exception as e:
        print(f"Qwen generate error: {e}")
        return ""
 
 
def qwen_fallback(question: str) -> dict:
    prompt = (
        "Return intent JSON only. No explanation.\n"
        "Allowed intents: TOTAL_LEADS, TODAY_LEADS, MONTH_LEADS, "
        "LEAD_DETAILS (firstname), PRIORITY_STATS, PRIORITY_SHOW (level)\n"
        f"Question: {question}\nJSON:"
    )
    raw   = qwen_generate(prompt, max_new_tokens=60)
    match = re.search(r"\{.*?\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass
    return {"intent": "UNKNOWN"}
 

def extract_text_from_pdf(file_bytes: bytes) -> str:
    text = ""
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"PDF extraction error: {e}")
    return text.strip()
 
 
def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    chunks, start = [], 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end].strip())
        start += size - overlap
    return [c for c in chunks if c]
 
 
def build_faiss_index(chunks: list[str]):
    global faiss_index, pdf_chunks
    if not embed_model:
        raise RuntimeError("Embedding model not loaded.")
    pdf_chunks = chunks
    embeddings = embed_model.encode(chunks, show_progress_bar=True, convert_to_numpy=True)
    dim        = embeddings.shape[1]
    index      = faiss.IndexFlatL2(dim)
    index.add(embeddings.astype(np.float32))
    faiss_index = index
    print(f"FAISS index built: {len(chunks)} chunks, dim={dim}")
 
 
def retrieve_chunks(query: str, k: int = TOP_K_CHUNKS) -> list[str]:
    if not faiss_index or not embed_model or not pdf_chunks:
        return []
    q_emb      = embed_model.encode([query], convert_to_numpy=True).astype(np.float32)
    _, indices = faiss_index.search(q_emb, k)
    return [pdf_chunks[i] for i in indices[0] if i < len(pdf_chunks)]
 
 
def answer_from_pdf(question: str, lang: str) -> str:
    chunks = retrieve_chunks(question)
    if not chunks:
        return ("No relevant content found in the PDF."
                if lang == "en" else "PDF में कोई प्रासंगिक जानकारी नहीं मिली।")
    context = "\n---\n".join(chunks)
    prompt  = (
        f"Answer the question based ONLY on the context below.\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
    answer = qwen_generate(prompt, max_new_tokens=300)
    return answer if answer else ("Could not generate an answer."
                                  if lang == "en" else "उत्तर उत्पन्न नहीं हो सका।")
 
 
CRM_KEYWORDS = {
    "lead", "leads", "total", "today", "priority", "high", "medium", "low",
    "status", "mobile", "contact", "details", "show", "count", "all",
    "sync", "crm", "आज", "कुल", "लीड", "दिखाओ", "prathmikta",
}
 
def detect_domain(question: str) -> str:
    q_lower   = question.lower()
    words     = set(re.findall(r"[a-zA-Z\u0900-\u097F]+", q_lower))
    crm_hit   = bool(words & CRM_KEYWORDS)
    pdf_ready = faiss_index is not None and bool(pdf_chunks)
 
    if crm_hit and not pdf_ready:
        return "CRM"
    if crm_hit and pdf_ready:
        return "BOTH"
    if pdf_ready:
        return "PDF"
    return "CRM"
 

def detect_language(text: str) -> str:
    hindi_chars = sum(1 for c in text if "\u0900" <= c <= "\u097F")
    return "hi" if len(text) > 0 and (hindi_chars / len(text)) > 0.2 else "en"
 
 
def text_to_audio_base64(text: str, lang: str) -> str:
    try:
        buf = io.BytesIO()
        gTTS(text=text, lang=lang).write_to_fp(buf)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode()
    except Exception as e:
        print(f"TTS error: {e}")
        return ""
 
 
def parse_created_time(lead: dict):
    raw     = lead.get("created_time") or lead.get("createdtime") or ""
    formats = ["%d-%m-%Y %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%d-%m-%Y"]
    for fmt in formats:
        try:
            return datetime.strptime(raw.strip(), fmt)
        except (ValueError, AttributeError):
            continue
    return None
 
 
def clean_lead(l: dict) -> dict:
    return {
        "leadid":       l.get("leadid"),
        "firstname":    l.get("firstname"),
        "lastname":     l.get("lastname"),
        "mobile":       l.get("mobile"),
        "lead_status":  l.get("lead_status"),
        "priority":     l.get("priority"),
        "created_time": l.get("created_time") or l.get("createdtime"),
    }
 

def crm_login():
    s = requests.Session()
    try:
        r    = s.get(LOGIN_URL, timeout=15)
        soup = BeautifulSoup(r.text, "html.parser")
        csrf = soup.find("input", {"name": "__vtrftk"})
        if csrf:
            s.post(LOGIN_URL, data={
                "__vtrftk": csrf.get("value"),
                "username":  USERNAME,
                "password":  PASSWORD,
                "module":    "Users",
                "action":    "Login",
            }, timeout=15)
            return s
    except Exception as e:
        print(f"CRM login error: {e}")
    return None
 
 
def extract_details(session, lead_id: str) -> dict:
    try:
        r    = session.get(LEAD_DETAIL_URL.format(lead_id), timeout=15)
        soup = BeautifulSoup(r.text, "html.parser")
 
        def get_val(id_):
            td = soup.find("td", id=id_)
            if td:
                span = td.find("span", class_="value")
                if span:
                    return span.get_text(strip=True)
            return None
 
        prio = get_val("Leads_detailView_fieldValue_priority")
        if not prio:
            try:
                label = soup.find("td", class_="fieldLabel",
                                  string=re.compile(r"Priority", re.I))
                if label:
                    val_td = label.find_next_sibling("td")
                    if val_td:
                        prio = val_td.get_text(strip=True)
            except Exception:
                pass
 
        return {
            "lead_status":  get_val("Leads_detailView_fieldValue_leadstatus"),
            "priority":     prio,
            "created_time": get_val("Leads_detailView_fieldValue_createdtime"),
        }
    except Exception as e:
        print(f"extract_details error for {lead_id}: {e}")
        return {}
 
 
def fetch_leads(force: bool = False) -> list:
    if not force and os.path.exists(CACHE_FILE):
        print("Loading leads from cache...")
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
 
    print("Syncing CRM...")
    session = crm_login()
    if not session:
        print("Login failed.")
        return []
 
    leads, page = [], 1
    while True:
        try:
            r = requests.post(LIST_API_URL, data={
                "access_token": ACCESS_TOKEN,
                "useruniqueid": USER_ID,
                "module":       MODULE,
                "page":         page,
            }, timeout=20)
            result = r.json().get("result", {})
            recs   = result.get("records", [])
            if not recs:
                break
            print(f"   Page {page} ({len(recs)} records)...")
            for l in recs:
                l.update(extract_details(session, l["leadid"]))
                leads.append(l)
                time.sleep(0.1)
            if not result.get("moreRecords"):
                break
            page += 1
        except Exception as e:
            print(f"Fetch error on page {page}: {e}")
            break
 
    with open(CACHE_FILE, "w") as f:
        json.dump(leads, f)
    print(f"Synced {len(leads)} leads.")
    return leads
 
 
def auto_sync():
    global LEADS
    while True:
        time.sleep(SYNC_INTERVAL_SECONDS)
        print(f"Auto-syncing... (every {SYNC_INTERVAL_SECONDS}s)")
        new = fetch_leads(force=True)
        if new:
            LEADS = new
            print(f"LEADS updated: {len(LEADS)} total.")
 
 

@app.on_event("startup")
async def startup_event():
    global LEADS
    load_llm()
    load_embed_model()
    LEADS = fetch_leads(force=True)
    Thread(target=auto_sync, daemon=True).start()
    print("CRM + PDF Bot server is ready!")
    print(f"Swagger Docs: http://localhost:{PORT}/docs")
 

def resolve_crm_intent(q: str) -> dict:
    q_lower = q.lower().strip()
 
    if "today" in q_lower or "आज" in q_lower:
        return {"intent": "TODAY_LEADS"}
    if "total" in q_lower or "all" in q_lower or "कुल" in q_lower:
        return {"intent": "TOTAL_LEADS"}
    if "priority" in q_lower or "prathmikta" in q_lower or "importance" in q_lower:
        if any(x in q_lower for x in ["count", "stats", "breakdown", "numbers"]):
            return {"intent": "PRIORITY_STATS"}
        for level in ["high", "medium", "low"]:
            if level in q_lower:
                return {"intent": "PRIORITY_SHOW", "level": level.capitalize()}
        return {"intent": "PRIORITY_STATS"}
    for level in ["high", "medium", "low"]:
        if level in q_lower and any(kw in q_lower for kw in ["lead", "show", "display"]):
            return {"intent": "PRIORITY_SHOW", "level": level.capitalize()}
    if "details" in q_lower or "show" in q_lower or "दिखाओ" in q_lower:
        words     = q_lower.split()
        firstname = words[-1].strip("?.!")
        return {"intent": "LEAD_DETAILS", "firstname": firstname}
 
    return qwen_fallback(q)
 
 
def handle_crm_intent(intent_data: dict, lang: str) -> tuple[str, list]:
    intent       = intent_data.get("intent")
    leads_result = []
    text         = ("Sorry, I didn't understand."
                    if lang == "en" else "माफ करें, मैं समझ नहीं पाया।")
 
    if intent == "TOTAL_LEADS":
        leads_result = LEADS
        text = (f"Total leads: {len(leads_result)}"
                if lang == "en" else f"कुल {len(leads_result)} लीड हैं।")
    elif intent == "TODAY_LEADS":
        today        = date.today()
        leads_result = [l for l in LEADS if (dt := parse_created_time(l)) and dt.date() == today]
        text = (f"Leads today: {len(leads_result)}"
                if lang == "en" else f"आज {len(leads_result)} लीड हैं।")
    elif intent == "LEAD_DETAILS":
        name         = intent_data.get("firstname", "").lower()
        leads_result = [l for l in LEADS if l.get("firstname", "").lower() == name]
        text = (f"Found {len(leads_result)} leads for '{name}'."
                if lang == "en" else f"'{name}' के लिए {len(leads_result)} लीड मिले।")
    elif intent == "PRIORITY_STATS":
        counts = {}
        for l in LEADS:
            p = (l.get("priority") or "Unknown").strip().capitalize()
            counts[p] = counts.get(p, 0) + 1
        breakdown = ", ".join(f"{k}: {v}" for k, v in sorted(counts.items()))
        text = (f"Priority Breakdown: {breakdown}"
                if lang == "en" else f"प्राथमिकता विवरण: {breakdown}")
    elif intent == "PRIORITY_SHOW":
        level        = intent_data.get("level", "High")
        leads_result = [l for l in LEADS if (l.get("priority") or "").strip().lower() == level.lower()]
        text = (f"Found {len(leads_result)} {level} priority leads."
                if lang == "en" else f"{len(leads_result)} {level} प्राथमिकता वाली लीड मिलीं।")
 
    return text, leads_result
 
 
@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    global pdf_name
    if not file.filename.lower().endswith(".pdf"):
        return JSONResponse(status_code=400,
                            content={"success": False, "message": "Only PDF files are accepted."})
    if not embed_model:
        return JSONResponse(status_code=503,
                            content={"success": False, "message": "Embedding model not loaded yet."})
    try:
        raw_bytes = await file.read()
        pdf_name  = file.filename
        text      = extract_text_from_pdf(raw_bytes)
        if not text:
            return JSONResponse(status_code=422,
                                content={"success": False, "message": "Could not extract text from PDF."})
        chunks = chunk_text(text)
        build_faiss_index(chunks)
        return {
            "success":      True,
            "filename":     pdf_name,
            "total_chunks": len(chunks),
            "message":      f"PDF '{pdf_name}' indexed with {len(chunks)} chunks. Ready to answer questions!",
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "message": str(e)})
 
 
@app.post("/ask")
async def ask(body: AskRequest):
    q = body.question.strip()
    if not q:
        return {"success": False, "text": "No question provided.",
                "source": None, "count": 0, "leads": [], "audio_base64": ""}
 
    lang   = detect_language(q)
    domain = detect_domain(q)
    print(f"Question: {q} | Lang: {lang} | Domain: {domain}")
 
    leads_result = []
    text, source = "", ""
 
    if domain == "CRM":
        intent_data        = resolve_crm_intent(q)
        text, leads_result = handle_crm_intent(intent_data, lang)
        source             = "CRM"
        print(f"   → CRM intent: {intent_data}")
 
    elif domain == "PDF":
        text   = answer_from_pdf(q, lang)
        source = "PDF"
        print(f"   → PDF RAG")
 
    elif domain == "BOTH":
        intent_data            = resolve_crm_intent(q)
        crm_text, leads_result = handle_crm_intent(intent_data, lang)
        if leads_result or intent_data.get("intent") in ("PRIORITY_STATS", "TOTAL_LEADS", "TODAY_LEADS"):
            text, source = crm_text, "CRM"
            print(f"   → BOTH → resolved to CRM ({intent_data.get('intent')})")
        else:
            text, source = answer_from_pdf(q, lang), "PDF"
            print(f"   → BOTH → resolved to PDF RAG")
 
    return {
        "success":      True,
        "source":       source,
        "text":         text,
        "count":        len(leads_result),
        "leads":        [clean_lead(l) for l in leads_result],
        "audio_base64": text_to_audio_base64(text, lang),
    }
 
 
@app.post("/refresh")
async def refresh():
    global LEADS
    LEADS = fetch_leads(force=True)
    return {"success": True, "count": len(LEADS), "message": "Leads refreshed."}
 
 
@app.get("/status")
async def status():
    return {
        "success":               True,
        "total_leads_in_memory": len(LEADS),
        "sync_interval_seconds": SYNC_INTERVAL_SECONDS,
        "llm_loaded":            model is not None,
        "embed_model_loaded":    embed_model is not None,
        "pdf_loaded":            faiss_index is not None,
        "pdf_name":              pdf_name or None,
        "pdf_chunks":            len(pdf_chunks),
    }
 

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=6006,
        reload=False, 
        log_level="info",
    )
