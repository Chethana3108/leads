from fastapi import FastAPI
from pydantic import BaseModel
import requests
from bs4 import BeautifulSoup
import json, time, os, base64, io, re
from datetime import datetime, date
from gtts import gTTS
from threading import Thread

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

import uvicorn


BASE_URL = "https://central.crm-doctor.com/crmsites/vaishnavitandelcrm510/"
LOGIN_URL = BASE_URL + "index.php"

LIST_API_URL = BASE_URL + "modules/Mobile/v1/listModuleRecords"

LEAD_DETAIL_URL = (
    BASE_URL
    + "index.php?module=Leads&view=Detail&record={}"
    + "&mode=showDetailViewByMode&requestMode=full"
    + "&tab_label=Lead%20Details&app=MARKETING"
)

USERNAME = "vaishnavitandelcrm510"
PASSWORD = "123456"

ACCESS_TOKEN = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyaWQiOiI0NiJ9.-qFr9dbFAekc_h4cGMAYpddOXM_W_0P8uqpK8HbBS_s"
USER_ID = "46"
MODULE = "Leads"

CACHE_FILE = "crm_leads_cache.json"
SYNC_INTERVAL_SECONDS = 60


app = FastAPI(title="CRM AI Assistant API")

LEADS = []
model = None
tokenizer = None

class AskRequest(BaseModel):
    question: str


def load_model():
    global model, tokenizer

    MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

    print("Loading AI Model...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )

    model.eval()

    print("Model Loaded")


def qwen_fallback(question):

    if not model or not tokenizer:
        return {"intent": "UNKNOWN"}

    prompt = (
        "Return intent JSON only.\n"
        "Allowed intents: TOTAL_LEADS, TODAY_LEADS, "
        "LEAD_DETAILS, PRIORITY_STATS, PRIORITY_SHOW\n"
        f"Question: {question}"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    out = model.generate(
        **inputs,
        max_new_tokens=60,
        do_sample=False
    )

    text = tokenizer.decode(out[0], skip_special_tokens=True)

    match = re.search(r"\{.*?\}", text, re.DOTALL)

    if match:
        return json.loads(match.group())

    return {"intent": "UNKNOWN"}


def detect_language(text):

    hindi_chars = sum(1 for c in text if "\u0900" <= c <= "\u097F")

    if len(text) > 0 and (hindi_chars / len(text)) > 0.2:
        return "hi"

    return "en"

def text_to_audio_base64(text, lang):

    try:
        buf = io.BytesIO()
        gTTS(text=text, lang=lang).write_to_fp(buf)
        buf.seek(0)

        return base64.b64encode(buf.read()).decode()

    except:
        return ""


def parse_created_time(lead):

    raw = lead.get("created_time") or ""

    formats = [
        "%d-%m-%Y %H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
        "%d-%m-%Y",
    ]

    for fmt in formats:
        try:
            return datetime.strptime(raw.strip(), fmt)
        except:
            continue

    return None


def clean_lead(l):

    return {
        "leadid": l.get("leadid"),
        "firstname": l.get("firstname"),
        "lastname": l.get("lastname"),
        "mobile": l.get("mobile"),
        "lead_status": l.get("lead_status"),
        "priority": l.get("priority"),
        "created_time": l.get("created_time"),
    }


def crm_login():

    s = requests.Session()

    r = s.get(LOGIN_URL)

    soup = BeautifulSoup(r.text, "html.parser")

    csrf = soup.find("input", {"name": "__vtrftk"})

    if csrf:
        s.post(
            LOGIN_URL,
            data={
                "__vtrftk": csrf.get("value"),
                "username": USERNAME,
                "password": PASSWORD,
                "module": "Users",
                "action": "Login",
            }
        )

        return s

    return None


def extract_details(session, lead_id):

    r = session.get(LEAD_DETAIL_URL.format(lead_id))

    soup = BeautifulSoup(r.text, "html.parser")

    def get_val(id_):
        td = soup.find("td", id=id_)
        if td:
            span = td.find("span", class_="value")
            if span:
                return span.get_text(strip=True)

    return {
        "lead_status": get_val("Leads_detailView_fieldValue_leadstatus"),
        "priority": get_val("Leads_detailView_fieldValue_priority"),
        "created_time": get_val("Leads_detailView_fieldValue_createdtime"),
    }


def fetch_leads(force=False):

    if not force and os.path.exists(CACHE_FILE):
        with open(CACHE_FILE) as f:
            return json.load(f)

    print("Syncing CRM...")

    session = crm_login()

    leads = []
    page = 1

    while True:

        r = requests.post(
            LIST_API_URL,
            data={
                "access_token": ACCESS_TOKEN,
                "useruniqueid": USER_ID,
                "module": MODULE,
                "page": page,
            }
        )

        result = r.json().get("result", {})
        recs = result.get("records", [])

        if not recs:
            break

        for l in recs:

            extra = extract_details(session, l["leadid"])

            l.update(extra)

            leads.append(l)

        if not result.get("moreRecords"):
            break

        page += 1

    with open(CACHE_FILE, "w") as f:
        json.dump(leads, f)

    print("Synced Leads:", len(leads))

    return leads


def auto_sync():

    global LEADS

    while True:

        time.sleep(SYNC_INTERVAL_SECONDS)

        new = fetch_leads(force=True)

        if new:
            LEADS = new

            print("Leads updated:", len(LEADS))


@app.on_event("startup")
def startup():

    global LEADS

    load_model()

    LEADS = fetch_leads(force=True)

    Thread(target=auto_sync, daemon=True).start()

    print("Server Ready")


def resolve_intent(q):

    q_lower = q.lower()

    if "today" in q_lower:
        return {"intent": "TODAY_LEADS"}

    if "total" in q_lower or "all" in q_lower:
        return {"intent": "TOTAL_LEADS"}

    if "priority" in q_lower:

        if "high" in q_lower:
            return {"intent": "PRIORITY_SHOW", "level": "High"}

        if "medium" in q_lower:
            return {"intent": "PRIORITY_SHOW", "level": "Medium"}

        if "low" in q_lower:
            return {"intent": "PRIORITY_SHOW", "level": "Low"}

        return {"intent": "PRIORITY_STATS"}

    if "details" in q_lower or "show" in q_lower:

        name = q_lower.split()[-1]

        return {"intent": "LEAD_DETAILS", "firstname": name}

    return qwen_fallback(q)


@app.post("/ask")
def ask(body: AskRequest):

    q = body.question.strip()

    lang = detect_language(q)

    intent_data = resolve_intent(q)

    intent = intent_data.get("intent")

    leads_result = []

    text = "Sorry I didn't understand."

    if intent == "TOTAL_LEADS":

        leads_result = LEADS

        text = f"Total leads: {len(leads_result)}"

    elif intent == "TODAY_LEADS":

        today = date.today()

        leads_result = [
            l for l in LEADS
            if (dt := parse_created_time(l)) and dt.date() == today
        ]

        text = f"Leads today: {len(leads_result)}"

    elif intent == "LEAD_DETAILS":

        name = intent_data.get("firstname", "")

        leads_result = [
            l for l in LEADS
            if l.get("firstname", "").lower() == name
        ]

        text = f"Found {len(leads_result)} leads for {name}"

    elif intent == "PRIORITY_SHOW":

        level = intent_data.get("level")

        leads_result = [
            l for l in LEADS
            if (l.get("priority") or "").lower() == level.lower()
        ]

        text = f"{len(leads_result)} {level} priority leads found"

    elif intent == "PRIORITY_STATS":

        counts = {}

        for l in LEADS:
            p = (l.get("priority") or "Unknown").capitalize()
            counts[p] = counts.get(p, 0) + 1

        text = f"Priority breakdown: {counts}"

    return {
        "success": True,
        "text": text,
        "count": len(leads_result),
        "leads": [clean_lead(l) for l in leads_result],
        "audio_base64": text_to_audio_base64(text, lang),
    }


@app.post("/refresh")
def refresh():

    global LEADS

    LEADS = fetch_leads(force=True)

    return {"success": True, "count": len(LEADS)}


@app.get("/status")
def status():

    return {
        "success": True,
        "total_leads": len(LEADS),
        "model_loaded": model is not None
    }


if __name__ == "__main__":

    uvicorn.run(
        "crm_bot:app",
        host="0.0.0.0",
        port=6006
    )