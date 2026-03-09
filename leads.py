from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.concurrency import run_in_threadpool

import requests
from bs4 import BeautifulSoup

import json
import time
import os
import base64
import io
import re
import asyncio

from datetime import datetime, date
from gtts import gTTS
from threading import Thread

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


app = FastAPI()


class Question(BaseModel):
    question: str




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

ACCESS_TOKEN = "YOUR_ACCESS_TOKEN"
USER_ID = "46"

MODULE = "Leads"

CACHE_FILE = "crm_leads_cache.json"

SYNC_INTERVAL_SECONDS = 60




torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True




MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

print("Loading Mistral Model...")

try:

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.float16
    )

    model.eval()

    print("Mistral Model Loaded")

except Exception as e:

    print("Model loading failed:", e)

    model = None
    tokenizer = None




def mistral_fallback(question):

    if not model:
        return {"intent": "UNKNOWN"}

    prompt = f"""
You are an intent classifier.

Return JSON only.

Allowed intents:
TOTAL_LEADS
TODAY_LEADS
LEAD_DETAILS
PRIORITY_STATS
PRIORITY_SHOW

Question: {question}
"""

    try:

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.inference_mode():

            output = model.generate(
                **inputs,
                max_new_tokens=60,
                do_sample=False
            )

        text = tokenizer.decode(output[0], skip_special_tokens=True)

        match = re.search(r"\{.*?\}", text, re.DOTALL)

        if match:
            return json.loads(match.group())

    except Exception as e:

        print("AI error:", e)

    return {"intent": "UNKNOWN"}



def detect_language(text):

    hindi_chars = sum(
        1 for c in text if "\u0900" <= c <= "\u097F"
    )

    if len(text) > 0 and (hindi_chars / len(text)) > 0.2:
        return "hi"

    return "en"


def text_to_audio_base64(text, lang):

    try:

        buf = io.BytesIO()

        gTTS(text=text, lang=lang).write_to_fp(buf)

        buf.seek(0)

        return base64.b64encode(buf.read()).decode()

    except Exception as e:

        print("TTS error:", e)

        return ""


def parse_created_time(lead):

    raw = lead.get("created_time") or lead.get("createdtime") or ""

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
            pass

    return None


def clean_lead(l):

    return {
        "leadid": l.get("leadid"),
        "firstname": l.get("firstname"),
        "lastname": l.get("lastname"),
        "mobile": l.get("mobile"),
        "lead_status": l.get("lead_status"),
        "priority": l.get("priority"),
        "created_time": l.get("created_time") or l.get("createdtime"),
    }



def crm_login():

    s = requests.Session()

    try:

        r = s.get(LOGIN_URL, timeout=15)

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
                },
                timeout=15,
            )

            return s

    except Exception as e:
        print("Login error:", e)

    return None



def extract_details(session, lead_id):

    try:

        r = session.get(LEAD_DETAIL_URL.format(lead_id), timeout=15)

        soup = BeautifulSoup(r.text, "html.parser")

        def get_val(id_):

            td = soup.find("td", id=id_)

            if td:

                span = td.find("span", class_="value")

                if span:
                    return span.get_text(strip=True)

            return None

        return {
            "lead_status": get_val("Leads_detailView_fieldValue_leadstatus"),
            "priority": get_val("Leads_detailView_fieldValue_priority"),
            "created_time": get_val("Leads_detailView_fieldValue_createdtime"),
        }

    except Exception as e:
        print("extract error:", e)

        return {}



def fetch_leads(force=False):

    if not force and os.path.exists(CACHE_FILE):

        with open(CACHE_FILE, "r") as f:
            return json.load(f)

    session = crm_login()

    if not session:
        return []

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
            },
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

    return leads



LEADS = fetch_leads(force=True)



def auto_sync():

    global LEADS

    while True:

        time.sleep(SYNC_INTERVAL_SECONDS)

        new = fetch_leads(force=True)

        if new:
            LEADS = new


Thread(target=auto_sync, daemon=True).start()



def resolve_intent(q):

    q_lower = q.lower().strip()

    if "today" in q_lower:
        return {"intent": "TODAY_LEADS"}

    if "total" in q_lower or "all" in q_lower:
        return {"intent": "TOTAL_LEADS"}

    if "priority" in q_lower:

        for level in ["high", "medium", "low"]:

            if level in q_lower:
                return {
                    "intent": "PRIORITY_SHOW",
                    "level": level.capitalize(),
                }

        return {"intent": "PRIORITY_STATS"}

    if "details" in q_lower or "show" in q_lower:

        firstname = q_lower.split()[-1]

        return {
            "intent": "LEAD_DETAILS",
            "firstname": firstname,
        }

    return mistral_fallback(q)



@app.post("/ask")
async def ask(data: Question):

    q = data.question.strip()

    if not q:

        return {
            "success": False,
            "text": "No question provided",
            "count": 0,
            "leads": [],
            "audio_base64": "",
        }

    lang = detect_language(q)

    intent_data = await asyncio.wait_for(
        run_in_threadpool(resolve_intent, q),
        timeout=15
    )

    intent = intent_data.get("intent")

    leads_result = []

    text = "Sorry, I didn't understand."

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

        name = intent_data.get("firstname", "").lower()

        leads_result = [
            l for l in LEADS
            if l.get("firstname", "").lower() == name
        ]

        text = f"Found {len(leads_result)} leads for '{name}'."

    elif intent == "PRIORITY_STATS":

        counts = {}

        for l in LEADS:

            p = (l.get("priority") or "Unknown").capitalize()

            counts[p] = counts.get(p, 0) + 1

        breakdown = ", ".join(f"{k}: {v}" for k, v in counts.items())

        text = f"Priority Breakdown: {breakdown}"

    elif intent == "PRIORITY_SHOW":

        level = intent_data.get("level", "High")

        leads_result = [
            l for l in LEADS
            if (l.get("priority") or "").lower() == level.lower()
        ]

        text = f"Found {len(leads_result)} {level} priority leads."

    audio = await run_in_threadpool(text_to_audio_base64, text, lang)

    return {
        "success": True,
        "text": text,
        "count": len(leads_result),
        "leads": [clean_lead(l) for l in leads_result],
        "audio_base64": audio,
    }




@app.get("/health")
def health():

    return {
        "status": "running",
        "total_leads": len(LEADS),
        "model_loaded": model is not None
    }



if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=6006
    )