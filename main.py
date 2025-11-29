import re
from pathlib import Path
from urllib.parse import urlparse

import joblib
import pandas as pd
from fastapi import FastAPI, Request, Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# ============================================================
# PATHS
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

tfidf = joblib.load(MODEL_DIR / "tfidf.pkl")
text_model = joblib.load(MODEL_DIR / "text_model.pkl")
url_model = joblib.load(MODEL_DIR / "url_model.pkl")

# ============================================================
# FASTAPI
# ============================================================
app = FastAPI()

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
static_dir = BASE_DIR / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# ============================================================
# SAFE DOMAIN CHECKERS
# ============================================================
SAFE_BASE_DOMAINS = {
    "google.com","gmail.com","youtube.com","google.co.in",
    "microsoft.com","office.com","live.com","outlook.com",
    "apple.com","icloud.com","android.com",

    "amazon.in","amazon.com","flipkart.com",
    "swiggy.com","zomato.com","bigbasket.com",

    "facebook.com","instagram.com","whatsapp.com","twitter.com",
    "x.com","linkedin.com","reddit.com","pinterest.com",

    "paytm.com","phonepe.com","bharatpe.com","googlepay.com",
    "razorpay.com","billdesk.com","ccavenue.com",

    "icicibank.com","hdfcbank.com","sbi.co.in","axisbank.com",
    "kotak.com","bankofbaroda.in","canarabank.com",

    "gov.in","nic.in","india.gov.in","incometax.gov.in",
    "uidai.gov.in","passportindia.gov.in",

    "airtel.in","jio.com","vi.in",

    "chatgpt.com","openai.com","api.openai.com",
    "vercel.app","onrender.com","github.com","github.io",
}

VALID_TLDS = {
    "com","org","net","gov","in","co","edu","info","app","io","ai",
    "store","shop","tech","dev","me","us","uk","ca","au","de","fr",
    "club","link","live","online","site","blog",
}

BAD_TLDS = {"xyz","top","zip","ml","ga","tk","ru","cn"}

def is_safe_domain(domain):
    if not domain: return False
    domain = domain.lower()
    return any(domain == b or domain.endswith("." + b) for b in SAFE_BASE_DOMAINS)

def is_auto_safe_domain(domain):
    if not domain: return False
    domain = domain.lower()
    if domain.startswith(("localhost", "127.0.0.1")):
        return True
    if re.match(r"^192\.168\.\d+\.\d+$", domain): return True
    if re.match(r"^10\.\d+\.\d+\.\d+$", domain): return True
    if re.match(r"^172\.(1[6-9]|2\d|3[0-1])\.\d+\.\d+$", domain): return True
    if "jupyter" in domain or "notebook" in domain:
        return True
    if domain.endswith(".local"):
        return True
    return False

def is_valid_safe_domain(domain):
    if not domain or "." not in domain:
        return False
    labels = domain.split(".")
    if labels[-1] not in VALID_TLDS:
        return False
    for l in labels:
        if not re.match(r"^[a-z0-9-]{1,63}$", l): return False
        if l.startswith("-") or l.endswith("-"): return False
    return len(domain) <= 253

def is_suspicious_domain(domain):
    if not domain: return True
    domain = domain.lower()
    if domain.count(".") > 5: return True
    if ".." in domain: return True
    if re.search(r"[^a-z0-9\.-]", domain): return True
    if re.fullmatch(r"[0-9\.-]+", domain): return True
    if domain.split(".")[-1] in BAD_TLDS: return True
    return False

# ============================================================
# URL HELPERS
# ============================================================
URL_REGEX = re.compile(r"((?:https?://|www\.)[^\s]+)")

def extract_url(text):
    m = URL_REGEX.search(text)
    if not m: return None
    url = m.group(1)
    if not url.startswith(("http://", "https://")):
        url = "http://" + url
    return url

def extract_domain(url):
    try:
        host = urlparse(url).netloc.lower()
        return host[4:] if host.startswith("www.") else host
    except:
        return None

def is_ip(url):
    try:
        host = urlparse(url).netloc
        return bool(re.match(r"^(?:\d{1,3}\.){3}\d{1,3}$", host))
    except:
        return False

# ============================================================
# MAIN PREDICTION
# ============================================================
def predict_sms_or_url(message):
    if not message:
        return {"error": "Empty message"}

    text_vec = tfidf.transform([message])
    text_prob = float(text_model.predict_proba(text_vec)[0][1])

    url = extract_url(message)
    domain = extract_domain(url) if url else None

    url_prob = None

    if url:
        feat = {
            "URLLength": len(url),
            "num_digits_url": sum(c.isdigit() for c in url),
            "NoOfLettersInURL": sum(c.isalpha() for c in url),
            "is_https": 1 if url.startswith("https://") else 0,
            "have_at": 1 if "@" in url else 0,
            "NoOfQMarkInURL": url.count("?"),
            "NoOfAmpersandInURL": url.count("&"),
            "NoOfEqualsInURL": url.count("="),
            "url_num_dots": url.count("."),
            "NoOfOtherSpecialCharsInURL": sum(c in "/#:;,_-" for c in url),
            "have_ip": 1 if is_ip(url) else 0,
        }

        df_pred = pd.DataFrame([feat])

        # SAFE FIX — guarantee all features exist
        for col in url_model.feature_names_in_:
            if col not in df_pred.columns:
                df_pred[col] = 0

        df_pred = df_pred[url_model.feature_names_in_]

        url_prob = float(url_model.predict_proba(df_pred)[0][1])

    # SAFE OVERRIDES
    if domain:
        if (is_auto_safe_domain(domain)
            or is_safe_domain(domain)
            or (is_valid_safe_domain(domain) and not is_suspicious_domain(domain))
        ):
            return {
                "Prediction": "HAM",
                "Text_Spam_Probability": round(text_prob, 4),
                "URL_Spam_Probability": round(url_prob, 4) if url_prob else None,
                "Final_Spam_Probability": 0.0,
                "Extracted_URL": url,
                "Domain": domain,
                "Is_Safe_Domain": True,
            }

    final_prob = text_prob if url_prob is None else 1 - ((1 - text_prob) * (1 - url_prob))
    label = "SPAM" if final_prob >= 0.5 else "HAM"

    return {
        "Prediction": label,
        "Text_Spam_Probability": round(text_prob, 4),
        "URL_Spam_Probability": round(url_prob, 4) if url_prob else None,
        "Final_Spam_Probability": round(final_prob, 4),
        "Extracted_URL": url,
        "Domain": domain,
        "Is_Safe_Domain": False,
    }

# ============================================================
# ROUTES
# ============================================================
@app.head("/")
async def head_root():
    return {}

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "result": None,
        "input_text": ""
    })

@app.post("/predict")
async def predict_form(request: Request, message: str = Form(None)):
    if not message:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "result": {"error": "Message cannot be empty"},
            "input_text": message
        })

    result = predict_sms_or_url(message)
    return templates.TemplateResponse("index.html", {
        "request": request,
        "result": result,
        "input_text": message
    })

class PredictRequest(BaseModel):
    message: str

@app.post("/api/predict")
async def api_predict(payload: PredictRequest):
    return predict_sms_or_url(payload.message)

# ============================================================
# LOCAL DEV
# ============================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
