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
# PATHS & MODEL LOADING
# ============================================================
BASE_DIR = Path(__file__).resolve().parent

tfidf = joblib.load(BASE_DIR / "tfidf.pkl")
text_model = joblib.load(BASE_DIR / "text_model.pkl")
url_model = joblib.load(BASE_DIR / "url_model.pkl")

# ============================================================
# FASTAPI APP & TEMPLATES
# ============================================================
app = FastAPI()

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

static_dir = BASE_DIR / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# ============================================================
# SAFE PUBLIC DOMAINS
# ============================================================
SAFE_BASE_DOMAINS = {
    "google.com", "gmail.com", "youtube.com", "google.co.in",
    "microsoft.com", "office.com", "live.com", "outlook.com",
    "apple.com", "icloud.com", "android.com",

    "amazon.in", "amazon.com", "flipkart.com",
    "swiggy.com", "zomato.com", "bigbasket.com",

    "facebook.com", "instagram.com", "whatsapp.com", "twitter.com",
    "x.com", "linkedin.com", "reddit.com", "pinterest.com",

    "paytm.com", "phonepe.com", "bharatpe.com", "googlepay.com",
    "razorpay.com", "billdesk.com", "ccavenue.com",

    "icicibank.com", "hdfcbank.com", "sbi.co.in", "axisbank.com",
    "kotak.com", "bankofbaroda.in", "canarabank.com",

    "gov.in", "nic.in", "india.gov.in", "incometax.gov.in",
    "uidai.gov.in", "passportindia.gov.in",

    "airtel.in", "jio.com", "vi.in",

    "chatgpt.com", "openai.com", "api.openai.com",
    "vercel.app", "onrender.com", "github.com", "github.io",
}

# ============================================================
# SAFE PUBLIC DOMAIN CHECK
# ============================================================
def is_safe_domain(domain: str) -> bool:
    if not domain:
        return False
    domain = domain.lower()
    for base in SAFE_BASE_DOMAINS:
        if domain == base or domain.endswith("." + base):
            return True
    return False


# ============================================================
# AUTO-SAFE: LOCALHOST / INTERNAL / JUPYTER
# ============================================================
def is_auto_safe_domain(domain: str) -> bool:
    if not domain:
        return False

    domain = domain.lower()

    # localhost / loopback
    if domain.startswith("localhost") or domain.startswith("127.0.0.1"):
        return True

    # Private IP ranges
    private_ip_ranges = [
        r"^192\.168\.\d+\.\d+$",
        r"^10\.\d+\.\d+\.\d+$",
        r"^172\.(1[6-9]|2\d|3[0-1])\.\d+\.\d+$",
    ]
    for pattern in private_ip_ranges:
        if re.match(pattern, domain):
            return True

    # Notebook environments
    if "jupyter" in domain or "notebook" in domain:
        return True

    # .local hostnames
    if domain.endswith(".local"):
        return True

    return False


# ============================================================
# UNIVERSAL VALID DOMAIN CHECKER
# ============================================================
VALID_TLDS = {
    "com", "org", "net", "gov", "in", "co", "edu", "info", "app", "io", "ai",
    "store", "shop", "tech", "dev", "me", "us", "uk", "ca", "au", "de", "fr",
    "club", "link", "live", "online", "site", "blog",
}


def is_valid_safe_domain(domain: str) -> bool:
    if not domain:
        return False

    domain = domain.lower()

    # must contain dot
    if "." not in domain:
        return False

    labels = domain.split(".")
    tld = labels[-1]

    # check TLD
    if tld not in VALID_TLDS:
        return False

    # validate labels
    for label in labels:
        if not re.match(r"^[a-z0-9-]{1,63}$", label):
            return False
        if label.startswith("-") or label.endswith("-"):
            return False

    # domain length check
    if len(domain) > 253:
        return False

    return True


# ============================================================
# SUSPICIOUS DOMAIN CHECKER
# ============================================================
BAD_TLDS = {"xyz", "top", "zip", "ml", "ga", "tk", "ru", "cn"}


def is_suspicious_domain(domain: str) -> bool:
    if not domain:
        return True

    domain = domain.lower()

    # too many dots → suspicious
    if domain.count(".") > 5:
        return True

    # invalid characters
    if re.search(r"[^a-z0-9\.-]", domain):
        return True

    # double dots
    if ".." in domain:
        return True

    # all-numeric domains
    if re.fullmatch(r"[0-9\-\.]+", domain):
        return True

    # bad TLDs
    if domain.split(".")[-1] in BAD_TLDS:
        return True

    return False


# ============================================================
# URL HELPERS
# ============================================================
URL_REGEX = re.compile(r"((?:https?://|www\.)[^\s]+)")


def extract_url(text: str):
    m = URL_REGEX.search(text)
    if not m:
        return None
    url = m.group(1)
    if not url.startswith(("http://", "https://")):
        url = "http://" + url
    return url


def extract_domain(url: str):
    try:
        host = urlparse(url).netloc.lower()
        if host.startswith("www."):
            host = host[4:]
        return host
    except Exception:
        return None


def is_ip(url: str) -> bool:
    try:
        host = urlparse(url).netloc
        return bool(re.match(r"^(?:\d{1,3}\.){3}\d{1,3}$", host))
    except Exception:
        return False


# ============================================================
# CORE PREDICTION FUNCTION
# ============================================================
def predict_sms_or_url(message: str):
    message = (message or "").strip()

    # ---------------- TEXT MODEL ----------------
    text_vec = tfidf.transform([message])
    text_prob = float(text_model.predict_proba(text_vec)[0][1])

    # ---------------- URL MODEL ----------------
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

        df_pred = (
            pd.DataFrame([feat])
            .reindex(columns=url_model.feature_names_in_, fill_value=0)
        )
        url_prob = float(url_model.predict_proba(df_pred)[0][1])

    # ========================================================
    # MASTER SAFE OVERRIDE: ANY SAFE DOMAIN → HAM
    # ========================================================
    if domain:
        if (
            is_auto_safe_domain(domain)
            or is_safe_domain(domain)
            or (is_valid_safe_domain(domain) and not is_suspicious_domain(domain))
        ):
            return {
                "Prediction": "HAM",
                "Text_Spam_Probability": round(text_prob, 4),
                "URL_Spam_Probability": round(url_prob, 4) if url_prob is not None else None,
                "Final_Spam_Probability": 0.0,
                "Extracted_URL": url,
                "Domain": domain,
                "Is_Safe_Domain": True,
            }

    # ========================================================
    # NORMAL COMBINATION
    # ========================================================
    if url_prob is None:
        final_prob = text_prob
    else:
        final_prob = 1 - ((1 - text_prob) * (1 - url_prob))

    label = "SPAM" if final_prob >= 0.5 else "HAM"

    return {
        "Prediction": label,
        "Text_Spam_Probability": round(text_prob, 4),
        "URL_Spam_Probability": round(url_prob, 4) if url_prob is not None else None,
        "Final_Spam_Probability": round(final_prob, 4),
        "Extracted_URL": url,
        "Domain": domain,
        "Is_Safe_Domain": False,
    }


# ============================================================
# Pydantic Model for JSON API
# ============================================================
class PredictRequest(BaseModel):
    message: str


# ============================================================
# ROUTES
# ============================================================
@app.get("/")
async def index(request: Request):
    """
    Simple HTML page with one input box (SMS/Text/URL).
    """
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "result": None,
            "input_text": "",
        },
    )


@app.post("/predict")
async def predict_form(request: Request, message: str = Form(...)):
    """
    Handles form submission from the HTML UI.
    """
    result = predict_sms_or_url(message)
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "result": result,
            "input_text": message,
        },
    )


@app.post("/api/predict")
async def predict_api(payload: PredictRequest):
    """
    JSON API endpoint for programmatic access.
    """
    result = predict_sms_or_url(payload.message)
    return result


# ============================================================
# DEV SERVER ENTRYPOINT (optional for local run)
# ============================================================
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
