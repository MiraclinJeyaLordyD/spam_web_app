from pathlib import Path
import re
from urllib.parse import urlparse
from typing import Optional, Dict, Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# -----------------------------
# Paths & model loading
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

tfidf = joblib.load(MODELS_DIR / "tfidf.pkl")
text_model = joblib.load(MODELS_DIR / "text_model.pkl")
url_model = joblib.load(MODELS_DIR / "url_model.pkl")

# -----------------------------
# FastAPI app & templates
# -----------------------------
app = FastAPI(title="Spam Detection Web App")

app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


class PredictRequest(BaseModel):
    message: str


# -----------------------------
# Helper functions (similar to train_and_predict.py logic)
# -----------------------------

# Regex to extract URLs (http, https, or www)
URL_REGEX = re.compile(r"((?:https?://|www\.)[^\s]+)", re.IGNORECASE)

SAFE_DOMAINS = {
    # Add your own safe/whitelisted domains here
    "google.com",
    "gmail.com",
    "youtube.com",
    "facebook.com",
    "instagram.com",
    "whatsapp.com",
    "amazon.in",
    "flipkart.com",
    "paytm.com",
    "phonepe.com",
    "icicibank.com",
    "sbi.co.in",
    "hdfcbank.com",
}

BAD_TLDS = {
    # Example suspicious TLDs
    "xyz",
    "top",
    "click",
    "link",
    "work",
    "info",
    "cn",
    "ru",
    "zip",
    "review",
    "country",
}


def extract_url(text: str) -> Optional[str]:
    """Extract the first URL from the text, if any."""
    match = URL_REGEX.search(text)
    if not match:
        return None
    url = match.group(1)
    # Normalize URL to make sure it has scheme
    if not url.lower().startswith(("http://", "https://")):
        url = "http://" + url
    return url.strip()


def extract_domain(url: str) -> Optional[str]:
    """Extract domain/host from URL."""
    try:
        parsed = urlparse(url)
        host = parsed.netloc or parsed.path
        host = host.split("/")[0]
        if host.startswith("www."):
            host = host[4:]
        return host.lower()
    except Exception:
        return None


def is_ip_address(host: str) -> bool:
    """Check if host is an IP address."""
    ip_pattern = re.compile(
        r"^(?:\d{1,3}\.){3}\d{1,3}$"
    )  # simple IPv4 check, enough for this use
    return bool(ip_pattern.match(host))


def is_safe_domain(domain: Optional[str]) -> bool:
    if not domain:
        return False
    # Exact match on known safe domains
    return domain in SAFE_DOMAINS


def has_bad_tld(domain: Optional[str]) -> bool:
    if not domain:
        return False
    parts = domain.split(".")
    if not parts:
        return False
    tld = parts[-1]
    return tld in BAD_TLDS


def build_url_features(url: str, domain: Optional[str]) -> pd.DataFrame:
    """
    Build URL feature dataframe to feed into url_model.
    This tries to match typical URL feature names used earlier.
    You can adjust if your train_and_predict.py uses slightly different names.
    """
    url_str = url or ""
    length = len(url_str)
    num_digits = sum(ch.isdigit() for ch in url_str)
    num_letters = sum(ch.isalpha() for ch in url_str)
    is_https = 1 if url_str.lower().startswith("https://") else 0
    have_at = 1 if "@" in url_str else 0
    num_qmarks = url_str.count("?")
    num_ampersand = url_str.count("&")
    num_equals = url_str.count("=")
    num_dots = url_str.count(".")
    # "Other special chars" (rough approximation)
    special_chars = r"/#:;,_-"
    num_other_special = sum(ch in special_chars for ch in url_str)
    host = extract_domain(url_str) or ""
    have_ip = 1 if is_ip_address(host) else 0

    # Base feature dict (names based on your previous dataset columns)
    feature_dict: Dict[str, Any] = {
        "URLLength": length,
        "num_digits_url": num_digits,
        "NoOfLettersInURL": num_letters,
        "is_https": is_https,
        "have_at": have_at,
        "have_ip": have_ip,
        "NoOfQMarkInURL": num_qmarks,
        "NoOfAmpersandInURL": num_ampersand,
        "NoOfEqualsInURL": num_equals,
        "NoOfOtherSpecialCharsInURL": num_other_special,
        "url_num_dots": num_dots,
        "url": url_str,
    }

    # Align with model's feature order if available
    feature_names = getattr(url_model, "feature_names_in_", None)
    if feature_names is not None:
        # Create a dictionary with exactly the model's feature names
        aligned = {name: 0 for name in feature_names}
        for k, v in feature_dict.items():
            if k in aligned:
                aligned[k] = v
        df = pd.DataFrame([aligned])
    else:
        # Fallback: just use feature_dict (model must have been trained the same way)
        df = pd.DataFrame([feature_dict])

    return df


def get_text_spam_prob(message: str) -> float:
    """Get spam probability from the text model."""
    X = tfidf.transform([message])
    # Assuming models support predict_proba (MultinomialNB, LogisticRegression etc.)
    prob = text_model.predict_proba(X)[0, 1]
    return float(prob)


def get_url_spam_prob(url: str, domain: Optional[str]) -> float:
    """Get spam probability from the URL model."""
    df_features = build_url_features(url, domain)
    prob = url_model.predict_proba(df_features)[0, 1]
    return float(prob)


def combine_probabilities(text_prob: float, url_prob: Optional[float]) -> float:
    """
    Combine text and URL spam probabilities into a final probability.
    This uses a noisy-OR style combination:
        final = 1 - (1 - text_prob) * (1 - url_prob)
    If there is no URL, we just return text_prob.

    If your train_and_predict.py uses a different formula,
    you can replace this function with that exact logic.
    """
    if url_prob is None:
        return text_prob
    return 1.0 - (1.0 - text_prob) * (1.0 - url_prob)


# -----------------------------
# Routes
# -----------------------------
@app.get("/")
async def index(request: Request):
    """Render the main HTML page."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
async def predict(request_data: PredictRequest):
    message = request_data.message.strip()

    if not message:
        return JSONResponse(
            status_code=400, content={"error": "Message cannot be empty."}
        )

    # 1. Text spam probability
    try:
        text_spam_prob = get_text_spam_prob(message)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Error in text model prediction: {str(e)}"},
        )

    # 2. URL extraction & probability
    extracted_url = extract_url(message)
    domain = extract_domain(extracted_url) if extracted_url else None

    url_spam_prob: Optional[float] = None

    if extracted_url and domain:
        try:
            url_spam_prob = get_url_spam_prob(extracted_url, domain)
        except Exception:
            # If URL model fails for some reason, keep it None
            url_spam_prob = None

        # Optional: adjust URL probability using safe-domain / bad TLD heuristics
        if is_safe_domain(domain):
            # Strongly reduce probability if known safe domain
            url_spam_prob = url_spam_prob * 0.3 if url_spam_prob is not None else 0.0
        elif has_bad_tld(domain):
            # Slightly boost probability for bad TLD
            url_spam_prob = min(1.0, (url_spam_prob or 0.0) + 0.2)

    # 3. Final combined probability
    final_spam_prob = combine_probabilities(text_spam_prob, url_spam_prob)

    prediction_label = "SPAM" if final_spam_prob >= 0.5 else "HAM"

    response_data = {
        "prediction": prediction_label,
        "text_spam_probability": round(text_spam_prob, 4),
        "url_spam_probability": round(url_spam_prob, 4) if url_spam_prob is not None else None,
        "final_spam_probability": round(final_spam_prob, 4),
        "extracted_url": extracted_url,
        "domain": domain,
    }

    return JSONResponse(content=response_data)
