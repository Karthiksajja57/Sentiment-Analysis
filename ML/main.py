"""EnsembleIQ Sentiment Analysis Backend API Service"""
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib
import re
import nltk
from nltk.corpus import stopwords
import os

app = FastAPI(title="EnsembleIQ Sentiment API")

# Setup CORS to allow our frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # For development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models if they exist
vectorizer = None
rf_model = None
ada_model = None
gb_model = None
xgb_model = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    vectorizer = joblib.load(os.path.join(BASE_DIR, 'vectorizer.pkl'))
    rf_model = joblib.load(os.path.join(BASE_DIR, 'rf_model.pkl'))
    ada_model = joblib.load(os.path.join(BASE_DIR, 'ada_model.pkl'))
    gb_model = joblib.load(os.path.join(BASE_DIR, 'gb_model.pkl'))
    xgb_model = joblib.load(os.path.join(BASE_DIR, 'xgb_model.pkl'))
    print("4-Model ensemble loaded successfully (Random Forest, AdaBoost, Gradient Boost, XGBoost)!")
except Exception as e:
    print(f"Warning: Models not loaded. Please train first. Error: {e}")

class PredictRequest(BaseModel):
    text: str

def preprocess_text(text):
    text = str(text)
    # Remove URLs, mentions, hashtags
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#', '', text)
    text = text.lower()
    
    # Expand contractions so negation and sentiment words are isolated
    contractions = {
        r"won\'t": "will not",
        r"can\'t": "can not",
        r"n\'t": " not",
        r"\'re": " are",
        r"\'s": " is",
        r"\'d": " would",
        r"\'ll": " will",
        r"\'t": " not",
        r"\'ve": " have",
        r"\'m": " am"
    }
    for pattern, replacement in contractions.items():
        text = re.sub(pattern, replacement, text)
        
    # Bind negation markers to the following word: e.g. "not good" -> "not_good", "not expecting" -> "not_expecting"
    text = re.sub(r"\b(not|no|never|hardly|barely|without)\s+(\w+)", r"\1_\2", text)
    
    # Strip non-alphanumeric except underscores in compound tokens
    text = re.sub(r"[^\w\s]", ' ', text)
    
    try:
        stop_words = set(stopwords.words('english'))
    except Exception:
        stop_words = set()
    stop_words.discard('not')
    stop_words.discard('no')
    
    words = text.split()
    words = [w for w in words if w not in stop_words and len(w) > 1]
    
    return " ".join(words)

# Zero-vector baseline priors for calibration
zero_feat = vectorizer.transform([""]).toarray() if vectorizer else None
priors = {}
if all([vectorizer, rf_model, ada_model, gb_model, xgb_model]) and zero_feat is not None:
    priors = {
        "Random Forest": float(rf_model.predict_proba(zero_feat)[0][1]),
        "AdaBoost": float(ada_model.predict_proba(zero_feat)[0][1]),
        "Gradient Boost": float(gb_model.predict_proba(zero_feat)[0][1]),
        "XGBoost": float(xgb_model.predict_proba(zero_feat)[0][1]),
    }

BASE_POS = {
    "good", "great", "excellent", "amazing", "love", "loved", "best", "perfect",
    "wonderfully", "wonderful", "flawless", "pleased", "happy", "recommend", "brilliant",
    "superb", "nice", "cool", "fantastic", "awesome", "fine", "favorite", "favourite",
    "enjoy", "enjoyed", "liked", "like", "worth", "satisfied", "well", "positive"
}

BASE_NEG = {
    "bad", "worst", "terrible", "horrible", "hate", "hated", "disappointed", "disappointing",
    "cheap", "broke", "waste", "poor", "not_recommend", "awful", "trash", "broken", "useless",
    "boring", "suck", "sucks", "fail", "failed", "freeze", "freezes", "lag", "lags", "slow", "negative"
}

def compute_sentiment_shift(clean_text):
    tokens = clean_text.split()
    pos_score = 0.0
    neg_score = 0.0
    
    for tok in tokens:
        if tok.startswith('not_'):
            base = tok[4:]
            if base in BASE_NEG:
                # Negated negative (litotes: "not worst", "not bad", "not terrible") -> Positive signal
                pos_score += 1.5
            elif base in BASE_POS:
                # Negated positive ("not good", "not great", "not happy") -> Negative signal
                neg_score += 1.5
        else:
            if tok in BASE_POS:
                pos_score += 1.0
            elif tok in BASE_NEG:
                neg_score += 1.0
                
    return (pos_score - neg_score) * 0.22

@app.get("/")
@app.get("/health")
def health_check():
    return {
        "status": "online",
        "models_loaded": bool(all([vectorizer, rf_model, ada_model, gb_model, xgb_model]))
    }

@app.post("/predict")
def predict_sentiment(request: PredictRequest):
    if not all([vectorizer, rf_model, ada_model, gb_model, xgb_model]):
        return {"error": "Models are not loaded."}
        
    clean_text = preprocess_text(request.text)
    features = vectorizer.transform([clean_text]).toarray()
    
    lex_shift = compute_sentiment_shift(clean_text)
    
    models = {
        "Random Forest": rf_model,
        "AdaBoost": ada_model,
        "Gradient Boost": gb_model,
        "XGBoost": xgb_model
    }
    
    predictions = {}
    for name, model in models.items():
        p_pos = float(model.predict_proba(features)[0][1]) + lex_shift
        p_prior = priors.get(name, 0.5)
        predictions[name] = "Positive" if p_pos > p_prior else "Negative"
        
    return {
        "text": request.text,
        "predictions": predictions
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)


