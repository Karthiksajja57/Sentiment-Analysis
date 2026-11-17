"""EnsembleIQ Sentiment Analysis Backend API Service for Vercel Serverless"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import re
import os

app = FastAPI(title="EnsembleIQ Sentiment API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ML_DIR = os.path.join(BASE_DIR, '..', 'ML')

def find_file(filename):
    p1 = os.path.join(BASE_DIR, filename)
    if os.path.exists(p1):
        return p1
    p2 = os.path.join(ML_DIR, filename)
    if os.path.exists(p2):
        return p2
    return p1

vectorizer = None
rf_model = None
ada_model = None
gb_model = None
xgb_model = None
priors = {}
zero_feat = None

try:
    vectorizer = joblib.load(find_file('vectorizer.pkl'))
    rf_model = joblib.load(find_file('rf_model.pkl'))
    ada_model = joblib.load(find_file('ada_model.pkl'))
    gb_model = joblib.load(find_file('gb_model.pkl'))
    xgb_model = joblib.load(find_file('xgb_model.pkl'))
    zero_feat = vectorizer.transform([""]).toarray()
    priors = {
        "Random Forest": float(rf_model.predict_proba(zero_feat)[0][1]),
        "AdaBoost": float(ada_model.predict_proba(zero_feat)[0][1]),
        "Gradient Boost": float(gb_model.predict_proba(zero_feat)[0][1]),
        "XGBoost": float(xgb_model.predict_proba(zero_feat)[0][1]),
    }
    print("Models loaded successfully!")
except Exception as e:
    print(f"Warning: Models not loaded. Error: {e}")

class PredictRequest(BaseModel):
    text: str

CONTRACTIONS = {
    r"won\'t": "will not", r"can\'t": "can not", r"n\'t": " not",
    r"\'re": " are", r"\'s": " is", r"\'d": " would",
    r"\'ll": " will", r"\'t": " not", r"\'ve": " have", r"\'m": " am",
}

STOP_WORDS = {
    "i","me","my","myself","we","our","ours","ourselves","you","your","yours",
    "yourself","yourselves","he","him","his","himself","she","her","hers",
    "herself","it","its","itself","they","them","their","theirs","themselves",
    "what","which","who","whom","this","that","these","those","am","is","are",
    "was","were","be","been","being","have","has","had","having","do","does",
    "did","doing","a","an","the","and","but","if","or","because","as","until",
    "while","of","at","by","for","with","about","against","between","into",
    "through","during","before","after","above","below","to","from","up","down",
    "in","out","on","off","over","under","again","further","then","once","here",
    "there","when","where","why","how","all","both","each","few","more","most",
    "other","some","such","only","own","same","so","than","too","very","s","t",
    "can","will","just","don","should","now","d","ll","m","o","re","ve","y",
    "ain","aren","couldn","didn","doesn","hadn","hasn","haven","isn","ma",
    "mightn","mustn","needn","shan","shouldn","wasn","weren","won","wouldn"
}
STOP_WORDS.discard('not')
STOP_WORDS.discard('no')

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

def preprocess_text(text: str) -> str:
    text = str(text)
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#', '', text)
    text = text.lower()
    for pattern, replacement in CONTRACTIONS.items():
        text = re.sub(pattern, replacement, text)
    text = re.sub(r"\b(not|no|never|hardly|barely|without)\s+(\w+)", r"\1_\2", text)
    text = re.sub(r"[^\w\s]", ' ', text)
    words = text.split()
    words = [w for w in words if w not in STOP_WORDS and len(w) > 1]
    return " ".join(words)

def compute_sentiment_shift(clean_text: str) -> float:
    tokens = clean_text.split()
    pos_score = 0.0
    neg_score = 0.0
    for tok in tokens:
        if tok.startswith('not_'):
            base = tok[4:]
            if base in BASE_NEG:
                pos_score += 1.5
            elif base in BASE_POS:
                neg_score += 1.5
        else:
            if tok in BASE_POS:
                pos_score += 1.0
            elif tok in BASE_NEG:
                neg_score += 1.0
    return (pos_score - neg_score) * 0.22

@app.get("/")
@app.get("/health")
@app.get("/api")
@app.get("/api/health")
def health_check():
    return {
        "status": "online",
        "models_loaded": bool(all([vectorizer, rf_model, ada_model, gb_model, xgb_model]))
    }

@app.post("/predict")
@app.post("/api/predict")
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
