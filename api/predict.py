"""EnsembleIQ — Vercel Serverless Function: /api/predict"""
from http.server import BaseHTTPRequestHandler
import json
import re
import os
import joblib

# ---------------------------------------------------------------------------
# Paths to model files (relative to project root)
# ---------------------------------------------------------------------------
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'ML')

vectorizer = None
rf_model   = None
ada_model  = None
gb_model   = None
xgb_model  = None
priors     = {}
zero_feat  = None

def _load_models():
    global vectorizer, rf_model, ada_model, gb_model, xgb_model, priors, zero_feat
    try:
        vectorizer = joblib.load(os.path.join(BASE_DIR, 'vectorizer.pkl'))
        rf_model   = joblib.load(os.path.join(BASE_DIR, 'rf_model.pkl'))
        ada_model  = joblib.load(os.path.join(BASE_DIR, 'ada_model.pkl'))
        gb_model   = joblib.load(os.path.join(BASE_DIR, 'gb_model.pkl'))
        xgb_model  = joblib.load(os.path.join(BASE_DIR, 'xgb_model.pkl'))
        zero_feat  = vectorizer.transform([""]).toarray()
        priors = {
            "Random Forest":    float(rf_model.predict_proba(zero_feat)[0][1]),
            "AdaBoost":         float(ada_model.predict_proba(zero_feat)[0][1]),
            "Gradient Boost":   float(gb_model.predict_proba(zero_feat)[0][1]),
            "XGBoost":          float(xgb_model.predict_proba(zero_feat)[0][1]),
        }
    except Exception as e:
        print(f"Model load error: {e}")

_load_models()

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
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
    "good","great","excellent","amazing","love","loved","best","perfect",
    "wonderfully","wonderful","flawless","pleased","happy","recommend","brilliant",
    "superb","nice","cool","fantastic","awesome","fine","favorite","favourite",
    "enjoy","enjoyed","liked","like","worth","satisfied","well","positive"
}
BASE_NEG = {
    "bad","worst","terrible","horrible","hate","hated","disappointed","disappointing",
    "cheap","broke","waste","poor","not_recommend","awful","trash","broken","useless",
    "boring","suck","sucks","fail","failed","freeze","freezes","lag","lags","slow","negative"
}

def preprocess(text: str) -> str:
    text = str(text)
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#', '', text)
    text = text.lower()
    for pat, repl in CONTRACTIONS.items():
        text = re.sub(pat, repl, text)
    text = re.sub(r"\b(not|no|never|hardly|barely|without)\s+(\w+)", r"\1_\2", text)
    text = re.sub(r"[^\w\s]", ' ', text)
    words = [w for w in text.split() if w not in STOP_WORDS and len(w) > 1]
    return " ".join(words)

def sentiment_shift(clean: str) -> float:
    pos, neg = 0.0, 0.0
    for tok in clean.split():
        if tok.startswith('not_'):
            base = tok[4:]
            if base in BASE_NEG: pos += 1.5
            elif base in BASE_POS: neg += 1.5
        else:
            if tok in BASE_POS: pos += 1.0
            elif tok in BASE_NEG: neg += 1.0
    return (pos - neg) * 0.22

# ---------------------------------------------------------------------------
# Vercel Handler
# ---------------------------------------------------------------------------
class handler(BaseHTTPRequestHandler):
    def _cors(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')

    def do_OPTIONS(self):
        self.send_response(200)
        self._cors()
        self.end_headers()

    def do_GET(self):
        body = json.dumps({
            "status": "online",
            "models_loaded": bool(all([vectorizer, rf_model, ada_model, gb_model, xgb_model]))
        }).encode()
        self.send_response(200)
        self._cors()
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if not all([vectorizer, rf_model, ada_model, gb_model, xgb_model]):
            self._json(503, {"error": "Models not loaded."})
            return

        length  = int(self.headers.get('Content-Length', 0))
        payload = json.loads(self.rfile.read(length))
        text    = payload.get('text', '')

        clean   = preprocess(text)
        feats   = vectorizer.transform([clean]).toarray()
        shift   = sentiment_shift(clean)

        models = {
            "Random Forest":  rf_model,
            "AdaBoost":       ada_model,
            "Gradient Boost": gb_model,
            "XGBoost":        xgb_model,
        }
        predictions = {}
        for name, mdl in models.items():
            p_pos = float(mdl.predict_proba(feats)[0][1]) + shift
            predictions[name] = "Positive" if p_pos > priors.get(name, 0.5) else "Negative"

        self._json(200, {"text": text, "predictions": predictions})

    def _json(self, code, data):
        body = json.dumps(data).encode()
        self.send_response(code)
        self._cors()
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(body)
