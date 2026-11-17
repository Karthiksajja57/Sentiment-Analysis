"""EnsembleIQ — Vercel Serverless Function: /api/health"""
from http.server import BaseHTTPRequestHandler
import json
import os
import joblib

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'ML')

def _models_loaded():
    try:
        for f in ['vectorizer.pkl','rf_model.pkl','ada_model.pkl','gb_model.pkl','xgb_model.pkl']:
            if not os.path.exists(os.path.join(BASE_DIR, f)):
                return False
        return True
    except Exception:
        return False

class handler(BaseHTTPRequestHandler):
    def _cors(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')

    def do_OPTIONS(self):
        self.send_response(200)
        self._cors()
        self.end_headers()

    def do_GET(self):
        body = json.dumps({
            "status": "online",
            "models_loaded": _models_loaded()
        }).encode()
        self.send_response(200)
        self._cors()
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(body)
