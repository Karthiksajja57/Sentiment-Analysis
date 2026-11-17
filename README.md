# EnsembleIQ — Multi-Model Sentiment Analysis System

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688?logo=fastapi)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5%2B-orange?logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)

> **EnsembleIQ** is a full-stack sentiment analysis application powered by an ensemble of four machine learning models. It classifies text as **Positive** or **Negative** using Random Forest, AdaBoost, Gradient Boosting, and XGBoost — with a custom negation-aware preprocessing pipeline and a sleek real-time Studio frontend.

---

## ✨ Features

- **4-Model Ensemble** — Random Forest, AdaBoost, Gradient Boosting, XGBoost vote independently for robust predictions
- **Negation-Aware Preprocessing** — Binds negation words to their targets (e.g. `not good` → `not_good`) for accurate sentiment capture
- **Contraction Expansion** — Automatically expands `can't`, `won't`, `isn't`, etc. before classification
- **Lexicon-Calibrated Scoring** — Custom positive/negative lexicon shifts raw probabilities for edge cases
- **Real-time REST API** — FastAPI backend with `/predict` endpoint and interactive Swagger docs
- **Studio Frontend UI** — Clean, responsive web interface served locally on port 3000
- **One-command Launch** — `run.py` boots both backend and frontend simultaneously

---

## 🗂️ Project Structure

```
ml exter project/
├── run.py                   # One-command launcher (starts both servers)
├── requirements.txt         # Python dependencies
├── .gitignore
│
└── ML/
    ├── train.py             # Model training pipeline
    ├── main.py              # FastAPI backend API
    ├── data.csv             # Training dataset (Sentiment140, ~1.6M tweets) — not tracked in git
    ├── vectorizer.pkl       # Fitted TF-IDF vectorizer (25,000 features)
    ├── rf_model.pkl         # Trained Random Forest model
    ├── ada_model.pkl        # Trained AdaBoost model
    ├── gb_model.pkl         # Trained Gradient Boosting model
    ├── xgb_model.pkl        # Trained XGBoost model
    └── frontend/
        ├── index.html       # Studio UI
        ├── style.css        # Styling
        ├── app.js           # Frontend logic
        └── serve.py         # Simple HTTP server for the frontend
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- pip

### 1. Clone the Repository

```bash
git clone https://github.com/Karthiksajja57/Sentiment-Analysis.git
cd Sentiment-Analysis
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download NLTK Stopwords

```python
import nltk
nltk.download('stopwords')
```

### 4. (Optional) Retrain Models

Pre-trained models (`.pkl` files) are included in the repository. To retrain from scratch, download the [Sentiment140 dataset](https://www.kaggle.com/datasets/kazanova/sentiment140) (`data.csv`, ~1.6M tweets) and place it in the `ML/` folder, then run:

```bash
python ML/train.py
```

> Training uses a balanced sample of 70,000 tweets (35,000 positive + 35,000 negative) with TF-IDF (unigrams, bigrams, trigrams).

### 5. Launch the Application

```bash
python run.py
```

This starts:
- **Backend API** at `http://localhost:8000`
- **Frontend Studio** at `http://localhost:3000`

The browser opens automatically. Press **Ctrl+C** to stop both servers.

---

## 🔌 API Usage

### Health Check

```
GET http://localhost:8000/health
```

**Response:**
```json
{
  "status": "online",
  "models_loaded": true
}
```

### Predict Sentiment

```
POST http://localhost:8000/predict
Content-Type: application/json

{
  "text": "I absolutely loved this product, it's amazing!"
}
```

**Response:**
```json
{
  "text": "I absolutely loved this product, it's amazing!",
  "predictions": {
    "Random Forest": "Positive",
    "AdaBoost": "Positive",
    "Gradient Boost": "Positive",
    "XGBoost": "Positive"
  }
}
```

### Interactive API Docs

Visit `http://localhost:8000/docs` for the auto-generated Swagger UI.

---

## 🧠 ML Pipeline

```
Raw Text Input
     │
     ▼
Preprocessing (preprocess_text)
  • Remove URLs, mentions, hashtags
  • Expand contractions  (can't → can not)
  • Negation binding     (not good → not_good)
  • Remove stopwords     (preserve "not", "no")
     │
     ▼
TF-IDF Vectorization
  • 25,000 features
  • Unigrams + Bigrams + Trigrams
  • Sublinear TF scaling
     │
     ▼
4 Independent Classifiers
  ┌──────────────────┐  ┌──────────────────┐
  │  Random Forest   │  │    AdaBoost      │
  │  100 estimators  │  │  60 estimators   │
  │  max_depth=25    │  │  depth-3 trees   │
  └──────────────────┘  └──────────────────┘
  ┌──────────────────┐  ┌──────────────────┐
  │ Gradient Boost   │  │    XGBoost       │
  │  70 estimators   │  │  70 estimators   │
  │  max_depth=5     │  │  max_depth=6     │
  └──────────────────┘  └──────────────────┘
     │
     ▼
Lexicon Shift Calibration
  • Custom positive/negative word lists
  • Negation-aware scoring adjustment
     │
     ▼
Final Per-Model Prediction
  (Positive / Negative)
```

---

## 📦 Dependencies

| Package | Version |
|---|---|
| fastapi | ≥ 0.115.0 |
| uvicorn | ≥ 0.30.0 |
| scikit-learn | ≥ 1.5.0 |
| xgboost | ≥ 2.0.0 |
| joblib | ≥ 1.4.0 |
| nltk | ≥ 3.9.0 |
| pandas | ≥ 2.2.0 |
| numpy | ≥ 2.0.0 |

---

## 📊 Dataset

Training is based on the **[Sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140)** dataset:
- 1.6 million tweets labelled as Positive (4) or Negative (0)
- Balanced sample of **70,000 tweets** (35k each class) used for training
- 80/20 train-test split

> `data.csv` is excluded from this repository due to its ~196 MB size. Download it from Kaggle and place it in `ML/data.csv` before training.

---

## 👤 Author

**Sajja Dinesh Karthik**
- GitHub: [@Karthiksajja57](https://github.com/Karthiksajja57)

---

## 📄 License

This project is licensed under the MIT License.

