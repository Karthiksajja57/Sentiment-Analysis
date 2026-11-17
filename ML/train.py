"""EnsembleIQ Model Training Pipeline"""
import pandas as pd
import numpy as np
import os
import re
import joblib
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

nltk.download('stopwords')

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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

print("Loading dataset...")
# Load dataset
data_path = os.path.join(BASE_DIR, 'data.csv')
df = pd.read_csv(data_path, encoding='latin-1', header=0, names=['target', 'ids', 'date', 'flag', 'user', 'text'])

# Sentiment140 uses 0 for negative, 4 for positive. Map 4 to 1.
df['target'] = df['target'].replace(4, 1)

print("Sampling balanced dataset...")
# Sample 35,000 of each class (70,000 total) for fast, robust training
df_pos = df[df['target'] == 1].sample(35000, random_state=42)
df_neg = df[df['target'] == 0].sample(35000, random_state=42)
df_sampled = pd.concat([df_pos, df_neg])

print("Preprocessing text with negation binding...")
df_sampled['clean_text'] = df_sampled['text'].apply(preprocess_text)

print("Vectorizing with unigrams, bigrams, and trigrams (1, 3)...")
# Using unigrams, bigrams, and trigrams to capture phrases like 'complete waste of', 'proved me wrong'
vectorizer = TfidfVectorizer(max_features=25000, ngram_range=(1, 3), sublinear_tf=True, min_df=2)
# Keep sparse CSR matrix directly: prevents massive memory consumption and speeds up training
X = vectorizer.fit_transform(df_sampled['clean_text'])
y = df_sampled['target'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


print("Training Random Forest (100 estimators, max_depth=25)...")
rf = RandomForestClassifier(n_estimators=100, max_depth=25, min_samples_split=10, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
print(f"Random Forest Accuracy: {accuracy_score(y_test, rf.predict(X_test)):.4f}")

print("Training AdaBoost (depth-3 decision trees)...")
try:
    ada = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=3), n_estimators=60, random_state=42)
except TypeError:
    ada = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=3), n_estimators=60, random_state=42)
ada.fit(X_train, y_train)
print(f"AdaBoost Accuracy: {accuracy_score(y_test, ada.predict(X_test)):.4f}")

print("Training Gradient Boosting (max_depth=5)...")
gb = GradientBoostingClassifier(n_estimators=70, max_depth=5, learning_rate=0.15, random_state=42)
gb.fit(X_train, y_train)
print(f"Gradient Boosting Accuracy: {accuracy_score(y_test, gb.predict(X_test)):.4f}")

print("Training XGBoost (max_depth=6, learning_rate=0.15)...")
xgb = XGBClassifier(n_estimators=70, max_depth=6, learning_rate=0.15, eval_metric='logloss', random_state=42, n_jobs=-1)
xgb.fit(X_train, y_train)
print(f"XGBoost Accuracy: {accuracy_score(y_test, xgb.predict(X_test)):.4f}")

print("Saving upgraded models...")
joblib.dump(vectorizer, os.path.join(BASE_DIR, 'vectorizer.pkl'))
joblib.dump(rf, os.path.join(BASE_DIR, 'rf_model.pkl'))
joblib.dump(ada, os.path.join(BASE_DIR, 'ada_model.pkl'))
joblib.dump(gb, os.path.join(BASE_DIR, 'gb_model.pkl'))
joblib.dump(xgb, os.path.join(BASE_DIR, 'xgb_model.pkl'))

print("All models successfully upgraded and saved!")

