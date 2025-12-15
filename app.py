# app.py (robust update)
from web3 import Web3
import os,ssl
from dotenv import load_dotenv
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from werkzeug.security import check_password_hash, generate_password_hash
import jwt
import datetime
import json
from pymongo import MongoClient, errors as pymongo_errors
import time
import re
import certifi
from PIL import Image
import io
import traceback
import joblib
import numpy as np
import warnings
import base64
import hashlib
import uuid
import json as _json
import tensorflow as tf
import csv
from collections import defaultdict
from functools import wraps

# Import Instagram Authenticity modules
try:
    from instagram_fetcher import InstagramMetadataFetcher
    from instagram_scorer import InstagramAuthenticityScorer
    from authenticity_schema import init_authenticity_schema
    AUTHENTICITY_MODULES_AVAILABLE = True
except ImportError as e:
    print(f"[WARNING] Authenticity modules not available: {e}")
    AUTHENTICITY_MODULES_AVAILABLE = False

# Import Influencer modules
try:
    from influencer_authenticator import InfluencerAuthenticator
    from influencer_fraud_detector import InfluencerFraudDetector
    from influencer_analytics import InfluencerAnalytics
    from influencer_campaigns_schema import init_influencer_campaigns_schema, init_payment_schema
    from influencer_instagram_verifier import InstagramInfluencerVerifier
    INFLUENCER_MODULES_AVAILABLE = True
except ImportError as e:
    print(f"[WARNING] Influencer modules not available: {e}")
    INFLUENCER_MODULES_AVAILABLE = False

# Optional imaging helpers (guarded)
try:
    import imagehash as _imagehash
except Exception:
    _imagehash = None
try:
    import pytesseract as _pytesseract
except Exception:
    _pytesseract = None
try:
    from pyzbar import pyzbar as _pyzbar
except Exception:
    _pyzbar = None

# ---------------------- CV model (Keras) ----------------------
cv_model_path = 'data/cv_model.h5'
CV_MODEL_PATH = cv_model_path
try:
    cv_model = tf.keras.models.load_model(cv_model_path)
except Exception:
    cv_model = None
CV_IMG_SIZE = int(os.getenv('CV_IMG_SIZE', '224'))
CV_AUTH_THRESHOLD = float(os.getenv('CV_AUTH_THRESHOLD', '0.5'))
# multimodal weights (can be tuned via .env)
MULTIMODAL_CV_WEIGHT = float(os.getenv('MULTIMODAL_CV_WEIGHT', '0.6'))
MULTIMODAL_NLP_WEIGHT = float(os.getenv('MULTIMODAL_NLP_WEIGHT', '0.4'))

try:
    # lazy import TensorFlow here only if available
    import tensorflow as _tf
    if os.path.exists(cv_model_path):
        try:
            cv_model = _tf.keras.models.load_model(cv_model_path)
            print(f"✅ CV model loaded from {cv_model_path}")
        except Exception as ee:
            print("⚠️ Could not load CV model at startup:", ee)
            cv_model = None
    else:
        print(f"ℹ️ CV model not found at {cv_model_path}; endpoint will try lazy-load on request")
except Exception as e:
    print("⚠️ TensorFlow not available or failed to import; CV model disabled until TF is installed:", e)
    cv_model = None

load_dotenv()
# Try to suppress noisy sklearn InconsistentVersionWarning when unpickling models
try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except Exception:
    # ignore if import not available
    pass

# Lazy import SentenceTransformer only in endpoint to avoid heavy startup cost
# from sentence_transformers import SentenceTransformer

# Load environment variables


# Initialize Flask app and CORS
app = Flask(__name__)
CORS(app)

# ---------------------- Rate Limiting (simple in-memory) ----------------------
_rate_limits = defaultdict(lambda: {'count': 0, 'reset_at': time.time() + 60})

def _check_rate_limit(key, max_requests=100, window_seconds=60):
    """Simple rate limiter: max_requests per window_seconds per key (usually IP)."""
    now = time.time()
    if key not in _rate_limits or now >= _rate_limits[key]['reset_at']:
        _rate_limits[key] = {'count': 1, 'reset_at': now + window_seconds}
        return True
    _rate_limits[key]['count'] += 1
    return _rate_limits[key]['count'] <= max_requests

def rate_limit(max_requests=100, window_seconds=60):
    """Decorator to apply rate limiting to endpoints."""
    def decorator(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            ip = request.remote_addr or 'unknown'
            if not _check_rate_limit(ip, max_requests, window_seconds):
                return jsonify({'error': 'Rate limit exceeded. Please try again later.'}), 429
            return f(*args, **kwargs)
        return wrapped
    return decorator

# Helper to decode JWT payload from Authorization header for role checks
def _decode_jwt_payload_from_header():
    try:
        auth_header = request.headers.get('Authorization', '')
        if not auth_header or not auth_header.startswith('Bearer '):
            return None
        token = auth_header.split(' ')[1]
        payload = jwt.decode(token, os.getenv('JWT_SECRET', 'supersecretkey'), algorithms=['HS256'])
        return payload
    except Exception:
        return None

# RBAC decorator: enforce role-based access
def role_required(*allowed_roles):
    """Decorator to enforce role-based access control.
    
    Example:
        @role_required('merchant', 'admin')
        def create_product():
            ...
    """
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            payload = _decode_jwt_payload_from_header()
            if not payload:
                return jsonify({'error': 'Unauthorized. Please provide a valid token.'}), 401
            user_role = payload.get('role')
            if user_role not in allowed_roles:
                return jsonify({'error': f'Forbidden. Required roles: {allowed_roles}'}), 403
            return f(*args, **kwargs)
        return wrapper
    return decorator

# Admin endpoints: list all products and users for admin panel
@app.route('/api/products', methods=['GET'])
@role_required('merchant', 'admin')
def get_all_products():
    try:
        if products_collection is None:
            return jsonify({'products': []})
        products = list(products_collection.find({}, {'_id':0}))
        return jsonify({'products': products})
    except Exception as e:
        return jsonify({'products': [], 'error': str(e)})

@app.route('/api/users', methods=['GET'])
@role_required('admin')
def get_all_users():
    try:
        if users_collection is None:
            return jsonify({'users': []})
        users = list(users_collection.find({}, {'_id':0}))
        return jsonify({'users': users})
    except Exception as e:
        return jsonify({'users': [], 'error': str(e)})


@app.route('/api/reports', methods=['GET'])
def get_reports():
    try:
        # If a dedicated reports collection exists, return it; otherwise synthesize from products flagged as reported
        # db is a pymongo Database object; use get_collection or db['reports'] rather than db.get()
        reports_col = None
        if 'db' in globals() and db is not None:
            try:
                reports_col = db.get_collection('reports') if hasattr(db, 'get_collection') else db['reports']
            except Exception:
                reports_col = None
        if reports_col is not None:
            reports = list(reports_col.find({}, {'_id':0}))
            return jsonify({'reports': reports})
        # fallback: look for reported field on products
        if products_collection is None:
            return jsonify({'reports': []})
        reports = []
        for p in products_collection.find({'reported': {'$exists': True}}, {'_id':0, 'reported':1, 'id':1}):
            rep = p.get('reported')
            if isinstance(rep, list):
                for r in rep:
                    reports.append({'product_id': p.get('id'), 'reason': r.get('reason') if isinstance(r, dict) else str(r), 'reporter': r.get('reporter') if isinstance(r, dict) else None, 'created_at': r.get('created_at') if isinstance(r, dict) else None})
            else:
                reports.append({'product_id': p.get('id'), 'reason': str(rep)})
        return jsonify({'reports': reports})
    except Exception as e:
        print('⚠️ get_reports error:', e)
        return jsonify({'reports': [], 'error': str(e)})


@app.route('/api/alerts', methods=['GET'])
def get_alerts():
    try:
        # Try to get alerts collection safely
        alerts_col = None
        if 'db' in globals() and db is not None:
            try:
                alerts_col = db.get_collection('alerts') if hasattr(db, 'get_collection') else db['alerts']
            except Exception:
                alerts_col = None
        if alerts_col is not None:
            alerts = list(alerts_col.find({}, {'_id':0}))
            return jsonify({'alerts': alerts})
        # Fallback: compute simple alert counts by product from products_collection
        if products_collection is None:
            return jsonify({'alerts': []})
        pipeline = [
            {'$match': {'alert_count': {'$exists': True}}},
            {'$project': {'product_id': '$id', 'count': '$alert_count', 'latest': '$alert_latest', 'product': {'name': '$name'}}}
        ]
        alerts = list(products_collection.aggregate(pipeline)) if hasattr(products_collection, 'aggregate') else []
        return jsonify({'alerts': alerts})
    except Exception as e:
        print('⚠️ get_alerts error:', e)
        return jsonify({'alerts': [], 'error': str(e)})


@app.route('/api/seller-activity', methods=['GET'])
def get_seller_activity():
    try:
        # Provide lightweight analytics: total products, recent registrations, top sellers
        total = 0
        recent = []
        top_sellers = []
        if products_collection is not None:
            total = products_collection.count_documents({}) if hasattr(products_collection, 'count_documents') else len(list(products_collection.find({})))
            # recent 5
            for p in products_collection.find({}, {'_id':0, 'name':1, 'brand':1, 'registered_at':1}).sort([('registered_at', -1)])[:5]:
                recent.append(p.get('name') or p.get('id'))
            # aggregate top sellers by seller field
            try:
                top = products_collection.aggregate([
                    {'$group': {'_id': '$seller', 'count': {'$sum': 1}}},
                    {'$sort': {'count': -1}},
                    {'$limit': 5}
                ])
                top_sellers = [{'seller': t.get('_id'), 'count': t.get('count')} for t in top]
            except Exception:
                top_sellers = []
        return jsonify({'count': total, 'recent': recent, 'top_sellers': top_sellers})
    except Exception as e:
        print('⚠️ seller_activity error:', e)
        return jsonify({'count': 0, 'recent': [], 'error': str(e)})


@app.route('/api/admin/user/update-role', methods=['POST'])
def admin_update_user_role():
    data = request.get_json() or {}
    email = data.get('email')
    new_role = data.get('role')
    if not email or not new_role:
        return jsonify({'error': 'email and role required'}), 400
    try:
        if users_collection is None:
            return jsonify({'error': 'User collection not available'}), 500
        result = users_collection.update_one({'email': email}, {'$set': {'role': new_role}})
        if result.matched_count == 0:
            return jsonify({'error': 'User not found'}), 404
        return jsonify({'success': True, 'email': email, 'role': new_role})
    except Exception as e:
        print('⚠️ admin_update_user_role error:', e)
        return jsonify({'error': str(e)}), 500


@app.route('/api/admin/user/delete', methods=['POST'])
def admin_delete_user():
    data = request.get_json() or {}
    email = data.get('email')
    if not email:
        return jsonify({'error': 'email required'}), 400
    try:
        if users_collection is None:
            return jsonify({'error': 'User collection not available'}), 500
        result = users_collection.delete_one({'email': email})
        if result.deleted_count == 0:
            return jsonify({'error': 'User not found'}), 404
        return jsonify({'success': True, 'email': email})
    except Exception as e:
        print('⚠️ admin_delete_user error:', e)
        return jsonify({'error': str(e)}), 500


@app.route('/api/admin/product/delete', methods=['POST'])
def admin_delete_product():
    data = request.get_json() or {}
    pid = data.get('product_id') or data.get('id')
    if not pid:
        return jsonify({'error': 'product_id required'}), 400
    try:
        if products_collection is None:
            return jsonify({'error': 'Products collection not available'}), 500
        # try delete by id field or _id
        res = products_collection.delete_one({'id': pid})
        if res.deleted_count == 0:
            # try by _id if possible
            try:
                from bson import ObjectId
                res = products_collection.delete_one({'_id': ObjectId(pid)})
            except Exception:
                res = None
        if res and res.deleted_count == 0:
            return jsonify({'error': 'Product not found'}), 404
        return jsonify({'success': True, 'product_id': pid})
    except Exception as e:
        print('⚠️ admin_delete_product error:', e)
        return jsonify({'error': str(e)}), 500

# ---------------------- Configuration ----------------------
# You can place these in your .env
# MONGODB_URI (preferred) OR MONGO_USER & MONGO_PASS & MONGO_HOST (alternative)
# MONGO_DBNAME (default: brandauth)
# MONGO_ALLOW_INSECURE_TLS (set to "true" to allow invalid certs during local dev)
# INFURA_URL, CHAIN_ID, PRIVATE_KEY, CONTRACT_ADDRESS, JWT_SECRET, PORT

MONGO_URI = os.getenv('MONGODB_URI')  # recommended: mongodb+srv://user:pass@cluster.../brandauth?...
MONGO_USER = os.getenv('MONGO_USER')
MONGO_PASS = os.getenv('MONGO_PASS')
MONGO_HOST = os.getenv('MONGO_HOST')  # e.g. cluster0.6o36esx.mongodb.net
MONGO_DBNAME = os.getenv('MONGO_DBNAME', 'brandauth')
MONGO_ALLOW_INSECURE_TLS = os.getenv('MONGO_ALLOW_INSECURE_TLS', 'false').lower() in ('1', 'true', 'yes')

# ---------------------- NLP Model Loading ----------------------
NLP_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'data', 'text', 'nlp_model.joblib')
VECTORIZER_PATH = os.path.join(os.path.dirname(__file__), 'data', 'text', 'tfidf_vectorizer.joblib')

nlp_model = None
tfidf_vectorizer = None
sb_clf = None
sb_model = None
sb_embedder = None
meta_clf = None
META_CLF_PATH = os.path.join(os.path.dirname(__file__), 'data', 'meta_clf.joblib')

def safe_joblib_load(path):
    try:
        return joblib.load(path)
    except Exception as e:
        print(f"⚠️ Warning: Failed to load '{path}': {e}")
        return None

nlp_model = safe_joblib_load(NLP_MODEL_PATH)
tfidf_vectorizer = safe_joblib_load(VECTORIZER_PATH)
if nlp_model and tfidf_vectorizer:
    print("✅ NLP model and vectorizer loaded.")
else:
    print("⚠️ NLP model and/or vectorizer not loaded (will return error if endpoints called).")

# try to load sbert classifier if present
SB_CLF_PATH = os.path.join(os.path.dirname(__file__), 'data', 'text', 'sbert_clf.joblib')
SB_MODEL_NAME_PATH = os.path.join(os.path.dirname(__file__), 'data', 'text', 'sbert_model_name.txt')
# Load SBERT classifier metadata if present, but DO NOT import heavy libraries at startup.
# Heavy imports (sentence_transformers / transformers) cause long startup times and can hang.
if os.path.exists(SB_CLF_PATH):
    try:
        sb_clf = safe_joblib_load(SB_CLF_PATH)
    except Exception as e:
        print("⚠️ Could not load SBERT classifier file:", e)
        sb_clf = None
    if os.path.exists(SB_MODEL_NAME_PATH):
        try:
            with open(SB_MODEL_NAME_PATH, 'r', encoding='utf-8') as f:
                sb_model = f.read().strip()
        except Exception as e:
            print("⚠️ Could not read SBERT model name file:", e)
            sb_model = None
    else:
        sb_model = None
    # Do not preload the SentenceTransformer here; instantiate lazily inside analysis if needed.
    sb_embedder = None
    print("ℹ️ SBERT classifier metadata loaded (embedder will be lazy-loaded):", "model=", sb_model, "clf_loaded=", sb_clf is not None)
else:
    sb_clf = None
    sb_model = None
    sb_embedder = None

# try loading meta classifier for fusion
if os.path.exists(META_CLF_PATH):
    try:
        meta_clf = safe_joblib_load(META_CLF_PATH)
        print("✅ Meta classifier loaded for multimodal fusion:", meta_clf is not None)
    except Exception as e:
        print("⚠️ Could not load meta classifier:", e)

def clean_text(text):
    text = (text or "").lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ---------------------- Multimodal helpers (CV + NLP) ----------------------
def _ensure_cv_model():
    global cv_model
    try:
        if cv_model is None and os.path.exists(CV_MODEL_PATH):
            print(f"ℹ️ Loading CV model from {CV_MODEL_PATH}")
            cv_model = tf.keras.models.load_model(CV_MODEL_PATH)
            print("✅ CV model loaded")
    except Exception as e:
        print("⚠️ Could not load CV model:", e)
        cv_model = None


def _ensure_nlp_models():
    global nlp_model, tfidf_vectorizer
    try:
        if nlp_model is None and os.path.exists(NLP_MODEL_PATH):
            nlp_model = safe_joblib_load(NLP_MODEL_PATH)
            print("✅ NLP model loaded")
    except Exception as e:
        print("⚠️ Could not load NLP model:", e)
        nlp_model = None
    try:
        if tfidf_vectorizer is None and os.path.exists(VECTORIZER_PATH):
            tfidf_vectorizer = safe_joblib_load(VECTORIZER_PATH)
            print("✅ TF-IDF vectorizer loaded")
    except Exception as e:
        print("⚠️ Could not load TF-IDF vectorizer:", e)
        tfidf_vectorizer = None


def _preprocess_pil_image(img_pil):
    try:
        img = img_pil.convert('RGB')
        img = img.resize((CV_IMG_SIZE, CV_IMG_SIZE))
        arr = np.asarray(img).astype('float32') / 255.0
        if len(arr.shape) == 3:
            arr = np.expand_dims(arr, axis=0)
        return arr
    except Exception as e:
        print('⚠️ Image preprocessing failed:', e)
        return None


def predict_cv_from_pil(img_pil):
    _ensure_cv_model()
    if cv_model is None:
        return None
    arr = _preprocess_pil_image(img_pil)
    if arr is None:
        return None
    try:
        pred = cv_model.predict(arr)
        # handle different output shapes
        score = float(pred[0][0]) if hasattr(pred[0], '__iter__') else float(pred[0])
        return max(0.0, min(1.0, score))
    except Exception as e:
        print('⚠️ CV prediction failed:', e)
        return None


def predict_nlp_scores(text):
    _ensure_nlp_models()
    if nlp_model is None or tfidf_vectorizer is None:
        return None, []
    clean = clean_text(text or '')
    try:
        vec = tfidf_vectorizer.transform([clean])
        if hasattr(nlp_model, 'predict_proba'):
            prob = float(nlp_model.predict_proba(vec)[0][1])
        else:
            # fallback to predict (0 or 1)
            pred = nlp_model.predict(vec)[0]
            prob = 1.0 if pred else 0.0
    except Exception as e:
        print('⚠️ NLP prediction failed:', e)
        prob = None

    # sentence-level scores
    sentences = []
    sent_scores = []
    try:
        raw_sents = re.split(r'[\.\!\?]+', text or '')
        for s in raw_sents:
            s = s.strip()
            if not s or len(s) < 4:
                continue
            sentences.append(s)
            try:
                vecs = tfidf_vectorizer.transform([clean_text(s)])
                if hasattr(nlp_model, 'predict_proba'):
                    sprob = float(nlp_model.predict_proba(vecs)[0][1])
                else:
                    sp = nlp_model.predict(vecs)[0]
                    sprob = 1.0 if sp else 0.0
            except Exception:
                sprob = None
            sent_scores.append({'text': s, 'score': sprob})
    except Exception as e:
        print('⚠️ Sentence scoring failed:', e)

    return prob, sent_scores


def fuse_scores(cv_score, nlp_score):
    # if a meta classifier is provided, prefer it
    try:
        if meta_clf is not None:
            arr = np.array([[cv_score if cv_score is not None else 0.0, nlp_score if nlp_score is not None else 0.0]])
            try:
                prob = float(meta_clf.predict_proba(arr)[0][1])
                return max(0.0, min(1.0, prob))
            except Exception:
                pass
    except Exception:
        pass
    # fallback to weighted average
    cv_val = cv_score if cv_score is not None else 0.0
    nlp_val = nlp_score if nlp_score is not None else 0.0
    total_w = MULTIMODAL_CV_WEIGHT + MULTIMODAL_NLP_WEIGHT
    if total_w == 0:
        return 0.0
    fused = (cv_val * MULTIMODAL_CV_WEIGHT + nlp_val * MULTIMODAL_NLP_WEIGHT) / total_w
    return max(0.0, min(1.0, float(fused)))


# Simple sentence splitter (naive). Used when analyzing multi-sentence reviews.
SENT_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')

def split_into_sentences(text):
    if not isinstance(text, str):
        return []
    parts = [s.strip() for s in SENT_SPLIT_RE.split(text) if s.strip()]
    return parts


def analyze_review_text(review):
    """Analyze review text at sentence level.
    Prefer SBERT classifier when available; otherwise fall back to TF-IDF + logistic NLP model.
    Returns dict with keys: sentences (list of {text,label,score,suspicious}), suspicious_fraction,
    overall_suspicious, top_suspicious, nlp_authentic_prob
    """
    sentences = split_into_sentences(review)
    if not sentences:
        sentences = [review]

    # If SBERT classifier is available, try to use it (lazy-load embedder). If it fails, fallback.
    use_sbert = sb_clf is not None and sb_model is not None
    probs = None
    sentence_results = []
    suspicious_count = 0

    if use_sbert:
        try:
            # lazy-load embedder if needed
            if sb_embedder is None:
                try:
                    from sentence_transformers import SentenceTransformer
                    # assign to global for reuse
                    globals()['sb_embedder'] = SentenceTransformer(sb_model)
                    print("✅ Lazy-loaded SBERT embedder for analysis")
                except Exception as e:
                    print("⚠️ Could not lazy-load SBERT embedder, falling back to TF-IDF NLP:", e)
                    use_sbert = False

            if use_sbert:
                embedder = globals().get('sb_embedder')
                embs = embedder.encode(sentences, convert_to_numpy=True)
                labels = sb_clf.predict(embs)
                try:
                    probs = sb_clf.predict_proba(embs)
                except Exception:
                    probs = None

                for i, s in enumerate(sentences):
                    lab = str(labels[i])
                    sc = None
                    if probs is not None:
                        classes = list(sb_clf.classes_)
                        try:
                            idx = classes.index(labels[i])
                            sc = float(probs[i][idx])
                        except Exception:
                            sc = None
                    susp = lab.lower() in ['counterfeit', 'fake']
                    if susp:
                        suspicious_count += 1
                    sentence_results.append({'text': s, 'label': lab, 'score': round(sc, 3) if sc is not None else None, 'suspicious': susp})

        except Exception as e:
            print('⚠️ SBERT analysis failed, falling back to TF-IDF NLP:', e)
            use_sbert = False

    # TF-IDF / logistic fallback (safe and fast)
    if not use_sbert:
        if tfidf_vectorizer is None or nlp_model is None:
            raise RuntimeError('NLP vectorizer or model not available for fallback analysis')
        try:
            sent_vecs = tfidf_vectorizer.transform(sentences)
            if hasattr(nlp_model, 'predict_proba'):
                sent_probs = nlp_model.predict_proba(sent_vecs)
                classes = list(nlp_model.classes_)
                # Try to determine which class index maps to "authentic"
                lower_classes = [str(c).lower() for c in classes]
                if 'authentic' in lower_classes:
                    auth_idx = lower_classes.index('authentic')
                elif '1' in lower_classes:
                    auth_idx = lower_classes.index('1')
                else:
                    # fallback: choose class with higher average probability across sentences
                    avg_probs = np.mean(sent_probs, axis=0)
                    auth_idx = int(np.argmax(avg_probs))

                auth_probs = []
                for i, s in enumerate(sentences):
                    auth_prob = float(sent_probs[i][auth_idx])
                    auth_probs.append(auth_prob)
                    label = 'authentic' if auth_prob >= 0.5 else 'counterfeit'
                    susp = label != 'authentic'
                    if susp:
                        suspicious_count += 1
                    sentence_results.append({'text': s, 'label': label, 'score': round(auth_prob, 3), 'suspicious': susp})

                nlp_auth_prob = float(sum(auth_probs) / len(auth_probs)) if auth_probs else None
                probs = sent_probs
            else:
                preds = nlp_model.predict(sent_vecs)
                for i, p in enumerate(preds):
                    label = str(p)
                    susp = label.lower() not in ('authentic', '1', 'true')
                    if susp:
                        suspicious_count += 1
                    sentence_results.append({'text': sentences[i], 'label': label, 'score': None, 'suspicious': susp})
                nlp_auth_prob = 1.0 - (suspicious_count / len(sentence_results))
        except Exception as e:
            print('⚠️ TF-IDF NLP fallback failed:', e)
            # safe default: mark whole text as unknown
            sentence_results = [{'text': s, 'label': 'unknown', 'score': None, 'suspicious': False} for s in sentences]
            nlp_auth_prob = None

    # compute summary values
    frac = suspicious_count / len(sentence_results) if sentence_results else 0.0
    overall = frac >= 0.3

    # compute nlp_auth_prob if not set earlier
    try:
        if 'nlp_auth_prob' in locals():
            nlp_auth_prob = locals()['nlp_auth_prob']
        elif 'nlp_auth_prob' not in locals() and 'nlp_auth_prob' not in globals():
            # if we set nlp_auth_prob above during TF-IDF path use that
            pass
    except Exception:
        pass

    if 'nlp_auth_prob' not in locals() and 'nlp_auth_prob' not in globals():
        # attempt to compute from probs if available
        nlp_auth_prob = None
        try:
            if probs is not None:
                # attempt to compute authentic probability by subtracting counterfeit prob if possible
                classes = None
                if use_sbert and sb_clf is not None:
                    classes = list(sb_clf.classes_)
                elif nlp_model is not None and hasattr(nlp_model, 'classes_'):
                    classes = list(nlp_model.classes_)
                if classes is not None:
                    counterfeit_inds = [i for i, c in enumerate(classes) if str(c).lower() in ('counterfeit', 'fake')]
                    if counterfeit_inds:
                        counterfeit_probs = [sum([probs[r][i] for i in counterfeit_inds]) for r in range(len(probs))]
                        avg_counterfeit = float(sum(counterfeit_probs) / len(counterfeit_probs))
                        nlp_auth_prob = 1.0 - avg_counterfeit
                    else:
                        nlp_auth_prob = 1.0
        except Exception:
            nlp_auth_prob = None

    # final fallback
    if nlp_auth_prob is None:
        n_susp = sum(1 for r in sentence_results if r.get('suspicious'))
        nlp_auth_prob = 1.0 - (n_susp / len(sentence_results)) if sentence_results else None

    top_susp = sorted([r for r in sentence_results if r.get('suspicious')], key=lambda x: (x.get('score') or 0), reverse=True)

    return {
        'sentences': sentence_results,
        'suspicious_fraction': round(frac, 3),
        'overall_suspicious': overall,
        'top_suspicious': top_susp[:3],
        'nlp_authentic_prob': round(nlp_auth_prob, 4) if nlp_auth_prob is not None else None
    }

# ---------------------- MongoDB Connection (patched for SSL) ----------------------
print("ℹ️ Loading MongoDB configuration...")

try:
    from pymongo import MongoClient
    import certifi
except ImportError:
    raise ImportError("Please install pymongo and certifi: pip install pymongo certifi")

MONGO_URI = os.getenv("MONGODB_URI")
MONGO_DBNAME = os.getenv("MONGO_DBNAME", "brandauth")
MONGO_ALLOW_INSECURE_TLS = os.getenv("MONGO_ALLOW_INSECURE_TLS", "false").lower() in ("1", "true", "yes")

mongo_client = None
db = None
users_collection = None
products_collection = None
_mongo_connected = False
_mongo_connect_error = None
influencer_events_collection = None
influencer_settings_collection = None
role_changes_collection = None

def try_connect_mongo():
    """Try connecting to MongoDB Atlas with proper TLS/SSL handling."""
    global mongo_client, db, users_collection, products_collection, _mongo_connected, _mongo_connect_error, influencer_events_collection, influencer_settings_collection, role_changes_collection
    try:
        if not MONGO_URI:
            raise ValueError("MONGODB_URI not set in .env")

        print("🔗 Connecting to MongoDB Atlas...")

        # Default options: secure SSL via certifi
        kwargs = {"tlsCAFile": certifi.where(), "tls": True}

        # Workaround for local SSL handshake failures
        if MONGO_ALLOW_INSECURE_TLS:
            print("⚠️ Allowing invalid TLS certificates for local development.")
            kwargs["tlsAllowInvalidCertificates"] = True

        mongo_client = MongoClient(MONGO_URI, **kwargs)

        # Simple ping test
        mongo_client.admin.command("ping")

        db = mongo_client.get_database(MONGO_DBNAME)
        users_collection = db["users"]
        products_collection = db["products"]
        influencer_events_collection = db["influencer_events"]
        influencer_settings_collection = db["influencer_settings"]
        role_changes_collection = db["role_changes"]
        _mongo_connected = True
        print(f"✅ Connected to MongoDB database: {MONGO_DBNAME}")

    except Exception as e:
        _mongo_connected = False
        _mongo_connect_error = str(e)
        print("❌ MongoDB connection failed:", e)
        print("👉 Check: IP Access, TLS settings, or .env file.")

try_connect_mongo()


# ---------------------- Web3 Connection ----------------------
INFURA_URL = os.getenv('INFURA_URL') or "http://127.0.0.1:8545"
w3 = Web3(Web3.HTTPProvider(INFURA_URL))
print("Web3 provider:", INFURA_URL, "Connected:", w3.is_connected())

chain_id = int(os.getenv('CHAIN_ID') or 11155111)
private_key = os.getenv('PRIVATE_KEY')
contract_address = os.getenv('CONTRACT_ADDRESS')

contract = None
if contract_address:
    try:
        ABI_PATH = os.path.join(os.path.dirname(__file__), 'contracts', 'ProductRegistry.abi.json')
        with open(ABI_PATH, 'r') as f:
            contract_abi = json.load(f)
        contract_address = Web3.to_checksum_address(contract_address.strip())
        contract = w3.eth.contract(address=contract_address, abi=contract_abi)
        print("✅ Contract loaded at", contract_address)
    except Exception as e:
        print("⚠️ Warning: Could not load contract:", str(e))

def _require_blockchain_ready():
    if not w3.is_connected():
        raise RuntimeError("Web3 provider not connected.")
    if not private_key:
        raise RuntimeError("PRIVATE_KEY not set.")
    if not contract:
        raise RuntimeError("CONTRACT_ADDRESS or ABI missing.")

# ---------------------- Blockchain Helper ----------------------
def register_on_blockchain(product_data):
    _require_blockchain_ready()
    addr = w3.eth.account.from_key(private_key).address
    base_gas_price = w3.to_wei('10', 'gwei')
    for attempt in range(5):
        try:
            nonce = w3.eth.get_transaction_count(addr)
            pid = product_data['id']
            if isinstance(pid, str) and pid.startswith('0x'):
                product_id_bytes = bytes.fromhex(pid[2:])
            else:
                product_id_bytes = bytes.fromhex(pid)
            tx = contract.functions.registerProduct(
                product_id_bytes,
                product_data['name'],
                product_data['brand'],
                int(datetime.datetime.now().timestamp())
            ).build_transaction({
                'chainId': chain_id,
                'gas': 200000,
                'gasPrice': base_gas_price + attempt * w3.to_wei('5', 'gwei'),
                'nonce': nonce,
            })
            signed_tx = w3.eth.account.sign_transaction(tx, private_key)
            tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
            receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
            return receipt
        except Exception as e:
            err_str = str(e).lower()
            if (('replacement transaction underpriced' in err_str) or ('nonce too low' in err_str)) and attempt < 4:
                print(f"Retrying due to: {str(e)}")
                time.sleep(5)
                continue
            raise

# ---------------------- Register Product ----------------------
def register_product_internal(data):
    # shared internal logic that expects validated data
    product_id = Web3.keccak(text=f"{data['name']}{data['brand']}{datetime.datetime.now()}").hex()
    product = {
        'id': product_id,
        'name': data['name'],
        'brand': data['brand'],
        'product_url': data.get('product_url', ''),
        'category': data.get('category', ''),
        'batchNumber': data.get('batchNumber', ''),
        'manufactureDate': data.get('manufactureDate', ''),
        'registration_date': datetime.datetime.now().isoformat()
    }
    receipt = register_on_blockchain(product)
    qr_data = {
        'product_id': product_id,
        'name': product['name'],
        'brand': product['brand'],
        'product_url': product.get('product_url'),
        'category': product['category'],
        'batchNumber': product['batchNumber'],
        'manufactureDate': product['manufactureDate'],
        'tx_hash': receipt.transactionHash.hex(),
        'contract_address': contract_address,
        'verification_url': f"http://localhost:3000/verify/{product_id}"
    }
    if products_collection is not None:
        try:
            products_collection.insert_one({**product, **qr_data})
        except Exception as e:
            print("⚠️ Warning: Could not insert product to DB:", e)
    return {'qr_data': qr_data, 'blockchain_receipt': {'tx_hash': receipt.transactionHash.hex(), 'block_number': receipt.blockNumber}}


def compute_image_phash(pil_image):
    """Compute perceptual hash (phash) for an image if imagehash is installed."""
    try:
        if _imagehash is None:
            return None
        return str(_imagehash.phash(pil_image))
    except Exception as e:
        print('Image phash error:', e)
        return None


def run_ocr_on_image(pil_image):
    """Run OCR using pytesseract if available; returns extracted text or None."""
    try:
        if _pytesseract is None:
            return None
        return _pytesseract.image_to_string(pil_image)

    except Exception as e:
        print('OCR error:', e)
        return None


def extract_image_features(pil_image, file_storage=None):
    """Return a dictionary of image qualities useful for verification and display.
    If `file_storage` (werkzeug FileStorage) is provided, attempt to read file size.
    """
    features = {}
    try:
        img = pil_image
        features['dimensions'] = {'width': img.width, 'height': img.height}
        features['format'] = getattr(img, 'format', None)
        # File size if available
        try:
            if file_storage is not None and hasattr(file_storage, 'content_length') and file_storage.content_length:
                features['file_size_bytes'] = file_storage.content_length
            else:
                # estimate by saving to bytes
                bio = io.BytesIO()
                img.save(bio, format='JPEG')
                features['file_size_bytes'] = bio.tell()
        except Exception:
            features['file_size_bytes'] = None

        # Perceptual hash
        try:
            features['phash'] = compute_image_phash(img)
        except Exception:
            features['phash'] = None

        # Mean and stddev color
        try:
            arr_stats = np.array(img.convert('RGB')).astype('float32') / 255.0
            if arr_stats.ndim == 3:
                features['mean_color'] = [float(np.mean(arr_stats[:, :, i])) for i in range(3)]
                features['stddev_color'] = [float(np.std(arr_stats[:, :, i])) for i in range(3)]
            else:
                features['mean_color'] = None
                features['stddev_color'] = None
        except Exception:
            features['mean_color'] = None
            features['stddev_color'] = None

        # Barcode/QR detection
        try:
            features['barcodes'] = decode_barcodes(img)
        except Exception:
            features['barcodes'] = []

        # OCR
        try:
            features['ocr_text'] = run_ocr_on_image(img)
        except Exception:
            features['ocr_text'] = None

        # EXIF and tampering heuristics
        try:
            exif = img._getexif() if hasattr(img, '_getexif') else None
            if exif:
                features['exif'] = {str(k): str(v) for k, v in exif.items()}
                suspicious_tags = ['Software', 'ProcessingSoftware', 'ImageHistory']
                suspicious = any(tag in features['exif'] for tag in suspicious_tags)
                features['tampering_flag'] = suspicious
                features['tampering_reason'] = 'EXIF indicates editing' if suspicious else None
            else:
                features['exif'] = None
                features['tampering_flag'] = True
                features['tampering_reason'] = 'Missing EXIF metadata (possible screenshot or edit).'
        except Exception:
            features['exif'] = None
            features['tampering_flag'] = True
            features['tampering_reason'] = 'Error reading EXIF metadata.'

    except Exception as e:
        print('extract_image_features error:', e)
        traceback.print_exc()
        return {}

    return features

# --- Barcode/QR Verification Endpoint ---
@app.route('/api/verify-barcode', methods=['POST'])
def verify_barcode():
    data = request.get_json() or {}
    code = data.get('code')
    if not code:
        return jsonify({'error': 'No barcode/QR code provided'}), 400
    # Example: lookup product in MongoDB and blockchain
    try:
        mongo = MongoClient(MONGO_URI, tlsCAFile=certifi.where())
        db = mongo[MONGO_DBNAME]
        product = db.products.find_one({'barcode': code})
        result = {'barcode': code}
        if product:
            result['product'] = {
                'name': product.get('name'),
                'brand': product.get('brand'),
                'registered': True,
                'blockchain_tx': product.get('blockchain_tx'),
                'contract_address': contract_address,
            }
            # Optionally, verify on blockchain
            if contract and product.get('blockchain_tx'):
                result['blockchain'] = {
                    'status': 'verified',
                    'tx_hash': product.get('blockchain_tx'),
                    'contract_address': contract_address,
                }
            else:
                result['blockchain'] = {'status': 'not found'}
        else:
            result['product'] = {'registered': False}
            result['blockchain'] = {'status': 'not found'}
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


def decode_barcodes(pil_image):
    """Decode barcodes/QR codes using pyzbar if available."""
    try:
        if _pyzbar is None:
            return []
        decoded = _pyzbar.decode(pil_image)
        results = []
        for d in decoded:
            results.append({'type': d.type, 'data': d.data.decode('utf-8', errors='ignore')})
        return results
    except Exception as e:
        print('Barcode decode error:', e)
        return []


def save_product_image(product_id, file_storage):
    """Save uploaded product image to data/images/products/<product_id>/ and return metadata."""
    base_dir = os.path.join(os.path.dirname(__file__), 'data', 'images', 'products', product_id)
    os.makedirs(base_dir, exist_ok=True)
    fname = f"{uuid.uuid4().hex}.jpg"
    path = os.path.join(base_dir, fname)
    try:
        img = Image.open(file_storage.stream).convert('RGB')
        img.save(path, format='JPEG', quality=90)
        phash = compute_image_phash(img)
        ocr = run_ocr_on_image(img)
        barcodes = decode_barcodes(img)
        return {'filename': path, 'phash': phash, 'ocr': ocr, 'barcodes': barcodes}
    except Exception as e:
        print('Save image error:', e)
        return {'error': str(e)}


@app.route('/api/register_detailed', methods=['POST'])
def register_detailed():
    """Register a product with detailed metadata and multiple images.
    Accepts multipart/form-data with fields:
      - name (required), brand (required)
      - description, serial_number, batchNumber, manufactureDate, manufacturer, origin_country
      - seller_info (JSON string), geolocation (lat,lng as JSON or fields lat & lng)
      - image files under field name 'images' (can be multiple)
    Returns product id and stored metadata.
    """
    try:
        form = request.form
        name = form.get('name')
        brand = form.get('brand')
        if not name or not brand:
            return jsonify({'error': 'name and brand are required'}), 400

        data = {
            'name': name,
            'brand': brand,
            'description': form.get('description'),
            'product_url': form.get('product_url'),
            'serial_number': form.get('serial_number'),
            'batchNumber': form.get('batchNumber'),
            'manufactureDate': form.get('manufactureDate'),
            'manufacturer': form.get('manufacturer'),
            'origin_country': form.get('origin_country'),
            'seller_info': None,
            'geolocation': None,
        }

        # parse seller_info JSON if provided
        s = form.get('seller_info')
        if s:
            try:
                data['seller_info'] = _json.loads(s)
            except Exception:
                data['seller_info'] = {'raw': s}

        # parse geolocation
        geo = form.get('geolocation')
        if geo:
            try:
                data['geolocation'] = _json.loads(geo)
            except Exception:
                data['geolocation'] = {'raw': geo}
        else:
            lat = form.get('lat'); lng = form.get('lng')
            if lat and lng:
                try:
                    data['geolocation'] = {'lat': float(lat), 'lng': float(lng)}
                except Exception:
                    data['geolocation'] = {'lat': lat, 'lng': lng}

        # process images
        images_meta = []
        files = request.files.getlist('images') or []
        for f in files:
            info = save_product_image('temp', f)
            images_meta.append(info)

        # create product id and store product document (without blockchain for initial)
        product_id = Web3.keccak(text=f"{name}{brand}{datetime.datetime.now()}").hex()
        product_doc = {
            'id': product_id,
            'name': name,
            'brand': brand,
            'metadata': data,
            'images': images_meta,
            'registered_at': datetime.datetime.utcnow().isoformat(),
            'reports': [],
        }

        # Move saved images into product-specific folder (if save_product_image used 'temp')
        # We already saved them under data/images/products/temp; rename/move into product id folder
        try:
            src_dir = os.path.join(os.path.dirname(__file__), 'data', 'images', 'products', 'temp')
            dst_dir = os.path.join(os.path.dirname(__file__), 'data', 'images', 'products', product_id)
            if os.path.isdir(src_dir):
                os.makedirs(dst_dir, exist_ok=True)
                for fname in os.listdir(src_dir):
                    try:
                        os.replace(os.path.join(src_dir, fname), os.path.join(dst_dir, fname))
                    except Exception:
                        pass
                # update image paths in product_doc
                for im in product_doc['images']:
                    if im.get('filename'):
                        im['filename'] = im['filename'].replace(os.path.join('images', 'products', 'temp'), os.path.join('images', 'products', product_id))
        except Exception as e:
            print('Image move warning:', e)

        # insert into DB
        if products_collection is not None:
            try:
                products_collection.insert_one(product_doc)
            except Exception as e:
                print('DB insert error (register_detailed):', e)

        # optionally register a product on blockchain (if requested via form 'register_blockchain'='1')
        if form.get('register_blockchain') == '1':
            try:
                bc_receipt = register_on_blockchain(product_doc)
                product_doc['blockchain'] = {'tx_hash': bc_receipt.transactionHash.hex(), 'block_number': bc_receipt.blockNumber}
            except Exception as be:
                print('Blockchain register failed:', be)

        return jsonify({'success': True, 'product': product_doc})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/product/<product_id>', methods=['GET'])
def get_product(product_id):
    try:
        if products_collection is None:
            return jsonify({'error': 'DB not available'}), 500
        prod = products_collection.find_one({'id': product_id})
        if not prod:
            return jsonify({'error': 'Product not found'}), 404
        # hide internal ObjectId before returning
        prod.pop('_id', None)
        return jsonify({'product': prod})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/report_suspicious', methods=['POST'])
def report_suspicious():
    """Allow users or influencers to report a product as suspicious.
    POST JSON: { product_id, reporter_name, reporter_handle, reason, evidence_images (optional list of base64) }
    """
    try:
        data = request.get_json() or {}
        pid = data.get('product_id')
        if not pid:
            return jsonify({'error': 'product_id required'}), 400
        report = {
            'report_id': uuid.uuid4().hex,
            'product_id': pid,
            'reporter_name': data.get('reporter_name'),
            'reporter_handle': data.get('reporter_handle'),
            'reason': data.get('reason'),
            'evidence': [],
            'created_at': datetime.datetime.utcnow().isoformat()
        }
        # store evidence images if provided as base64
        ev_images = data.get('evidence_images') or []
        for b64 in ev_images:
            try:
                raw = base64.b64decode(b64)
                img = Image.open(io.BytesIO(raw)).convert('RGB')
                path = os.path.join(os.path.dirname(__file__), 'data', 'reports', f"{uuid.uuid4().hex}.jpg")
                os.makedirs(os.path.dirname(path), exist_ok=True)
                img.save(path, format='JPEG', quality=80)
                report['evidence'].append(path)
            except Exception as e:
                print('evidence save error:', e)

        # insert into a reports collection
        try:
            reports_col = db.get_collection('reports') if db is not None else None
            if reports_col is not None:
                reports_col.insert_one(report)
            # also add minimal report reference into the product doc
            if products_collection is not None:
                products_collection.update_one({'id': pid}, {'$push': {'reports': {'report_id': report['report_id'], 'reason': report['reason'], 'reporter': report.get('reporter_handle'), 'created_at': report['created_at']}}})
        except Exception as e:
            print('report insert error:', e)

        return jsonify({'success': True, 'report': report})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/influencer_activity', methods=['GET'])
def influencer_activity():
    """Aggregate reports by reporter_handle to show influencers/reporters who flagged products frequently."""
    try:
        reports_col = db.get_collection('reports') if db is not None else None
        if reports_col is None:
            return jsonify({'error': 'reports collection not available'}), 500
        pipeline = [
            {'$group': {'_id': '$reporter_handle', 'count': {'$sum': 1}, 'latest': {'$max': '$created_at'}}},
            {'$sort': {'count': -1}}
        ]
        agg = list(reports_col.aggregate(pipeline))
        return jsonify({'influencers': agg})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# ---------------------- Endpoints ----------------------
@app.route('/api/health_check', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'ok',
        'message': '✅ Authentication server is running',
        'mongo_connected': _mongo_connected,
        'mongo_error': _mongo_connect_error,
        'web3_connected': w3.is_connected(),
        'contract_loaded': contract is not None,
        'rbac_enabled': True,
        'endpoints_available': [
            '/api/health_check',
            '/api/signup',
            '/api/login',
            '/api/multimodal_check (requires auth)',
            '/api/influencer/* (influencer verification)',
            '/api/products (requires auth)'
        ]
    })

@app.route('/api/signup', methods=['POST'])
def signup():
    data = request.get_json() or {}
    email = data.get('email')
    password = data.get('password')
    phone = data.get('phone')
    role = data.get('role')
    allowed_roles = ['seller', 'buyer', 'influencer']
    # Require email, password, and valid role
    if not email or not password or not role:
        return jsonify({'error': 'Email, password, and role are required'}), 400
    if role not in allowed_roles:
        return jsonify({'error': f"Role must be one of: {', '.join(allowed_roles)}"}), 400
    if users_collection is not None and users_collection.find_one({'email': email}):
        return jsonify({'error': 'Email exists'}), 409
    pw_errors = []
    if len(password) < 8:
        pw_errors.append('at least 8 characters')
    if not re.search(r'[A-Z]', password):
        pw_errors.append('an uppercase letter')
    if not re.search(r'[a-z]', password):
        pw_errors.append('a lowercase letter')
    if not re.search(r'[0-9]', password):
        pw_errors.append('a digit')
    if not re.search(r'[^A-Za-z0-9]', password):
        pw_errors.append('a special character')
    if pw_errors:
        return jsonify({'error': 'Password must contain ' + ', '.join(pw_errors)}), 400
    hashed_pw = generate_password_hash(password)
    user_doc = {'email': email, 'password': hashed_pw, 'role': role}
    if phone:
        user_doc['phone'] = phone
    if users_collection is not None:
        try:
            users_collection.insert_one(user_doc)
        except Exception as e:
            print("⚠️ Could not insert user:", e)
    token = jwt.encode({
        'email': email,
        'role': role,
        'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=12)
    }, os.getenv('JWT_SECRET', 'supersecretkey'), algorithm='HS256')
    return jsonify({'success': True, 'token': token, 'role': role})

@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json() or {}
    email = data.get('email')
    password = data.get('password')
    
    # Validate required fields
    if not email or not password:
        return jsonify({'error': 'Email and password are required'}), 400
    
    # Find user in database
    user = users_collection.find_one({'email': email}) if users_collection is not None else None
    if not user:
        return jsonify({'error': 'User not found. Please sign up first.'}), 404
    
    # Check password
    if not check_password_hash(user.get('password', ''), password):
        return jsonify({'error': 'Invalid password. Please try again.'}), 401
    
    # Generate token
    token = jwt.encode({
        'email': email,
        'role': user.get('role', 'seller'),
        'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=12)
    }, os.getenv('JWT_SECRET', 'supersecretkey'), algorithm='HS256')
    
    return jsonify({
        'success': True,
        'message': 'Login successful',
        'token': token,
        'role': user.get('role', 'seller'),
        'email': email
    })

@app.route('/api/forgot-password', methods=['POST'])
def forgot_password():
    """Send a password reset email with a reset token"""
    data = request.get_json() or {}
    email = data.get('email')
    
    if not email:
        return jsonify({'error': 'Email is required'}), 400
    
    # Find user in database
    user = users_collection.find_one({'email': email}) if users_collection is not None else None
    if not user:
        # For security, don't reveal if user exists
        return jsonify({'success': True, 'message': 'If an account with this email exists, a reset link has been sent.'}), 200
    
    # Generate reset token (valid for 1 hour)
    reset_token = jwt.encode({
        'email': email,
        'purpose': 'password_reset',
        'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=1)
    }, os.getenv('JWT_SECRET', 'supersecretkey'), algorithm='HS256')
    
    # Store reset token in database (optional but recommended)
    if users_collection is not None:
        try:
            users_collection.update_one(
                {'email': email},
                {'$set': {'reset_token': reset_token, 'reset_token_created': datetime.datetime.utcnow()}}
            )
        except Exception as e:
            print(f"⚠️ Could not store reset token: {e}")
    
    # In production, you would send an email here with:
    # reset_link = f"{FRONTEND_URL}/reset-password?token={reset_token}"
    # For now, we'll just return the token (in production, NEVER do this)
    print(f"✅ Password reset token generated for {email}: {reset_token}")
    
    return jsonify({
        'success': True,
        'message': 'If an account with this email exists, a reset link has been sent.',
        # Remove the token from production - this is only for testing
        'token': reset_token
    }), 200

@app.route('/api/reset-password', methods=['POST'])
def reset_password():
    """Reset password using a valid reset token"""
    data = request.get_json() or {}
    token = data.get('token')
    password = data.get('password')
    
    if not token or not password:
        return jsonify({'error': 'Token and password are required'}), 400
    
    # Validate token
    try:
        decoded = jwt.decode(token, os.getenv('JWT_SECRET', 'supersecretkey'), algorithms=['HS256'])
        if decoded.get('purpose') != 'password_reset':
            return jsonify({'error': 'Invalid token'}), 400
        email = decoded.get('email')
    except jwt.ExpiredSignatureError:
        return jsonify({'error': 'Reset link has expired. Please request a new one.'}), 400
    except jwt.InvalidTokenError:
        return jsonify({'error': 'Invalid or tampered reset link'}), 400
    
    # Validate password
    pw_errors = []
    if len(password) < 8:
        pw_errors.append('at least 8 characters')
    if not re.search(r'[A-Z]', password):
        pw_errors.append('an uppercase letter')
    if not re.search(r'[a-z]', password):
        pw_errors.append('a lowercase letter')
    if not re.search(r'[0-9]', password):
        pw_errors.append('a digit')
    if not re.search(r'[^A-Za-z0-9]', password):
        pw_errors.append('a special character')
    if pw_errors:
        return jsonify({'error': 'Password must contain ' + ', '.join(pw_errors)}), 400
    
    # Update user password
    hashed_pw = generate_password_hash(password)
    if users_collection is not None:
        try:
            result = users_collection.update_one(
                {'email': email},
                {'$set': {'password': hashed_pw, 'reset_token': None}}
            )
            if result.matched_count == 0:
                return jsonify({'error': 'User not found'}), 404
        except Exception as e:
            print(f"⚠️ Could not update password: {e}")
            return jsonify({'error': 'Failed to reset password'}), 500
    
    return jsonify({
        'success': True,
        'message': 'Password reset successful. You can now log in with your new password.'
    }), 200

@app.route('/api/send-otp', methods=['POST'])
def send_otp():
    """Generate and send OTP for password reset or signup verification"""
    import random
    data = request.get_json() or {}
    email = data.get('email')
    purpose = data.get('purpose', 'password_reset')  # 'password_reset' or 'signup'
    
    if not email:
        return jsonify({'error': 'Email is required'}), 400
    
    # Generate 6-digit OTP
    otp = str(random.randint(100000, 999999))
    otp_created = datetime.datetime.utcnow()
    
    # Store OTP in database (expires in 10 minutes)
    if users_collection is not None:
        try:
            users_collection.update_one(
                {'email': email},
                {'$set': {
                    'otp': otp,
                    'otp_created': otp_created,
                    'otp_purpose': purpose,
                    'otp_attempts': 0
                }},
                upsert=False  # Don't create if user doesn't exist
            )
        except Exception as e:
            print(f"⚠️ Could not store OTP: {e}")
            return jsonify({'error': 'Failed to generate OTP'}), 500
    
    # In production, send email with OTP
    # For now, return OTP for testing (REMOVE IN PRODUCTION)
    print(f"✅ OTP generated for {email}: {otp}")
    
    return jsonify({
        'success': True,
        'message': f'OTP sent to {email}. Valid for 10 minutes.',
        'otp': otp  # Remove in production
    }), 200

@app.route('/api/verify-otp', methods=['POST'])
def verify_otp():
    """Verify OTP and return temporary reset token"""
    data = request.get_json() or {}
    email = data.get('email')
    otp = data.get('otp')
    
    if not email or not otp:
        return jsonify({'error': 'Email and OTP are required'}), 400
    
    # Find user and check OTP
    user = users_collection.find_one({'email': email}) if users_collection is not None else None
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    stored_otp = user.get('otp')
    otp_created = user.get('otp_created')
    otp_attempts = user.get('otp_attempts', 0)
    
    # Check attempts
    if otp_attempts >= 5:
        return jsonify({'error': 'Too many failed attempts. Request a new OTP.'}), 429
    
    # Check if OTP expired (10 minutes)
    if not otp_created or (datetime.datetime.utcnow() - otp_created).total_seconds() > 600:
        return jsonify({'error': 'OTP has expired. Request a new one.'}), 400
    
    # Verify OTP
    if stored_otp != otp:
        # Increment failed attempts
        if users_collection is not None:
            users_collection.update_one(
                {'email': email},
                {'$inc': {'otp_attempts': 1}}
            )
        return jsonify({'error': 'Invalid OTP. Please try again.'}), 400
    
    # OTP verified! Generate temporary reset token (valid 15 minutes)
    reset_token = jwt.encode({
        'email': email,
        'purpose': 'otp_verified_reset',
        'exp': datetime.datetime.utcnow() + datetime.timedelta(minutes=15)
    }, os.getenv('JWT_SECRET', 'supersecretkey'), algorithm='HS256')
    
    # Clear OTP
    if users_collection is not None:
        users_collection.update_one(
            {'email': email},
            {'$set': {'otp': None, 'otp_attempts': 0}}
        )
    
    return jsonify({
        'success': True,
        'message': 'OTP verified successfully.',
        'reset_token': reset_token
    }), 200

@app.route('/api/reset-password-with-otp', methods=['POST'])
def reset_password_with_otp():
    """Reset password using OTP-verified token"""
    data = request.get_json() or {}
    reset_token = data.get('reset_token')
    password = data.get('password')
    
    if not reset_token or not password:
        return jsonify({'error': 'Reset token and password are required'}), 400
    
    # Validate token
    try:
        decoded = jwt.decode(reset_token, os.getenv('JWT_SECRET', 'supersecretkey'), algorithms=['HS256'])
        if decoded.get('purpose') != 'otp_verified_reset':
            return jsonify({'error': 'Invalid reset token'}), 400
        email = decoded.get('email')
    except jwt.ExpiredSignatureError:
        return jsonify({'error': 'Reset token has expired. Request a new OTP.'}), 400
    except jwt.InvalidTokenError:
        return jsonify({'error': 'Invalid reset token'}), 400
    
    # Validate password
    pw_errors = []
    if len(password) < 8:
        pw_errors.append('at least 8 characters')
    if not re.search(r'[A-Z]', password):
        pw_errors.append('an uppercase letter')
    if not re.search(r'[a-z]', password):
        pw_errors.append('a lowercase letter')
    if not re.search(r'[0-9]', password):
        pw_errors.append('a digit')
    if not re.search(r'[^A-Za-z0-9]', password):
        pw_errors.append('a special character')
    if pw_errors:
        return jsonify({'error': 'Password must contain ' + ', '.join(pw_errors)}), 400
    
    # Update password
    hashed_pw = generate_password_hash(password)
    if users_collection is not None:
        try:
            result = users_collection.update_one(
                {'email': email},
                {'$set': {'password': hashed_pw}}
            )
            if result.matched_count == 0:
                return jsonify({'error': 'User not found'}), 404
        except Exception as e:
            print(f"⚠️ Could not update password: {e}")
            return jsonify({'error': 'Failed to reset password'}), 500
    
    return jsonify({
        'success': True,
        'message': 'Password reset successful. You can now log in with your new password.'
    }), 200

@app.route('/api/register', methods=['POST'])
def api_register():
    try:
        data = request.json or {}
        if not data.get('name') or not data.get('brand'):
            return jsonify({'error': 'name and brand required'}), 400
        # ensure blockchain available
        if not w3.is_connected() or not contract or not private_key:
            return jsonify({'error': 'Blockchain not configured or connected'}), 500
        result = register_product_internal(data)
        return jsonify({'success': True, **result})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/nlp_check', methods=['POST'])
def nlp_check():
    if nlp_model is None or tfidf_vectorizer is None:
        return jsonify({'error': 'NLP model not loaded'}), 500
    description = (request.json or {}).get('description')
    cleaned = clean_text(description)
    try:
        X = tfidf_vectorizer.transform([cleaned])
        proba = nlp_model.predict_proba(X)[0]
        classes = list(nlp_model.classes_)
        # Find index for 'counterfeit' (or 'fake')
        if 'counterfeit' in [c.lower() for c in classes]:
            idx = [c.lower() for c in classes].index('counterfeit')
        elif 'fake' in [c.lower() for c in classes]:
            idx = [c.lower() for c in classes].index('fake')
        else:
            idx = 0  # fallback to first class
        score = float(proba[idx])
        if score >= 0.5:
            label = classes[idx]
            suspicious = True
        else:
            # pick the other class as authentic
            label = [c for c in classes if c != classes[idx]][0] if len(classes) > 1 else 'authentic'
            suspicious = False
        return jsonify({'label': label, 'score': round(score, 3), 'suspicious': suspicious})
    except Exception as e:
        print("⚠️ NLP check error:", e)
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/cv_check', methods=['POST'])
def cv_check():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    file = request.files['image']
    try:
        img = Image.open(file.stream)
        img_rgb = img.convert('RGB')
        img_resized = img_rgb.resize((CV_IMG_SIZE, CV_IMG_SIZE))
        arr = np.array(img_resized).astype('float32')
        try:
            from tensorflow.keras.applications.efficientnet import preprocess_input
            arr = preprocess_input(arr)
        except Exception:
            arr = arr / 255.0
        arr = np.expand_dims(arr, 0)

        # --- Feature extraction ---
        features = extract_image_features(img, file_storage=file)

        model_to_use = cv_model
        # try lazy-load if not loaded at startup
        if model_to_use is None:
            try:
                import tensorflow as _tf
                if os.path.exists(CV_MODEL_PATH):
                    model_to_use = _tf.keras.models.load_model(CV_MODEL_PATH)
                    print(f"✅ CV model lazy-loaded from {CV_MODEL_PATH}")
                else:
                    return jsonify({'error': 'CV model not found on server'}), 500
            except Exception as le:
                print('CV lazy-load error:', le)
                return jsonify({'error': 'Unable to load CV model: ' + str(le)}), 500

        preds = model_to_use.predict(arr)
        preds_a = np.array(preds).ravel()
        # interpret prediction: if single-output sigmoid -> probability is preds[0]
        # if two-output softmax -> take index 1 as 'authentic' probability
        if preds_a.size == 1:
            prob = float(preds_a[0])
        elif preds_a.size >= 2:
            prob = float(preds_a[1])
        else:
            prob = float(preds_a[0])

        authentic = prob >= CV_AUTH_THRESHOLD
        return jsonify({'authentic': bool(authentic), 'confidence': round(prob, 4), 'image_features': features})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/sbert_check', methods=['POST'])
def sbert_check():
    try:
        if sb_clf is None or sb_model is None:
            return jsonify({'error': 'SBERT classifier not available. Train first.'}), 500
        data = request.get_json() or {}
        text = data.get('text') or data.get('description')
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        # use preloaded embedder when possible to avoid repeated heavy loads
        try:
            if sb_embedder is not None:
                embedder = sb_embedder
            else:
                from sentence_transformers import SentenceTransformer
                embedder = SentenceTransformer(sb_model)
            emb = embedder.encode([text], convert_to_numpy=True)
        except Exception as ie:
            print('Embedder load/encode error:', ie)
            traceback.print_exc()
            return jsonify({'error': 'Embedder error: ' + str(ie)}), 500
        pred = sb_clf.predict(emb)[0]
        score = None
        try:
            proba = sb_clf.predict_proba(emb)[0]
            classes = list(sb_clf.classes_)
            idx = classes.index(pred)
            score = float(proba[idx])
        except Exception:
            score = None
        suspicious = str(pred).lower() in ['counterfeit', 'fake']
        return jsonify({'label': str(pred), 'score': round(score, 3) if score is not None else None, 'suspicious': suspicious})
    except Exception as e:
        print('SBERT CHECK ERROR:', str(e))
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/review_analyze', methods=['POST'])
def review_analyze():
    """Analyze a full product review by splitting into sentences and running SBERT classifier per sentence.
    POST JSON: { "review": "..." }
    Returns: { sentences: [{text,label,score,suspicious}], suspicious_fraction: float, overall_suspicious: bool }
    """
    try:
        if sb_clf is None or sb_model is None:
            return jsonify({'error': 'SBERT classifier not available. Train first.'}), 500
        data = request.get_json() or {}
        review = data.get('review') or data.get('text') or data.get('description')
        if not review:
            return jsonify({'error': 'No review text provided'}), 400
        try:
            res = analyze_review_text(review)
            return jsonify(res)
        except Exception as e:
            traceback.print_exc()
            return jsonify({'error': str(e)}), 500
    except Exception as e:
        print('REVIEW ANALYZE ERROR:', str(e))
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/model_health', methods=['GET'])
def model_health():
    """Return quick health/status for NLP components and a tiny sample test prediction.
    Useful to verify SBERT embedder & classifier are loaded and working.
    """
    status = {
        'nlp_model_loaded': bool(nlp_model),
        'tfidf_loaded': bool(tfidf_vectorizer),
        'sbert_clf_loaded': bool(sb_clf),
        'sbert_model_name': sb_model or None,
        'sbert_embedder_loaded': sb_embedder is not None
    }

    sample = "This looks like an authentic product from the brand"
    sample_result = None

    # Try quick SBERT encode + predict if possible
    try:
        if sb_clf is not None and sb_model is not None:
            # prefer preloaded embedder
            try:
                if sb_embedder is not None:
                    embedder = sb_embedder
                else:
                    from sentence_transformers import SentenceTransformer
                    embedder = SentenceTransformer(sb_model)
                emb = embedder.encode([sample], convert_to_numpy=True)
                pred = sb_clf.predict(emb)[0]
                score = None
                try:
                    probs = sb_clf.predict_proba(emb)[0]
                    classes = list(sb_clf.classes_)
                    score = float(probs[classes.index(pred)])
                except Exception:
                    score = None
                sample_result = {'label': str(pred), 'score': round(score, 3) if score is not None else None}
                status['sample_predict_ok'] = True
            except Exception as ie:
                status['sample_predict_ok'] = False
                status['sample_error'] = str(ie)
        else:
            status['sample_predict_ok'] = False
            status['sample_error'] = 'SBERT classifier or model name missing'
    except Exception as e:
        status['sample_predict_ok'] = False
        status['sample_error'] = str(e)

    return jsonify({'status': status, 'sample': sample, 'sample_result': sample_result})


# --- Archive integration helpers ---
def _ensure_archive_dirs():
    base = os.path.join(os.path.dirname(__file__), 'dataset', 'archive')
    images_dir = os.path.join(base, 'images')
    meta_dir = os.path.join(base, 'meta')
    try:
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(meta_dir, exist_ok=True)
    except Exception:
        pass
    return images_dir, meta_dir

def save_to_archive(pil_image, metadata: dict):
    """Save a PIL image and metadata into dataset/archive and optionally insert into DB.
    Returns dict with paths and archive_id on success, otherwise None.
    """
    try:
        images_dir, meta_dir = _ensure_archive_dirs()
        archive_id = str(uuid.uuid4())
        img_name = f"{archive_id}.jpg"
        img_path = os.path.join(images_dir, img_name)
        try:
            pil_image.save(img_path, format='JPEG', quality=90)
        except Exception:
            # fallback: convert to RGB then save
            try:
                pil_image.convert('RGB').save(img_path, format='JPEG', quality=90)
            except Exception as e:
                print('Failed to save archive image:', e)
                return None

        meta = dict(metadata or {})
        meta['archive_image'] = os.path.relpath(img_path, start=os.path.dirname(__file__))
        meta['saved_at'] = datetime.datetime.utcnow().isoformat()
        meta['archive_id'] = archive_id

        meta_path = os.path.join(meta_dir, archive_id + '.json')
        try:
            with open(meta_path, 'w', encoding='utf-8') as mf:
                json.dump(meta, mf, ensure_ascii=False, indent=2)
        except Exception as e:
            print('Failed to write archive metadata:', e)

        # If a legacy/frontend archive CSV exists, append a summary row there for integration
        try:
            frontend_archive_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'brand-auth-app', 'archive (1)'))
            if os.path.exists(frontend_archive_dir) and os.path.isdir(frontend_archive_dir):
                csv_path = os.path.join(frontend_archive_dir, 'archived_added.csv')
                header = ['archive_id', 'product_id', 'product_url', 'description', 'cv_conf', 'nlp_score', 'saved_at', 'image_path']
                row = [archive_id, meta.get('product_id') or meta.get('id') or '', meta.get('product_url') or '', meta.get('description') or '', meta.get('cv_conf') if meta.get('cv_conf') is not None else '', meta.get('nlp_score') if meta.get('nlp_score') is not None else '', meta.get('saved_at') or '', meta.get('archive_image') or meta.get('archive_image')]
                try:
                    write_header = not os.path.exists(csv_path)
                    with open(csv_path, 'a', newline='', encoding='utf-8') as cf:
                        writer = csv.writer(cf)
                        if write_header:
                            writer.writerow(header)
                        writer.writerow(row)
                except Exception as e:
                    print('Failed to append to frontend archive CSV:', e)
        except Exception:
            pass

        # optional DB insert
        try:
            if 'db' in globals() and db is not None:
                coll = None
                try:
                    coll = db.get_collection('archive') if hasattr(db, 'get_collection') else db['archive']
                except Exception:
                    coll = None
                if coll is not None:
                    coll.insert_one({'_id': archive_id, 'image_path': img_path, 'metadata': meta, 'created_at': meta['saved_at']})
        except Exception:
            pass

        return {'archive_id': archive_id, 'image_path': img_path, 'meta_path': meta_path}
    except Exception as e:
        print('save_to_archive error:', e)
        traceback.print_exc()
        return None


@app.route('/api/multimodal_check', methods=['POST'])
@role_required('merchant', 'seller', 'influencer', 'admin')
def multimodal_check():
    """Accepts an image file (form field 'image') OR image URL (form field 'image_url') and/or text (form field 'description' or JSON body).
    Returns detailed CV, NLP and fused scores with explanations.
    Requires valid JWT token with appropriate role.
    """
    import urllib.request
    result = {'cv': None, 'nlp': None, 'fused': None}

    # Basic request logging for debugging frontend connectivity
    try:
        remote = request.remote_addr
        form_keys = list(request.form.keys()) if request.form else []
        file_keys = list(request.files.keys()) if request.files else []
        print(f"[DEBUG] /api/multimodal_check called from {remote} form_keys={form_keys} file_keys={file_keys}")
    except Exception as _:
        pass

    # --- CV part ---
    try:
        cv_prob = None
        img = None
        # Try to get image from file upload
        if 'image' in request.files:
            file = request.files['image']
            img = Image.open(file.stream).convert('RGB')
        # Or try to get image from URL (form or JSON)
        elif request.form.get('image_url'):
            image_url = request.form.get('image_url')
            try:
                with urllib.request.urlopen(image_url, timeout=10) as response:
                    img = Image.open(response).convert('RGB')
            except Exception as url_err:
                result['cv'] = {'error': f'Failed to load image from URL: {str(url_err)}'}
        elif request.is_json and (request.get_json() or {}).get('image_url'):
            image_url = (request.get_json() or {}).get('image_url')
            try:
                with urllib.request.urlopen(image_url, timeout=10) as response:
                    img = Image.open(response).convert('RGB')
            except Exception as url_err:
                result['cv'] = {'error': f'Failed to load image from URL: {str(url_err)}'}

        if img is not None:
            # compute image qualities for display
            features = extract_image_features(img)
            cv_prob = predict_cv_from_pil(img)
            if cv_prob is None:
                result['cv'] = {'error': 'CV model unavailable or prediction failed', 'image_features': features}
            else:
                result['cv'] = {'authentic_prob': round(cv_prob, 4), 'image_features': features}
    except Exception as e:
        traceback.print_exc()
        result['cv'] = {'error': str(e)}

    # --- NLP part ---
    try:
        text = None
        if request.is_json:
            text = (request.get_json() or {}).get('description') or (request.get_json() or {}).get('text')
        if not text:
            # also check form fields
            text = request.form.get('description') or request.form.get('text')

        if text:
            nlp_prob, sent_scores = predict_nlp_scores(text)
            if nlp_prob is None:
                result['nlp'] = {'error': 'NLP model unavailable or prediction failed'}
            else:
                result['nlp'] = {'authentic_prob': round(nlp_prob, 4), 'sentences': sent_scores}
        else:
            result['nlp'] = {'error': 'No text provided'}
    except Exception as e:
        traceback.print_exc()
        result['nlp'] = {'error': str(e)}

    # --- Product metadata (optional) ---
    try:
        product_id = None
        product_url = None
        # try JSON body first
        if request.is_json:
            j = request.get_json() or {}
            product_id = j.get('product_id') or j.get('id')
            product_url = j.get('product_url') or j.get('url')
        # fall back to form fields
        if not product_id:
            product_id = request.form.get('product_id') or request.form.get('id')
        if not product_url:
            product_url = request.form.get('product_url') or request.form.get('url')
        if product_id or product_url:
            meta = {}
            if product_id:
                meta['product_id'] = product_id
            if product_url:
                meta['product_url'] = product_url
            result['meta'] = meta
            # persist a minimal audit record if DB is available
            try:
                if 'db' in globals() and db is not None:
                    coll = None
                    try:
                        coll = db.get_collection('audit_logs') if hasattr(db, 'get_collection') else db['audit_logs']
                    except Exception:
                        coll = None
                    if coll is not None:
                        coll.insert_one({'_id': str(uuid.uuid4()), 'event': 'multimodal_check', 'meta': meta, 'created_at': datetime.datetime.utcnow().isoformat()})
            except Exception:
                pass
    except Exception:
        pass

    # --- Fuse ---
    try:
        cv_present = result.get('cv') and result['cv'].get('authentic_prob') is not None
        nlp_present = result.get('nlp') and result['nlp'].get('authentic_prob') is not None
        if cv_present and nlp_present:
            cv_p = float(result['cv']['authentic_prob'])
            nlp_p = float(result['nlp']['authentic_prob'])
            fused = fuse_scores(cv_p, nlp_p)
            result['fused'] = {'authentic_prob': round(fused, 4), 'authentic': fused >= 0.5}
        elif cv_present:
            p = float(result['cv']['authentic_prob'])
            result['fused'] = {'authentic_prob': round(p, 4), 'authentic': p >= 0.5, 'note': 'Only CV available'}
        elif nlp_present:
            p = float(result['nlp']['authentic_prob'])
            result['fused'] = {'authentic_prob': round(p, 4), 'authentic': p >= 0.5, 'note': 'Only NLP available'}
        else:
            result['fused'] = {'error': 'No usable modalities available'}
    except Exception as e:
        traceback.print_exc()
        result['fused'] = {'error': str(e)}

    # --- Optional: persist authentic verifications to archive dataset ---
    try:
        do_archive = False
        fused_block = result.get('fused') or {}
        try:
            if isinstance(fused_block, dict):
                if fused_block.get('authentic') is True:
                    do_archive = True
                else:
                    try:
                        if float(fused_block.get('authentic_prob') or 0) >= float(os.getenv('ARCHIVE_MIN_PROB', '0.5')):
                            do_archive = True
                    except Exception:
                        pass
        except Exception:
            pass

        # Allow caller override with `archive` flag in JSON or form
        try:
            arc_flag = None
            if request.is_json:
                arc_flag = (request.get_json() or {}).get('archive')
            if arc_flag is None:
                arc_flag = request.form.get('archive')
            if isinstance(arc_flag, str) and arc_flag.lower() in ('1', 'true', 'yes'):
                do_archive = True
            if isinstance(arc_flag, (int, float)) and int(arc_flag) == 1:
                do_archive = True
        except Exception:
            pass

        if do_archive and 'img' in locals() and img is not None:
            meta_for_archive = {'product_id': product_id if 'product_id' in locals() else None, 'product_url': product_url if 'product_url' in locals() else None, 'description': text if 'text' in locals() else None, 'cv': result.get('cv'), 'nlp': result.get('nlp'), 'fused': result.get('fused')}
            saved = save_to_archive(img, meta_for_archive)
            if saved:
                result['archive'] = saved
    except Exception as e:
        print('Archive persistence error:', e)
        traceback.print_exc()

    return jsonify(result)

@app.route('/api/seller-activity', methods=['GET'])
def seller_activity():
    try:
        if not products_collection:
            return jsonify({'count': 0, 'recent': ['DB not connected or not available']})
        count = products_collection.count_documents({})
        recent_products = products_collection.find({}, {'name': 1, 'brand': 1, 'registration_date': 1}).sort('registration_date', -1).limit(5)
        recent = [f"Registered product '{p.get('name','Unknown')}' ({p.get('registration_date','')})" for p in recent_products]
        print(f"[DEBUG] Seller Activity - Count: {count}, Recent: {recent}")
        return jsonify({'count': count, 'recent': recent})
    except Exception as e:
        print(f"[ERROR] Seller Activity: {str(e)}")
        return jsonify({'count': 0, 'recent': [f'Error: {str(e)}']})

# Seller Products endpoint for Merchant Studio
@app.route('/api/seller-products', methods=['GET'])
def get_seller_products():
    """Returns products registered by the current seller"""
    try:
        auth_header = request.headers.get('Authorization', '')
        print(f"[seller-products] Auth header present: {bool(auth_header)}")
        
        if not auth_header or not auth_header.startswith('Bearer '):
            print("[seller-products] No valid Bearer token, returning mock data")
            return jsonify({'products': [], 'total_verifications': 0, 'total_revenue': 0})
        
        # Safely extract token
        parts = auth_header.split(' ')
        if len(parts) != 2:
            print("[seller-products] Invalid auth header format")
            return jsonify({'products': [], 'total_verifications': 0, 'total_revenue': 0})
        
        token = parts[1]
        try:
            payload = jwt.decode(token, os.getenv('JWT_SECRET', 'supersecretkey'), algorithms=['HS256'])
            seller_id = payload.get('user_id')
            print(f"[seller-products] Decoded token, seller_id: {seller_id}")
            if not seller_id:
                print("[seller-products] No user_id in token")
                return jsonify({'products': [], 'total_verifications': 0, 'total_revenue': 0})
        except jwt.InvalidTokenError as e:
            print(f"[seller-products] JWT error: {str(e)}")
            return jsonify({'products': [], 'total_verifications': 0, 'total_revenue': 0})
        
        if products_collection is None:
            return jsonify({'products': [], 'total_verifications': 0, 'total_revenue': 0})
        
        # Find products by seller_id
        products = list(products_collection.find({'seller_id': seller_id}, {'_id': 0}))
        
        # Calculate totals
        total_verifications = sum(p.get('verification_count', 0) for p in products)
        total_revenue = sum(p.get('revenue', 0) for p in products)
        
        print(f"[seller-products] Returning {len(products)} products for seller {seller_id}")
        return jsonify({
            'products': products,
            'total_verifications': total_verifications,
            'total_revenue': total_revenue
        })
    except Exception as e:
        print(f"[ERROR] Seller Products: {str(e)}")
        return jsonify({'error': str(e), 'products': []}), 500

# Seller Insights endpoint for Merchant Insights
@app.route('/api/seller-insights', methods=['GET'])
def get_seller_insights():
    """Returns analytics insights for the seller's products"""
    try:
        auth_header = request.headers.get('Authorization', '')
        print(f"[seller-insights] Auth header present: {bool(auth_header)}")
        
        if not auth_header or not auth_header.startswith('Bearer '):
            print("[seller-insights] No valid Bearer token, returning mock data")
            return jsonify({
                'totalVerifications': 0,
                'positiveRate': 0,
                'topProducts': [],
                'activityTrend': [],
                'sellerRiskScore': 0
            })
        
        # Safely extract token
        parts = auth_header.split(' ')
        if len(parts) != 2:
            print("[seller-insights] Invalid auth header format")
            return jsonify({
                'totalVerifications': 0,
                'positiveRate': 0,
                'topProducts': [],
                'activityTrend': [],
                'sellerRiskScore': 0
            })
        
        token = parts[1]
        try:
            payload = jwt.decode(token, os.getenv('JWT_SECRET', 'supersecretkey'), algorithms=['HS256'])
            seller_id = payload.get('user_id')
            print(f"[seller-insights] Decoded token, seller_id: {seller_id}")
            if not seller_id:
                print("[seller-insights] No user_id in token")
                return jsonify({
                    'totalVerifications': 0,
                    'positiveRate': 0,
                    'topProducts': [],
                    'activityTrend': [],
                    'sellerRiskScore': 0
                })
        except jwt.InvalidTokenError as e:
            print(f"[seller-insights] JWT error: {str(e)}")
            return jsonify({
                'totalVerifications': 0,
                'positiveRate': 0,
                'topProducts': [],
                'activityTrend': [],
                'sellerRiskScore': 0
            })
        
        if products_collection is None:
            return jsonify({
                'totalVerifications': 0,
                'positiveRate': 0,
                'topProducts': [],
                'activityTrend': [],
                'sellerRiskScore': 0
            })
        
        # Get seller's products
        products = list(products_collection.find({'seller_id': seller_id}, {'_id': 0}))
        
        total_verifications = sum(p.get('verification_count', 0) for p in products)
        positive_count = sum(p.get('positive_verifications', 0) for p in products)
        positive_rate = (positive_count / total_verifications * 100) if total_verifications > 0 else 0
        
        # Top products by verification count
        top_products = sorted(products, key=lambda x: x.get('verification_count', 0), reverse=True)[:3]
        top_products_data = [
            {
                'name': p.get('name', 'Unknown'),
                'verifications': p.get('verification_count', 0),
                'positiveRate': (p.get('positive_verifications', 0) / p.get('verification_count', 1) * 100) if p.get('verification_count', 0) > 0 else 0
            }
            for p in top_products
        ]
        
        # Mock activity trend (can be enhanced with time-series data)
        activity_trend = [
            {'day': 'Mon', 'verifications': 180},
            {'day': 'Tue', 'verifications': 215},
            {'day': 'Wed', 'verifications': 195},
            {'day': 'Thu', 'verifications': 245},
            {'day': 'Fri', 'verifications': 220},
            {'day': 'Sat', 'verifications': 189},
            {'day': 'Sun', 'verifications': 210}
        ]
        
        # Risk score (0-100, lower is better)
        seller_risk_score = max(0, min(100, 50 - int(positive_rate / 2)))
        
        print(f"[seller-insights] Returning insights for seller {seller_id}")
        return jsonify({
            'totalVerifications': total_verifications,
            'positiveRate': round(positive_rate, 1),
            'topProducts': top_products_data,
            'activityTrend': activity_trend,
            'sellerRiskScore': seller_risk_score
        })
    except Exception as e:
        print(f"[ERROR] Seller Insights: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Influencer endpoints: events recording, stats, referrals, top, CSV report, and simple settings
@app.route('/api/influencer-event', methods=['POST'])
@rate_limit(max_requests=1000, window_seconds=60)
def influencer_event():
    """Record an influencer event with validation."""
    try:
        data = request.get_json(force=True) or {}
        influencer = data.get('influencer')
        product_id = data.get('product_id')
        event_type = data.get('event_type')
        revenue = data.get('revenue', 0)
        
        # Validation
        if not influencer or not isinstance(influencer, str) or len(influencer.strip()) == 0:
            return jsonify({'error': 'influencer (non-empty string) required'}), 400
        if not product_id or not isinstance(product_id, str) or len(product_id.strip()) == 0:
            return jsonify({'error': 'product_id (non-empty string) required'}), 400
        if not event_type or event_type not in ['impression', 'click', 'verification', 'purchase']:
            return jsonify({'error': 'event_type must be: impression, click, verification, or purchase'}), 400
        try:
            revenue = float(revenue or 0)
            if revenue < 0:
                return jsonify({'error': 'revenue must be >= 0'}), 400
        except (ValueError, TypeError):
            return jsonify({'error': 'revenue must be a number'}), 400
        
        # Persist event
        if influencer_events_collection is not None:
            influencer_events_collection.insert_one({
                'influencer': influencer.strip(),
                'product_id': product_id.strip(),
                'event_type': event_type,
                'revenue': revenue,
                'metadata': data.get('metadata', {}),
                'timestamp': datetime.datetime.utcnow().isoformat()
            })
        return jsonify({'status': 'ok', 'message': 'Event recorded'}), 201
    except Exception as e:
        print('[ERROR] influencer-event:', e)
        return jsonify({'error': str(e)}), 500


@app.route('/api/influencer-stats', methods=['GET'])
def influencer_stats():
    try:
        # permission check: only influencers or admins can view detailed stats
        payload = _decode_jwt_payload_from_header()
        if payload is None or payload.get('role') not in ('influencer', 'admin'):
            return jsonify({'error': 'Forbidden'}), 403
        # optional filters
        start = request.args.get('start')
        end = request.args.get('end')
        influencer = request.args.get('influencer')

        if influencer_events_collection is None:
            # fallback demo
            return jsonify({
                'stats': {'impressions':12450,'clicks':2130,'verifications':843,'purchases':312,'conversionRate':14.6,'estRevenue':46230},
                'top': [{'name':'alice_influencer','conversions':120,'revenue':18000,'handle':'@alice'}],
                'timeline': []
            })

        query = {}
        if influencer:
            query['influencer'] = influencer
        if start or end:
            time_query = {}
            if start:
                try:
                    time_query['$gte'] = start
                except Exception:
                    pass
            if end:
                try:
                    time_query['$lte'] = end
                except Exception:
                    pass
            if time_query:
                query['timestamp'] = time_query

        events = list(influencer_events_collection.find(query))
        impressions = sum(1 for e in events if e.get('event_type') == 'impression')
        clicks = sum(1 for e in events if e.get('event_type') == 'click')
        verifications = sum(1 for e in events if e.get('event_type') == 'verification')
        purchases = sum(1 for e in events if e.get('event_type') == 'purchase')
        revenue = sum(float(e.get('revenue', 0) or 0) for e in events)
        conversionRate = (purchases / clicks * 100) if clicks > 0 else 0

        # top influencers by purchases
        stats_by_influencer = {}
        for e in events:
            name = e.get('influencer') or 'unknown'
            rec = stats_by_influencer.setdefault(name, {'conversions':0,'revenue':0})
            if e.get('event_type') == 'purchase':
                rec['conversions'] += 1
                rec['revenue'] += float(e.get('revenue', 0) or 0)

        top = sorted([{'name':k,'conversions':v['conversions'],'revenue':v['revenue']} for k,v in stats_by_influencer.items()], key=lambda x: x['conversions'], reverse=True)[:10]

        timeline = sorted(events, key=lambda x: x.get('timestamp', ''), reverse=True)[:50]

        # Sanitize Mongo documents to make them JSON serializable
        def sanitize_doc(d):
            out = {}
            for k, v in d.items():
                if k == '_id':
                    try:
                        out[k] = str(v)
                    except Exception:
                        out[k] = v
                elif isinstance(v, (datetime.datetime,)):
                    out[k] = v.isoformat()
                else:
                    # leave primitives as-is; if there are nested ObjectIds they will become strings via str()
                    try:
                        # attempt to convert ObjectId-like objects
                        if hasattr(v, '__class__') and v.__class__.__name__ == 'ObjectId':
                            out[k] = str(v)
                        else:
                            out[k] = v
                    except Exception:
                        out[k] = v
            return out

        safe_timeline = [sanitize_doc(e) for e in timeline]

        return jsonify({
            'stats': {'impressions':impressions,'clicks':clicks,'verifications':verifications,'purchases':purchases,'conversionRate':round(conversionRate,1),'estRevenue':round(revenue,2)},
            'top': top,
            'timeline': safe_timeline
        })
    except Exception as e:
        print('[ERROR] influencer-stats:', e)
        return jsonify({'error': str(e)}), 500


@app.route('/api/influencer-referrals', methods=['GET'])
def influencer_referrals():
    try:
        # permission: only influencer or admin can fetch raw referral events
        payload = _decode_jwt_payload_from_header()
        if payload is None or payload.get('role') not in ('influencer','admin'):
            return jsonify({'error': 'Forbidden'}), 403
        # pagination
        limit = int(request.args.get('limit', 50))
        page = int(request.args.get('page', 0))
        skip = page * limit
        if influencer_events_collection is None:
            return jsonify({'events': []})
        cursor = influencer_events_collection.find({}).sort('timestamp', -1).skip(skip).limit(limit)
        events = list(cursor)
        # sanitize events
        def sanitize(d):
            if isinstance(d, dict):
                d = dict(d)
                if '_id' in d:
                    try: d['_id'] = str(d['_id'])
                    except: pass
                for k, v in d.items():
                    if isinstance(v, datetime.datetime):
                        d[k] = v.isoformat()
                return d
            return d
        safe_events = [sanitize(e) for e in events]
        return jsonify({'events': safe_events})
    except Exception as e:
        print('[ERROR] influencer-referrals:', e)
        return jsonify({'error': str(e)}), 500


@app.route('/api/influencer-top', methods=['GET'])
def influencer_top():
    try:
        payload = _decode_jwt_payload_from_header()
        if payload is None or payload.get('role') not in ('influencer', 'admin'):
            return jsonify({'error': 'Forbidden'}), 403
        if influencer_events_collection is None:
            return jsonify({'top': []})
        # aggregate purchases by influencer
        pipeline = [
            {'$match': {'event_type': 'purchase'}},
            {'$group': {'_id': '$influencer', 'conversions': {'$sum': 1}, 'revenue': {'$sum': '$revenue'}}},
            {'$sort': {'conversions': -1}},
            {'$limit': 20}
        ]
        res = list(influencer_events_collection.aggregate(pipeline))
        top = [{'name': r['_id'], 'conversions': r.get('conversions',0), 'revenue': r.get('revenue',0)} for r in res]
        return jsonify({'top': top})
    except Exception as e:
        print('[ERROR] influencer-top:', e)
        return jsonify({'error': str(e)}), 500


@app.route('/api/influencer-report', methods=['GET'])
def influencer_report():
    try:
        payload = _decode_jwt_payload_from_header()
        if payload is None or payload.get('role') not in ('influencer', 'admin'):
            return jsonify({'error': 'Forbidden'}), 403
        # type=top|events
        rtype = request.args.get('type', 'top')
        import csv, io
        if influencer_events_collection is None:
            return jsonify({'error': 'mongo-unavailable'}), 503
        if rtype == 'events':
            cursor = influencer_events_collection.find({}).sort('timestamp', -1).limit(1000)
            rows = [['timestamp','influencer','event_type','product_id','revenue']]
            for e in cursor:
                rows.append([e.get('timestamp'), e.get('influencer'), e.get('event_type'), e.get('product_id'), e.get('revenue',0)])
        else:
            pipeline = [
                {'$match': {'event_type':'purchase'}},
                {'$group': {'_id':'$influencer','conversions':{'$sum':1},'revenue':{'$sum':'$revenue'}}},
                {'$sort': {'conversions': -1}},
                {'$limit': 100}
            ]
            res = list(influencer_events_collection.aggregate(pipeline))
            rows = [['influencer','conversions','revenue']]
            for r in res:
                rows.append([r.get('_id'), r.get('conversions',0), r.get('revenue',0)])

        si = io.StringIO()
        writer = csv.writer(si)
        for row in rows:
            writer.writerow(row)
        output = si.getvalue()
        return (output, 200, {'Content-Type': 'text/csv', 'Content-Disposition':'attachment; filename="influencer_report.csv"'})
    except Exception as e:
        print('[ERROR] influencer-report:', e)
        return jsonify({'error': str(e)}), 500


@app.route('/api/influencer-settings', methods=['GET','POST'])
def influencer_settings():
    try:
        payload = _decode_jwt_payload_from_header()
        # Only influencers or admins can list/set settings
        if payload is None or payload.get('role') not in ('influencer','admin'):
            return jsonify({'error':'Forbidden'}), 403
        if influencer_settings_collection is None:
            return jsonify({'settings': []})
        if request.method == 'GET':
                settings = list(influencer_settings_collection.find({}))
                # sanitize settings
                safe_settings = []
                for s in settings:
                    try:
                        s = dict(s)
                        if '_id' in s:
                            s['_id'] = str(s['_id'])
                        safe_settings.append(s)
                    except Exception:
                        safe_settings.append(s)
                return jsonify({'settings': safe_settings})
        else:
            data = request.get_json(force=True)
            if not data or 'influencer' not in data:
                return jsonify({'error':'influencer required'}), 400
            influencer = data['influencer']
            record = {'influencer': influencer, 'commission_pct': float(data.get('commission_pct', 0)), 'updated_at': datetime.datetime.utcnow().isoformat()}
            influencer_settings_collection.update_one({'influencer':influencer}, {'$set': record}, upsert=True)
            return jsonify({'status':'ok'})
    except Exception as e:
        print('[ERROR] influencer-settings:', e)
        return jsonify({'error': str(e)}), 500

# Delete product endpoint for Merchant Studio
@app.route('/api/delete-product/<product_id>', methods=['DELETE'])
def delete_seller_product(product_id):
    """Allows a seller to delete their own product"""
    try:
        auth_header = request.headers.get('Authorization', '')
        print(f"[delete-product] Auth header present: {bool(auth_header)}")
        
        if not auth_header or not auth_header.startswith('Bearer '):
            print("[delete-product] No valid Bearer token")
            return jsonify({'error': 'Unauthorized'}), 401
        
        token = auth_header.split(' ')[1]
        try:
            payload = jwt.decode(token, os.getenv('JWT_SECRET', 'supersecretkey'), algorithms=['HS256'])
            seller_id = payload.get('user_id')
            print(f"[delete-product] Decoded token, seller_id: {seller_id}, product_id: {product_id}")
        except jwt.InvalidTokenError as e:
            print(f"[delete-product] JWT error: {str(e)}")
            return jsonify({'error': 'Invalid token'}), 401
        
        if products_collection is None:
            return jsonify({'error': 'Database not available'}), 500
        
        # Verify the product belongs to this seller before deleting
        from bson import ObjectId
        try:
            product_oid = ObjectId(product_id)
        except:
            product_oid = product_id  # Use as-is if not a valid ObjectId
        
        product = products_collection.find_one({'_id': product_oid, 'seller_id': seller_id})
        if not product:
            # Also try without ObjectId conversion (string ID)
            product = products_collection.find_one({'id': product_id, 'seller_id': seller_id})
        
        if not product:
            print(f"[delete-product] Product not found or unauthorized for seller {seller_id}")
            return jsonify({'error': 'Product not found or unauthorized'}), 404
        
        # Delete the product
        result = products_collection.delete_one({'_id': product.get('_id') or product_id})
        
        if result.deleted_count > 0:
            print(f"[delete-product] Product {product_id} deleted successfully")
            return jsonify({'success': True, 'message': 'Product deleted successfully'}), 200
        else:
            return jsonify({'error': 'Failed to delete product'}), 500
            
    except Exception as e:
        print(f"[ERROR] Delete Product: {str(e)}")
        return jsonify({'error': str(e)}), 500
        return jsonify({'error': str(e)}), 500


# ---------------------- Influencer Role Endpoints ----------------------
@app.route('/api/become-influencer', methods=['POST'])
def become_influencer():
    """Self-service: logged-in user can request/promote themselves to influencer.
    Server validates token, updates users_collection role, logs to role_changes, and returns a new JWT.
    """
    try:
        payload = _decode_jwt_payload_from_header()
        if payload is None:
            return jsonify({'error': 'Unauthorized'}), 401
        email = payload.get('email')
        if not email:
            return jsonify({'error': 'No email in token'}), 400

        # Read current role
        old_role = None
        if users_collection is not None:
            user = users_collection.find_one({'email': email})
            if user:
                old_role = user.get('role')
                users_collection.update_one({'email': email}, {'$set': {'role': 'influencer'}})
            else:
                # If user not found, create a minimal user record as influencer
                users_collection.insert_one({'email': email, 'role': 'influencer'})

        # Audit log
        try:
            if role_changes_collection is not None:
                role_changes_collection.insert_one({
                    'user_email': email,
                    'old_role': old_role,
                    'new_role': 'influencer',
                    'changed_by': email,
                    'timestamp': datetime.datetime.utcnow().isoformat()
                })
        except Exception as _:
            pass

        # Re-issue token with updated role
        new_token = jwt.encode({
            'email': email,
            'role': 'influencer',
            'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=12)
        }, os.getenv('JWT_SECRET', 'supersecretkey'), algorithm='HS256')

        return jsonify({'token': new_token, 'role': 'influencer'})
    except Exception as e:
        print('[ERROR] become-influencer:', e)
        return jsonify({'error': str(e)}), 500


@app.route('/api/admin/create-influencer', methods=['POST'])
def admin_create_influencer():
    """Admin API to assign influencer role to an existing user (or create)"""
    try:
        payload = _decode_jwt_payload_from_header()
        if payload is None or payload.get('role') != 'admin':
            return jsonify({'error': 'Forbidden'}), 403
        admin_email = payload.get('email')
        data = request.get_json(force=True) or {}
        target_email = data.get('email')
        if not target_email:
            return jsonify({'error': 'email required'}), 400

        old_role = None
        if users_collection is not None:
            user = users_collection.find_one({'email': target_email})
            if user:
                old_role = user.get('role')
                users_collection.update_one({'email': target_email}, {'$set': {'role': 'influencer'}})
            else:
                users_collection.insert_one({'email': target_email, 'role': 'influencer'})

        # Audit log
        try:
            if role_changes_collection is not None:
                role_changes_collection.insert_one({
                    'user_email': target_email,
                    'old_role': old_role,
                    'new_role': 'influencer',
                    'changed_by': admin_email,
                    'timestamp': datetime.datetime.utcnow().isoformat()
                })
        except Exception:
            pass

        return jsonify({'success': True, 'email': target_email, 'role': 'influencer'})
    except Exception as e:
        print('[ERROR] admin_create_influencer:', e)
        return jsonify({'error': str(e)}), 500


# ---------------------- Influencer Profile & Onboarding ----------------------
@app.route('/api/influencer-profile', methods=['GET', 'POST'])
@rate_limit(max_requests=100, window_seconds=60)
def influencer_profile():
    """Get or update influencer profile (handle, social links, payout details)."""
    try:
        payload = _decode_jwt_payload_from_header()
        if payload is None or payload.get('role') not in ('influencer', 'admin'):
            return jsonify({'error': 'Unauthorized'}), 401
        
        influencer_email = payload.get('email')
        
        if request.method == 'GET':
            if influencer_settings_collection is None:
                return jsonify({'profile': {}})
            profile = influencer_settings_collection.find_one({'user_email': influencer_email})
            return jsonify({'profile': profile or {}})
        else:  # POST
            data = request.get_json(force=True) or {}
            handle = data.get('handle', '').strip()
            social_links = data.get('social_links', {})
            payout_email = data.get('payout_email', '').strip()
            bio = data.get('bio', '').strip()
            
            # Validation
            if not handle or len(handle) < 3:
                return jsonify({'error': 'handle must be at least 3 chars'}), 400
            if payout_email and '@' not in payout_email:
                return jsonify({'error': 'invalid payout_email'}), 400
            
            profile = {
                'user_email': influencer_email,
                'handle': handle,
                'social_links': social_links,
                'payout_email': payout_email,
                'bio': bio,
                'updated_at': datetime.datetime.utcnow().isoformat()
            }
            
            if influencer_settings_collection is not None:
                influencer_settings_collection.update_one(
                    {'user_email': influencer_email},
                    {'$set': profile},
                    upsert=True
                )
            
            return jsonify({'profile': profile}), 200
    except Exception as e:
        print('[ERROR] influencer-profile:', e)
        return jsonify({'error': str(e)}), 500


# ---------------------- Payout Simulation ----------------------
@app.route('/api/influencer-payout', methods=['POST'])
@rate_limit(max_requests=50, window_seconds=60)
def influencer_payout():
    """Simulate payout processing for an influencer."""
    try:
        payload = _decode_jwt_payload_from_header()
        if payload is None or payload.get('role') not in ('influencer', 'admin'):
            return jsonify({'error': 'Unauthorized'}), 401
        
        data = request.get_json(force=True) or {}
        influencer_handle = data.get('influencer')
        amount = data.get('amount', 0)
        
        if not influencer_handle:
            return jsonify({'error': 'influencer required'}), 400
        
        try:
            amount = float(amount)
            if amount <= 0:
                return jsonify({'error': 'amount must be > 0'}), 400
        except (ValueError, TypeError):
            return jsonify({'error': 'amount must be a number'}), 400
        
        # In a real system, this would process payouts via Stripe, PayPal, etc.
        # For now, just log it
        payout_log = {
            'influencer': influencer_handle,
            'amount': amount,
            'status': 'simulated',  # In production: pending, completed, failed
            'processed_by': payload.get('email'),
            'timestamp': datetime.datetime.utcnow().isoformat()
        }
        
        return jsonify({
            'success': True,
            'message': f'Payout of ${amount} simulated for {influencer_handle}',
            'payout': payout_log
        }), 200
    except Exception as e:
        print('[ERROR] influencer-payout:', e)
        return jsonify({'error': str(e)}), 500


# ---------------------- Server-Sent Events (SSE) for Live Feed ----------------------
_sse_clients = []

@app.route('/api/influencer-stream')
def influencer_stream():
    """SSE endpoint: stream live influencer events to clients."""
    try:
        payload = _decode_jwt_payload_from_header()
        if payload is None or payload.get('role') not in ('influencer', 'admin'):
            return jsonify({'error': 'Unauthorized'}), 401
        
        def event_stream():
            """Generate SSE events from the influencer_events collection."""
            if influencer_events_collection is None:
                yield 'data: {"error": "mongo-unavailable"}\n\n'
                return
            
            last_timestamp = request.args.get('since')
            query = {}
            if last_timestamp:
                query['timestamp'] = {'$gte': last_timestamp}
            
            # Send recent events first
            cursor = influencer_events_collection.find(query).sort('timestamp', -1).limit(50)
            for event in cursor:
                yield f'data: {json.dumps({k: v for k, v in event.items() if k != "_id"})}\n\n'
            
            # In production, continue polling or use change streams
            # For now, send a heartbeat
            for i in range(600):  # 10 minutes max connection
                time.sleep(1)
                yield f': heartbeat {i}\n\n'
        
        return Response(event_stream(), mimetype='text/event-stream')
    except Exception as e:
        print('[ERROR] influencer-stream:', e)
        return jsonify({'error': str(e)}), 500


# ---------------------- Instagram Seller Authenticity Check Endpoints ----------------------

# Initialize authenticity collections on startup
@app.before_request
def _init_authenticity_on_startup():
    """Initialize authenticity schema once on first request."""
    if not hasattr(app, '_authenticity_initialized'):
        if AUTHENTICITY_MODULES_AVAILABLE and db is not None:
            try:
                init_authenticity_schema(db)
                app._authenticity_initialized = True
            except Exception as e:
                print(f"[WARNING] Could not initialize authenticity schema: {e}")


@app.route('/api/v1/check', methods=['POST'])
@rate_limit(max_requests=100, window_seconds=60)
def authenticity_check():
    """
    POST /api/v1/check
    
    Check Instagram profile/post authenticity.
    
    Request JSON:
    {
      "url": "https://instagram.com/seller_handle",
      "user_image": "<base64 optional>",  // User-supplied product image
      "check_type": "profile" | "post"     // Defaults to auto-detect
    }
    
    Response JSON:
    {
      "handle": "seller_handle",
      "score": 28,
      "verdict": "Likely Fake",
      "reasons": [
        "Profile age: 2 months",
        "No Instagram Shop found",
        ...
      ],
      "metadata": {
        "followers": 1743,
        "posts": 14,
        ...
      }
    }
    """
    if not AUTHENTICITY_MODULES_AVAILABLE:
        return jsonify({'error': 'Authenticity check service unavailable'}), 503
    
    try:
        data = request.get_json(force=True) or {}
        url = data.get('url', '').strip()
        
        if not url:
            return jsonify({'error': 'url required'}), 400
        
        # Initialize modules
        fetcher = InstagramMetadataFetcher(timeout=10)
        scorer = InstagramAuthenticityScorer()
        
        # Fetch metadata
        metadata = fetcher.fetch_profile_metadata(url)
        
        if 'error' in metadata:
            return jsonify({'error': metadata['error']}), 400
        
        # Score the profile
        score, verdict, reasons = scorer.score_profile(metadata)
        
        # Save check to database
        check_record = {
            'input_url': url,
            'handle': metadata.get('handle', ''),
            'score': score,
            'verdict': verdict,
            'reasons': reasons,
            'metadata': {
                'followers': metadata.get('follower_count', 0),
                'following': metadata.get('following_count', 0),
                'posts': metadata.get('post_count', 0),
                'verified': metadata.get('verified', False),
                'website': metadata.get('website'),
                'bio': metadata.get('bio', ''),
                'og_image': metadata.get('og_image'),
            },
            'created_at': datetime.datetime.utcnow().isoformat(),
        }
        
        # Get user ID from JWT if available
        payload = _decode_jwt_payload_from_header()
        if payload and payload.get('email'):
            check_record['user_email'] = payload.get('email')
        
        # Store in DB if available
        try:
            if db and 'checks' in db.list_collection_names():
                db['checks'].insert_one(check_record)
                check_record['_id'] = str(check_record['_id'])
        except Exception as e:
            print(f"[WARNING] Could not save check: {e}")
        
        return jsonify({
            'handle': metadata.get('handle', ''),
            'score': score,
            'verdict': verdict,
            'reasons': reasons,
            'metadata': check_record['metadata'],
        }), 200
    
    except Exception as e:
        print(f'[ERROR] authenticity-check: {e}')
        return jsonify({'error': str(e)}), 500


@app.route('/api/v1/checks/<check_id>', methods=['GET'])
def get_authenticity_check(check_id):
    """GET /api/v1/checks/<check_id>
    
    Retrieve a saved authenticity check by ID.
    """
    if not AUTHENTICITY_MODULES_AVAILABLE or db is None:
        return jsonify({'error': 'Service unavailable'}), 503
    
    try:
        from bson import ObjectId
        try:
            oid = ObjectId(check_id)
        except Exception:
            return jsonify({'error': 'Invalid check ID'}), 400
        
        check = db['checks'].find_one({'_id': oid})
        if not check:
            return jsonify({'error': 'Check not found'}), 404
        
        # Sanitize ObjectId
        check['_id'] = str(check['_id'])
        return jsonify(check), 200
    
    except Exception as e:
        print(f'[ERROR] get-authenticity-check: {e}')
        return jsonify({'error': str(e)}), 500


@app.route('/api/v1/brand/<domain>', methods=['GET'])
def get_brand_info(domain):
    """GET /api/v1/brand/<domain>
    
    Lookup a known brand by domain (e.g., nike.com).
    Returns trusted status and reference data.
    """
    if not AUTHENTICITY_MODULES_AVAILABLE or db is None:
        return jsonify({'brand': None}), 200
    
    try:
        brand = db['brands'].find_one({'domain': domain.lower()})
        if not brand:
            return jsonify({'brand': None}), 200
        
        brand['_id'] = str(brand['_id'])
        return jsonify({'brand': brand}), 200
    
    except Exception as e:
        print(f'[ERROR] get-brand: {e}')
        return jsonify({'error': str(e)}), 500


@app.route('/api/v1/brands', methods=['GET'])
def list_brands():
    """GET /api/v1/brands
    
    List all trusted brands (paginated).
    Query params: limit=50, page=0
    """
    if not AUTHENTICITY_MODULES_AVAILABLE or db is None:
        return jsonify({'brands': []}), 200
    
    try:
        limit = int(request.args.get('limit', 50))
        page = int(request.args.get('page', 0))
        skip = page * limit
        
        brands = list(db['brands'].find({})
                     .sort('name', 1)
                     .skip(skip)
                     .limit(limit))
        
        for b in brands:
            b['_id'] = str(b['_id'])
        
        total = db['brands'].count_documents({})
        
        return jsonify({
            'brands': brands,
            'total': total,
            'page': page,
            'limit': limit,
        }), 200
    
    except Exception as e:
        print(f'[ERROR] list-brands: {e}')
        return jsonify({'error': str(e)}), 500


@app.route('/api/v1/brand', methods=['POST'])
def create_or_update_brand():
    """POST /api/v1/brand (admin only)
    
    Create or update a trusted brand.
    
    Request JSON:
    {
      "name": "Nike",
      "domain": "nike.com",
      "official_handles": ["nike", "nikestore"],
      "verified_by_admin": true,
      "notes": "..."
    }
    """
    if not AUTHENTICITY_MODULES_AVAILABLE or db is None:
        return jsonify({'error': 'Service unavailable'}), 503
    
    # Check admin role
    payload = _decode_jwt_payload_from_header()
    if payload is None or payload.get('role') != 'admin':
        return jsonify({'error': 'Admin access required'}), 403
    
    try:
        data = request.get_json(force=True) or {}
        
        if not data.get('name') or not data.get('domain'):
            return jsonify({'error': 'name and domain required'}), 400
        
        brand_record = {
            'name': data['name'],
            'domain': data['domain'].lower(),
            'official_handles': data.get('official_handles', []),
            'verified_by_admin': data.get('verified_by_admin', False),
            'notes': data.get('notes', ''),
            'updated_at': datetime.datetime.utcnow().isoformat(),
        }
        
        result = db['brands'].update_one(
            {'name': brand_record['name']},
            {'$set': brand_record},
            upsert=True
        )
        
        return jsonify({
            'status': 'ok',
            'upserted': result.upserted_id is not None,
            'modified': result.modified_count > 0,
        }), 201
    except Exception as e:
        print(f'[ERROR] create-brand: {e}')
        return jsonify({'error': str(e)}), 500


# ======================== INFLUENCER VERIFICATION & AUTHENTICATION ========================

@app.route('/api/influencer/authenticate', methods=['POST'])
@rate_limit(max_requests=100, window_seconds=60)
def authenticate_influencer():
    """Verify and authenticate an influencer's Instagram account"""
    if not INFLUENCER_MODULES_AVAILABLE:
        return jsonify({'error': 'Influencer modules not available'}), 503
    
    try:
        data = request.get_json() or {}
        instagram_url = data.get('instagram_url') or data.get('url')
        
        if not instagram_url:
            return jsonify({'error': 'instagram_url or url required'}), 400
        
        # Authenticate
        authenticator = InfluencerAuthenticator()
        result = authenticator.authenticate_influencer(instagram_url)
        
        # Save to database
        try:
            if db and 'influencer_authenticity' in db.list_collection_names():
                db['influencer_authenticity'].insert_one({
                    'handle': result.get('handle'),
                    'score': result.get('score'),
                    'verdict': result.get('verdict'),
                    'verified_badge': result.get('verified_badge'),
                    'followers': result.get('followers'),
                    'tier': result.get('tier'),
                    'risk_flags': result.get('risk_flags', []),
                    'safe_to_hire': result.get('safe_to_hire'),
                    'checked_at': datetime.datetime.utcnow().isoformat(),
                    'full_data': result
                })
        except Exception as e:
            print(f'[WARNING] Could not save authentication: {e}')
        
        return jsonify(result), 200
    
    except Exception as e:
        print(f'[ERROR] authenticate_influencer: {e}')
        return jsonify({'error': str(e)}), 500


@app.route('/api/influencers/fraud-check', methods=['POST'])
@rate_limit(max_requests=50, window_seconds=60)
def fraud_check_influencers():
    """Analyze influencers for fraudulent activity"""
    if not INFLUENCER_MODULES_AVAILABLE:
        return jsonify({'error': 'Influencer modules not available'}), 503
    
    try:
        data = request.get_json() or {}
        influencer_ids = data.get('influencer_ids', [])
        
        if not influencer_ids:
            return jsonify({'error': 'influencer_ids required'}), 400
        
        detector = InfluencerFraudDetector()
        reports = []
        
        for inf_id in influencer_ids:
            try:
                # Get influencer data and events
                if not db:
                    continue
                
                influencer = db.get_collection('influencer_settings').find_one({'_id': inf_id})
                events = list(db.get_collection('influencer_events').find({'influencer_id': inf_id}).limit(1000))
                
                if influencer:
                    report = detector.analyze_influencer_activity(influencer, events)
                    reports.append(report)
                    
                    # Save fraud log
                    try:
                        db['influencer_fraud_logs'].insert_one({
                            'influencer_id': inf_id,
                            'fraud_score': report.get('fraud_score'),
                            'risk_level': report.get('risk_level'),
                            'flags': report.get('fraud_flags'),
                            'detected_at': datetime.datetime.utcnow().isoformat(),
                            'full_report': report
                        })
                    except:
                        pass
            except Exception as e:
                print(f'[ERROR] fraud check for {inf_id}: {e}')
        
        bulk_report = detector.bulk_fraud_check([
            (r, []) for r in reports
        ]) if reports else {'individual_reports': [], 'summary': {}}
        
        return jsonify(bulk_report), 200
    
    except Exception as e:
        print(f'[ERROR] fraud_check_influencers: {e}')
        return jsonify({'error': str(e)}), 500


@app.route('/api/influencer/<influencer_id>/analytics', methods=['GET'])
@rate_limit(max_requests=100, window_seconds=60)
def get_influencer_analytics(influencer_id):
    """Get comprehensive analytics for an influencer"""
    if not INFLUENCER_MODULES_AVAILABLE:
        return jsonify({'error': 'Influencer modules not available'}), 503
    
    try:
        time_period = request.args.get('days', 30, type=int)
        
        analytics = InfluencerAnalytics()
        
        if not db:
            return jsonify({'error': 'Database unavailable'}), 503
        
        # Get influencer and events
        influencer = db.get_collection('influencer_settings').find_one({'_id': influencer_id})
        events = list(db.get_collection('influencer_events').find({'influencer_id': influencer_id}))
        
        if not influencer:
            return jsonify({'error': 'Influencer not found'}), 404
        
        result = analytics.calculate_comprehensive_metrics(influencer, events, time_period)
        
        # Cache analytics
        try:
            db['performance_analytics'].insert_one({
                'influencer_id': influencer_id,
                'data': result,
                'calculated_at': datetime.datetime.utcnow().isoformat()
            })
        except:
            pass
        
        return jsonify(result), 200
    
    except Exception as e:
        print(f'[ERROR] get_influencer_analytics: {e}')
        return jsonify({'error': str(e)}), 500


# ======================== CAMPAIGN MANAGEMENT ========================

@app.route('/api/campaigns', methods=['GET', 'POST'])
@rate_limit(max_requests=100, window_seconds=60)
def manage_campaigns():
    """Get or create campaigns"""
    try:
        if request.method == 'GET':
            if not db:
                return jsonify({'error': 'Database unavailable'}), 503
            
            status = request.args.get('status')
            limit = min(request.args.get('limit', 50, type=int), 500)
            
            query = {}
            if status:
                query['status'] = status
            
            campaigns = list(db['campaigns'].find(query).sort('created_at', -1).limit(limit))
            
            # Convert ObjectId to string
            for c in campaigns:
                c['_id'] = str(c['_id'])
            
            return jsonify({'campaigns': campaigns, 'total': len(campaigns)}), 200
        
        elif request.method == 'POST':
            payload = request.get_json() or {}
            
            # Validate required fields
            required = ['brand_id', 'title', 'budget', 'target_tier']
            if not all(payload.get(f) for f in required):
                return jsonify({'error': 'Missing required fields'}), 400
            
            campaign = {
                'brand_id': payload['brand_id'],
                'title': payload['title'],
                'description': payload.get('description', ''),
                'budget': float(payload['budget']),
                'target_tier': payload['target_tier'],
                'target_niche': payload.get('target_niche', []),
                'deliverables': payload.get('deliverables', {}),
                'commission_structure': payload.get('commission_structure', {}),
                'status': 'draft',
                'proposals_count': 0,
                'accepted_proposals': 0,
                'created_at': datetime.datetime.utcnow().isoformat(),
                'updated_at': datetime.datetime.utcnow().isoformat()
            }
            
            if db:
                result = db['campaigns'].insert_one(campaign)
                campaign['_id'] = str(result.inserted_id)
            
            return jsonify(campaign), 201
    
    except Exception as e:
        print(f'[ERROR] manage_campaigns: {e}')
        return jsonify({'error': str(e)}), 500


@app.route('/api/campaigns/<campaign_id>/proposals', methods=['GET', 'POST'])
@rate_limit(max_requests=100, window_seconds=60)
def campaign_proposals(campaign_id):
    """Get or send campaign proposals"""
    try:
        if request.method == 'GET':
            if not db:
                return jsonify({'error': 'Database unavailable'}), 503
            
            status = request.args.get('status')
            query = {'campaign_id': campaign_id}
            
            if status:
                query['status'] = status
            
            proposals = list(db['campaign_proposals'].find(query).sort('created_at', -1).limit(100))
            
            for p in proposals:
                p['_id'] = str(p['_id'])
            
            return jsonify({'proposals': proposals}), 200
        
        elif request.method == 'POST':
            payload = request.get_json() or {}
            
            required = ['influencer_id', 'influencer_handle']
            if not all(payload.get(f) for f in required):
                return jsonify({'error': 'Missing required fields'}), 400
            
            proposal = {
                'campaign_id': campaign_id,
                'influencer_id': payload['influencer_id'],
                'influencer_handle': payload['influencer_handle'],
                'influencer_followers': payload.get('influencer_followers', 0),
                'status': 'pending',
                'proposal_message': payload.get('message', ''),
                'requested_fee': float(payload.get('requested_fee', 0)),
                'terms': payload.get('terms', {}),
                'sent_at': datetime.datetime.utcnow().isoformat(),
                'expires_at': (datetime.datetime.utcnow() + datetime.timedelta(days=3)).isoformat()
            }
            
            if db:
                result = db['campaign_proposals'].insert_one(proposal)
                proposal['_id'] = str(result.inserted_id)
                
                # Update campaign proposal count
                db['campaigns'].update_one(
                    {'_id': campaign_id},
                    {'$inc': {'proposals_count': 1}}
                )
            
            return jsonify(proposal), 201
    
    except Exception as e:
        print(f'[ERROR] campaign_proposals: {e}')
        return jsonify({'error': str(e)}), 500


@app.route('/api/campaigns/<campaign_id>/proposals/<proposal_id>/respond', methods=['POST'])
@rate_limit(max_requests=50, window_seconds=60)
def respond_to_proposal(campaign_id, proposal_id):
    """Influencer responds to campaign proposal"""
    try:
        payload = request.get_json() or {}
        response_status = payload.get('status')  # accepted or rejected
        
        if response_status not in ['accepted', 'rejected']:
            return jsonify({'error': 'Invalid status'}), 400
        
        if db:
            db['campaign_proposals'].update_one(
                {'_id': proposal_id, 'campaign_id': campaign_id},
                {'$set': {
                    'status': response_status,
                    'response_at': datetime.datetime.utcnow().isoformat(),
                    'counter_offer': payload.get('counter_offer')
                }}
            )
            
            if response_status == 'accepted':
                db['campaigns'].update_one(
                    {'_id': campaign_id},
                    {'$inc': {'accepted_proposals': 1}}
                )
        
        return jsonify({'status': 'ok', 'proposal_status': response_status}), 200
    
    except Exception as e:
        print(f'[ERROR] respond_to_proposal: {e}')
        return jsonify({'error': str(e)}), 500


# ======================== INFLUENCER MARKETPLACE ========================

@app.route('/api/marketplace/influencers', methods=['GET'])
@rate_limit(max_requests=200, window_seconds=60)
def marketplace_search():
    """Search and filter influencers on marketplace"""
    try:
        if not db:
            return jsonify({'error': 'Database unavailable'}), 503
        
        # Build query
        query = {'available_for_hire': True}
        
        # Filters
        tier = request.args.get('tier')
        niche = request.args.get('niche')
        min_followers = request.args.get('min_followers', type=int)
        min_engagement = request.args.get('min_engagement', type=float)
        min_authenticity = request.args.get('min_authenticity', 0, type=int)
        
        if tier:
            query['tier'] = tier
        if niche:
            query['niche'] = niche
        if min_followers:
            query['followers'] = {'$gte': min_followers}
        if min_engagement:
            query['engagement_rate'] = {'$gte': min_engagement}
        if min_authenticity > 0:
            query['authenticity_score'] = {'$gte': min_authenticity}
        
        # Sort
        sort_by = request.args.get('sort_by', 'followers')
        order = -1 if request.args.get('order', 'desc') == 'desc' else 1
        
        limit = min(request.args.get('limit', 50, type=int), 500)
        
        influencers = list(db['influencer_marketplace'].find(query)
                          .sort(sort_by, order)
                          .limit(limit))
        
        # Sanitize
        for inf in influencers:
            inf['_id'] = str(inf.get('_id', ''))
        
        return jsonify({
            'influencers': influencers,
            'total': len(influencers),
            'filters_applied': {
                'tier': tier,
                'niche': niche,
                'min_followers': min_followers,
                'min_engagement': min_engagement,
                'min_authenticity': min_authenticity
            }
        }), 200
    
    except Exception as e:
        print(f'[ERROR] marketplace_search: {e}')
        return jsonify({'error': str(e)}), 500


@app.route('/api/marketplace/influencers/<influencer_id>', methods=['GET'])
@rate_limit(max_requests=200, window_seconds=60)
def get_marketplace_influencer(influencer_id):
    """Get detailed influencer profile for marketplace"""
    try:
        if not db:
            return jsonify({'error': 'Database unavailable'}), 503
        
        inf = db['influencer_marketplace'].find_one({'_id': influencer_id})
        
        if not inf:
            return jsonify({'error': 'Influencer not found'}), 404
        
        inf['_id'] = str(inf.get('_id', ''))
        
        # Include recent analytics
        analytics = db['performance_analytics'].find_one({'influencer_id': influencer_id})
        if analytics:
            inf['recent_analytics'] = analytics.get('data')
        
        return jsonify(inf), 200
    
    except Exception as e:
        print(f'[ERROR] get_marketplace_influencer: {e}')
        return jsonify({'error': str(e)}), 500


@app.route('/api/marketplace/recommendations', methods=['GET'])
@rate_limit(max_requests=100, window_seconds=60)
def get_marketplace_recommendations():
    """Get AI-generated influencer recommendations based on brand niche"""
    try:
        if not db:
            return jsonify({'error': 'Database unavailable'}), 503
        
        niche = request.args.get('niche', 'general')
        budget = request.args.get('budget', type=float)
        
        # Find matching influencers
        query = {'available_for_hire': True}
        if niche != 'general':
            query['niche'] = niche
        
        # Sort by engagement rate and authenticity
        influencers = list(db['influencer_marketplace'].find(query)
                          .sort([('engagement_rate', -1), ('authenticity_score', -1)])
                          .limit(10))
        
        # Filter by budget if specified
        if budget:
            recommended = [i for i in influencers if i.get('rate_per_post', 0) <= budget]
        else:
            recommended = influencers
        
        for inf in recommended:
            inf['_id'] = str(inf.get('_id', ''))
        
        return jsonify({
            'recommendations': recommended[:5],
            'niche': niche,
            'budget_filter': budget
        }), 200
    
    except Exception as e:
        print(f'[ERROR] get_marketplace_recommendations: {e}')
        return jsonify({'error': str(e)}), 500


# ======================== A/B TESTING ========================

@app.route('/api/campaigns/<campaign_id>/ab-tests', methods=['GET', 'POST'])
@rate_limit(max_requests=100, window_seconds=60)
def manage_ab_tests(campaign_id):
    """Create and manage A/B tests for campaigns"""
    try:
        if request.method == 'GET':
            if not db:
                return jsonify({'error': 'Database unavailable'}), 503
            
            tests = list(db['ab_tests'].find({'campaign_id': campaign_id}))
            
            for t in tests:
                t['_id'] = str(t.get('_id', ''))
            
            return jsonify({'tests': tests}), 200
        
        elif request.method == 'POST':
            payload = request.get_json() or {}
            
            required = ['test_name', 'test_type']
            if not all(payload.get(f) for f in required):
                return jsonify({'error': 'Missing required fields'}), 400
            
            ab_test = {
                'campaign_id': campaign_id,
                'test_name': payload['test_name'],
                'test_type': payload['test_type'],  # hashtag, posting_time, etc.
                'variant_a': payload.get('variant_a', {}),
                'variant_b': payload.get('variant_b', {}),
                'status': 'draft',
                'created_at': datetime.datetime.utcnow().isoformat()
            }
            
            if db:
                result = db['ab_tests'].insert_one(ab_test)
                ab_test['_id'] = str(result.inserted_id)
            
            return jsonify(ab_test), 201
    
    except Exception as e:
        print(f'[ERROR] manage_ab_tests: {e}')
        return jsonify({'error': str(e)}), 500

# ==================== INSTAGRAM INFLUENCER VERIFICATION ENDPOINTS ====================

@app.route('/api/influencer/verify-instagram', methods=['POST'])
@rate_limit(max_requests=100, window_seconds=60)
def verify_influencer_instagram():
    """
    Verify Instagram influencer account
    
    Request:
    {
        "instagram_url": "https://instagram.com/username",
        "instagram_handle": "username"  (alternative to URL)
    }
    
    Response:
    {
        "handle": "username",
        "verification_badge": true/false,
        "followers": 150000,
        "following": 500,
        "posts": 250,
        "authenticity_score": 85.5,
        "verdict": "genuine",  (genuine/suspicious/likely_fake)
        "safe_to_hire": true,
        "engagement_rate": 0.0250,
        "engagement_authenticity": 78.0,
        "risk_flags": ["string"],
        "tier": "macro",
        "recommendations": ["string"],
        "checked_at": "2025-12-09T..."
    }
    """
    if not INFLUENCER_MODULES_AVAILABLE:
        return jsonify({'error': 'Influencer modules not available'}), 503
    
    try:
        data = request.get_json() or {}
        instagram_url = data.get('instagram_url')
        instagram_handle = data.get('instagram_handle')
        
        if not instagram_url and not instagram_handle:
            return jsonify({'error': 'instagram_url or instagram_handle required'}), 400
        
        # Use URL or handle
        input_account = instagram_url or instagram_handle
        
        # Verify
        verifier = InstagramInfluencerVerifier()
        result = verifier.verify_instagram_account(input_account)
        
        if 'error' in result:
            return jsonify(result), 400
        
        # Save to database
        try:
            if db and 'influencer_instagram_verification' in db.list_collection_names():
                db['influencer_instagram_verification'].insert_one({
                    'handle': result.get('handle'),
                    'verification_badge': result.get('verification_badge'),
                    'authenticity_score': result.get('authenticity_score'),
                    'verdict': result.get('verdict'),
                    'safe_to_hire': result.get('safe_to_hire'),
                    'followers': result.get('followers'),
                    'engagement_rate': result.get('engagement_rate'),
                    'tier': result.get('tier'),
                    'risk_flags': result.get('risk_flags', []),
                    'checked_at': datetime.datetime.utcnow().isoformat(),
                    'full_data': result
                })
        except Exception as e:
            print(f'[WARNING] Could not save verification: {e}')
        
        return jsonify(result), 200
    
    except Exception as e:
        print(f'[ERROR] verify_influencer_instagram: {e}')
        return jsonify({'error': str(e)}), 500


@app.route('/api/influencer/verify-instagram/batch', methods=['POST'])
@rate_limit(max_requests=50, window_seconds=60)
def verify_influencers_batch():
    """
    Verify multiple Instagram influencer accounts
    
    Request:
    {
        "influencers": [
            "https://instagram.com/user1",
            "https://instagram.com/user2"
        ]
    }
    
    Response:
    {
        "verified": 2,
        "results": [
            {handle, verification_badge, authenticity_score, verdict, ...},
            ...
        ]
    }
    """
    if not INFLUENCER_MODULES_AVAILABLE:
        return jsonify({'error': 'Influencer modules not available'}), 503
    
    try:
        data = request.get_json() or {}
        influencers = data.get('influencers', [])
        
        if not influencers:
            return jsonify({'error': 'influencers array required'}), 400
        
        if len(influencers) > 50:
            return jsonify({'error': 'Maximum 50 influencers per request'}), 400
        
        verifier = InstagramInfluencerVerifier()
        results = verifier.compare_influencers(influencers)
        
        # Save all results
        try:
            if db and 'influencer_instagram_verification' in db.list_collection_names():
                for result in results:
                    if 'error' not in result:
                        db['influencer_instagram_verification'].insert_one({
                            'handle': result.get('handle'),
                            'verification_badge': result.get('verification_badge'),
                            'authenticity_score': result.get('authenticity_score'),
                            'verdict': result.get('verdict'),
                            'safe_to_hire': result.get('safe_to_hire'),
                            'followers': result.get('followers'),
                            'engagement_rate': result.get('engagement_rate'),
                            'tier': result.get('tier'),
                            'risk_flags': result.get('risk_flags', []),
                            'checked_at': datetime.datetime.utcnow().isoformat(),
                            'full_data': result
                        })
        except Exception as e:
            print(f'[WARNING] Could not save batch verifications: {e}')
        
        return jsonify({
            'verified': len(results),
            'results': results
        }), 200
    
    except Exception as e:
        print(f'[ERROR] verify_influencers_batch: {e}')
        return jsonify({'error': str(e)}), 500


@app.route('/api/influencer/instagram-check/<handle>', methods=['GET'])
@rate_limit(max_requests=200, window_seconds=60)
def get_instagram_verification(handle):
    """
    Get cached Instagram verification for handle
    
    Response:
    {
        "cached": true,
        "verification_badge": true,
        "authenticity_score": 85.5,
        "verdict": "genuine",
        "safe_to_hire": true,
        "engagement_rate": 0.025,
        "followers": 150000,
        "tier": "macro",
        "checked_at": "2025-12-09T..."
    }
    """
    try:
        if not db or 'influencer_instagram_verification' not in db.list_collection_names():
            return jsonify({'error': 'Database not available'}), 503
        
        # Find cached verification
        verification = db['influencer_instagram_verification'].find_one(
            {'handle': handle.lower()},
            sort=[('checked_at', -1)]
        )
        
        if not verification:
            return jsonify({'error': 'No verification found for this handle'}), 404
        
        # Remove MongoDB _id from response
        verification.pop('_id', None)
        verification['cached'] = True
        
        return jsonify(verification), 200
    
    except Exception as e:
        print(f'[ERROR] get_instagram_verification: {e}')
        return jsonify({'error': str(e)}), 500


@app.route('/api/influencer/instagram-badge-check/<handle>', methods=['GET'])
@rate_limit(max_requests=300, window_seconds=60)
def check_verification_badge(handle):
    """
    Quick check: Does influencer have Instagram verification badge?
    
    Response:
    {
        "handle": "username",
        "has_verification_badge": true,
        "authenticity_score": 85.5,
        "safe_to_hire": true,
        "followers": 150000,
        "checked_at": "2025-12-09T..."
    }
    """
    try:
        if not db or 'influencer_instagram_verification' not in db.list_collection_names():
            # Try to verify now
            if not INFLUENCER_MODULES_AVAILABLE:
                return jsonify({'error': 'Verification service not available'}), 503
            
            verifier = InstagramInfluencerVerifier()
            result = verifier.verify_instagram_account(handle)
            
            if 'error' in result:
                return jsonify(result), 400
            
            return jsonify({
                'handle': result.get('handle'),
                'has_verification_badge': result.get('verification_badge'),
                'authenticity_score': result.get('authenticity_score'),
                'safe_to_hire': result.get('safe_to_hire'),
                'followers': result.get('followers'),
                'tier': result.get('tier'),
                'checked_at': result.get('checked_at')
            }), 200
        
        # Get from cache
        verification = db['influencer_instagram_verification'].find_one(
            {'handle': handle.lower()},
            sort=[('checked_at', -1)]
        )
        
        if not verification:
            # Verify now if not cached
            if not INFLUENCER_MODULES_AVAILABLE:
                return jsonify({'error': 'Verification service not available'}), 503
            
            verifier = InstagramInfluencerVerifier()
            result = verifier.verify_instagram_account(handle)
            
            if 'error' in result:
                return jsonify(result), 400
            
            return jsonify({
                'handle': result.get('handle'),
                'has_verification_badge': result.get('verification_badge'),
                'authenticity_score': result.get('authenticity_score'),
                'safe_to_hire': result.get('safe_to_hire'),
                'followers': result.get('followers'),
                'tier': result.get('tier'),
                'checked_at': result.get('checked_at')
            }), 200
        
        return jsonify({
            'handle': verification.get('handle'),
            'has_verification_badge': verification.get('verification_badge'),
            'authenticity_score': verification.get('authenticity_score'),
            'safe_to_hire': verification.get('safe_to_hire'),
            'followers': verification.get('followers'),
            'tier': verification.get('tier'),
            'checked_at': verification.get('checked_at')
        }), 200
    
    except Exception as e:
        print(f'[ERROR] check_verification_badge: {e}')
        return jsonify({'error': str(e)}), 500


@app.route('/api/influencer/instagram-stats/<handle>', methods=['GET'])
@rate_limit(max_requests=200, window_seconds=60)
def get_instagram_stats(handle):
    """
    Get Instagram account statistics
    
    Response:
    {
        "handle": "username",
        "followers": 150000,
        "following": 500,
        "posts": 250,
        "engagement_rate": 0.025,
        "avg_likes_per_post": 3750,
        "avg_comments_per_post": 95,
        "follower_to_following_ratio": 300.0,
        "posts_per_month": 8.2,
        "tier": "macro",
        "checked_at": "2025-12-09T..."
    }
    """
    try:
        if not db or 'influencer_instagram_verification' not in db.list_collection_names():
            return jsonify({'error': 'Database not available'}), 503
        
        verification = db['influencer_instagram_verification'].find_one(
            {'handle': handle.lower()},
            sort=[('checked_at', -1)]
        )
        
        if not verification:
            return jsonify({'error': 'No data found for this handle'}), 404
        
        full_data = verification.get('full_data', {})
        
        return jsonify({
            'handle': full_data.get('handle'),
            'followers': full_data.get('followers'),
            'following': full_data.get('following'),
            'posts': full_data.get('posts'),
            'engagement_rate': full_data.get('engagement_rate'),
            'avg_likes_per_post': full_data.get('engagement_rate'),  # From full data
            'avg_comments_per_post': full_data.get('engagement_rate'),  # From full data
            'follower_to_following_ratio': full_data.get('follower_following_ratio'),
            'posts_per_month': full_data.get('posts_per_month'),
            'tier': full_data.get('tier'),
            'checked_at': full_data.get('checked_at')
        }), 200
    
    except Exception as e:
        print(f'[ERROR] get_instagram_stats: {e}')
        return jsonify({'error': str(e)}), 500


# Initialize influencer collections on startup
def _init_influencer_collections_on_startup():
    """Initialize influencer collections"""
    try:
        if not INFLUENCER_MODULES_AVAILABLE or not db:
            return
        
        init_influencer_campaigns_schema(db)
        init_payment_schema(db)
        print("[INFO] Influencer collections initialized")
    except Exception as e:
        print(f"[WARNING] Could not init influencer collections: {e}")


@app.before_request
def _init_on_first_request_influencer():
    """Initialize influencer collections on first request"""
    if not hasattr(_init_on_first_request_influencer, 'done'):
        _init_influencer_collections_on_startup()
        _init_on_first_request_influencer.done = True


if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    debug_mode = os.getenv('FLASK_DEBUG', 'false').lower() in ('1', 'true', 'yes')
    print("\n" + "="*60)
    print("🚀 BRAND AUTH BACKEND STARTING")
    print("="*60)
    print(f"✅ Flask app initialized")
    print(f"📦 MongoDB connection: {'✅ Connected' if _mongo_connected else '⚠️ Not connected'}")
    if _mongo_connect_error:
        print(f"   Error: {_mongo_connect_error}")
    print(f"🌐 Server: http://0.0.0.0:{port}")
    print(f"🔧 Debug mode: {'ON' if debug_mode else 'OFF'}")
    print(f"📡 CORS: Enabled (all origins)")
    print("="*60)
    print("Endpoints available:")
    print("  /api/health_check         - Server status")
    print("  /api/signup               - Register new user")
    print("  /api/login                - Login user")
    print("  /api/multimodal_check     - Verify product (file/URL + text)")
    print("  /api/influencer/*         - Influencer verification")
    print("="*60 + "\n")
    
    try:
        app.run(host='0.0.0.0', port=port, debug=debug_mode, use_reloader=False)
    except KeyboardInterrupt:
        print("\n\n🛑 Server shutdown requested (Ctrl+C)")
    except Exception as e:
        print(f"\n\n❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()


