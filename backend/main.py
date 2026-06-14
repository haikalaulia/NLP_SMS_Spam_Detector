"""
SMS Spam Detector — FastAPI Backend
Menjalankan inferensi ONNX untuk IndoBERT dan XLM-RoBERTa
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import numpy as np
import os
import re
import time

# ── ONNX Runtime (opsional, fallback ke mode mock jika belum ada model) ──
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

# ── Tokenizer dari HuggingFace ──
try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


# ─────────────────────────────────────────────
# Konfigurasi
# ─────────────────────────────────────────────

LABEL_NAMES = ["normal", "promo", "penipuan"]
LABEL_EMOJI = {"normal": "✅", "promo": "📢", "penipuan": "⚠️"}
LABEL_COLOR = {"normal": "#22c55e", "promo": "#f59e0b", "penipuan": "#ef4444"}
MAX_LEN = 128

MODEL_CONFIGS = {
    "indobert": {
        "name": "IndoBERT",
        "description": "indobenchmark/indobert-base-p1",
        "onnx_path": os.path.join(os.path.dirname(__file__), "../models/indobert.onnx"),
        "tokenizer_name": "indobenchmark/indobert-base-p1",
    },
    "xlmroberta": {
        "name": "XLM-RoBERTa",
        "description": "xlm-roberta-base",
        "onnx_path": os.path.join(os.path.dirname(__file__), "../models/xlmroberta.onnx"),
        "tokenizer_name": "xlm-roberta-base",
    },
}

# ─────────────────────────────────────────────
# Model Loader
# ─────────────────────────────────────────────

sessions = {}
tokenizers = {}


def load_model(model_key: str):
    """Load ONNX session dan tokenizer, cache di memori."""
    config = MODEL_CONFIGS[model_key]

    # Tokenizer
    if model_key not in tokenizers and TRANSFORMERS_AVAILABLE:
        try:
            tokenizers[model_key] = AutoTokenizer.from_pretrained(config["tokenizer_name"])
        except Exception as e:
            print(f"[WARN] Gagal load tokenizer {model_key}: {e}")

    # ONNX Session
    if model_key not in sessions and ONNX_AVAILABLE:
        onnx_path = config["onnx_path"]
        if os.path.exists(onnx_path):
            try:
                sess_options = ort.SessionOptions()
                sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                sessions[model_key] = ort.InferenceSession(
                    onnx_path,
                    sess_options=sess_options,
                    providers=["CPUExecutionProvider"],
                )
                print(f"[OK] ONNX model loaded: {model_key}")
            except Exception as e:
                print(f"[WARN] Gagal load ONNX {model_key}: {e}")
        else:
            print(f"[INFO] Model file belum ada: {onnx_path}")


def preprocess_text(text: str) -> str:
    """Pembersihan teks SMS dasar."""
    text = text.strip()
    text = re.sub(r"http\S+|www\.\S+", "[URL]", text)
    text = re.sub(r"\s+", " ", text)
    return text


def softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - np.max(logits))
    return e / e.sum()


def predict_real(model_key: str, text: str) -> dict:
    """Inferensi menggunakan ONNX + tokenizer asli."""
    tokenizer = tokenizers.get(model_key)
    session = sessions.get(model_key)

    if tokenizer is None or session is None:
        return predict_mock(model_key, text)

    encoding = tokenizer(
        text,
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="np",
    )

    inputs = {
        "input_ids": encoding["input_ids"].astype(np.int64),
        "attention_mask": encoding["attention_mask"].astype(np.int64),
    }
    if "token_type_ids" in encoding:
        inputs["token_type_ids"] = encoding["token_type_ids"].astype(np.int64)

    outputs = session.run(None, inputs)
    logits = outputs[0][0]
    probs = softmax(logits)

    pred_idx = int(np.argmax(probs))
    return {
        "label": LABEL_NAMES[pred_idx],
        "confidence": float(probs[pred_idx]),
        "scores": {LABEL_NAMES[i]: float(probs[i]) for i in range(3)},
        "mode": "onnx",
    }


def predict_mock(model_key: str, text: str) -> dict:
    """
    Mock prediction — digunakan saat model ONNX belum tersedia.
    Simulasi berbasis rule sederhana untuk demo UI.
    """
    text_lower = text.lower()

    PENIPUAN_KEYWORDS = ["menang", "hadiah", "transfer", "rekening", "pin", "otp",
                         "klik", "selamat", "jackpot", "undian", "verifikasi", "pinjaman",
                         "kode", "konfirmasi", "password", "dompet", "dana", "ovo"]
    PROMO_KEYWORDS = ["promo", "diskon", "gratis", "bonus", "kuota", "paket",
                      "voucher", "cashback", "telkomsel", "indosat", "xl", "axis",
                      "tri", "beli", "harga", "murah", "spesial", "hemat"]

    penipuan_score = sum(k in text_lower for k in PENIPUAN_KEYWORDS) * 0.15
    promo_score = sum(k in text_lower for k in PROMO_KEYWORDS) * 0.15

    # Sedikit variasi antar model
    np.random.seed(hash(text + model_key) % (2**31))
    noise = np.random.dirichlet([1, 1, 1]) * 0.08

    base = np.array([
        max(0.1, 0.8 - penipuan_score - promo_score),
        min(0.85, promo_score + 0.1),
        min(0.85, penipuan_score + 0.05),
    ])
    probs = softmax(base + noise)

    pred_idx = int(np.argmax(probs))
    return {
        "label": LABEL_NAMES[pred_idx],
        "confidence": float(probs[pred_idx]),
        "scores": {LABEL_NAMES[i]: float(probs[i]) for i in range(3)},
        "mode": "mock",
    }


# ─────────────────────────────────────────────
# FastAPI App
# ─────────────────────────────────────────────

app = FastAPI(
    title="SMS Spam Detector API",
    description="Klasifikasi SMS spam Bahasa Indonesia — IndoBERT vs XLM-RoBERTa",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Load semua model saat startup
@app.on_event("startup")
async def startup_event():
    for key in MODEL_CONFIGS:
        load_model(key)


# ── Schema ──

class PredictRequest(BaseModel):
    text: str
    model: Optional[str] = "both"  # "indobert" | "xlmroberta" | "both"


class ModelResult(BaseModel):
    label: str
    confidence: float
    scores: dict
    mode: str
    emoji: str
    color: str
    latency_ms: float


class PredictResponse(BaseModel):
    text: str
    text_clean: str
    results: dict[str, ModelResult]


# ── Endpoints ──

@app.get("/")
def root():
    return {"message": "SMS Spam Detector API", "status": "running"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "onnx_available": ONNX_AVAILABLE,
        "transformers_available": TRANSFORMERS_AVAILABLE,
        "loaded_models": list(sessions.keys()),
    }


@app.get("/models")
def list_models():
    return {
        key: {
            "name": cfg["name"],
            "description": cfg["description"],
            "loaded": key in sessions,
        }
        for key, cfg in MODEL_CONFIGS.items()
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Teks tidak boleh kosong.")

    text_clean = preprocess_text(req.text)

    # Tentukan model mana yang dijalankan
    if req.model == "both":
        model_keys = list(MODEL_CONFIGS.keys())
    elif req.model in MODEL_CONFIGS:
        model_keys = [req.model]
    else:
        raise HTTPException(status_code=400, detail=f"Model tidak dikenal: {req.model}")

    results = {}
    for key in model_keys:
        t0 = time.perf_counter()
        pred = predict_real(key, text_clean)
        latency = (time.perf_counter() - t0) * 1000

        results[key] = ModelResult(
            label=pred["label"],
            confidence=pred["confidence"],
            scores=pred["scores"],
            mode=pred["mode"],
            emoji=LABEL_EMOJI[pred["label"]],
            color=LABEL_COLOR[pred["label"]],
            latency_ms=round(latency, 1),
        )

    return PredictResponse(text=req.text, text_clean=text_clean, results=results)
