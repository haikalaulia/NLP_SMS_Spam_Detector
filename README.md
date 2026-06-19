# SpamSense — SMS Spam Detector Indonesia

Aplikasi web detektor SMS spam berbahasa Indonesia berbasis **IndoBERT** dan **XLM-RoBERTa** dengan backend **FastAPI + ONNX Runtime**.

---

## Struktur Proyek

```
sms-spam-detector/
├── backend/
│   ├── main.py              ← FastAPI app (API endpoint)
│   └── requirements.txt
├── frontend/
│   └── index.html           ← Web UI (buka langsung di browser)
├── models/
│   ├── indobert.onnx               ← model IndoBERT (dari hasil fine-tuning)
│   ├── xlmroberta.onnx             ← model XLM-RoBERTa (dari hasil fine-tuning)
│   ├── indobert_tokenizer/         ← tokenizer.json, tokenizer_config.json, config.json
│   └── xlmroberta_tokenizer/       ← tokenizer.json, tokenizer_config.json, config.json
└── README.md
```

> Model `.onnx` dan tokenizer sudah disiapkan langsung saat proses fine-tuning oleh tim. Backend membaca keduanya dari file/folder lokal — **tidak ada download dari HuggingFace Hub** dan **tidak butuh PyTorch**, cukup ONNX Runtime untuk inferensi.

---

## Setup & Cara Menjalankan

### 1. Install dependensi backend

```bash
cd backend
pip install -r requirements.txt
```

### 2. Taruh model ONNX dan tokenizer

Letakkan file/folder ke `models/` dengan nama tepat seperti ini:

```
models/indobert.onnx
models/xlmroberta.onnx
models/indobert_tokenizer/        (isi: tokenizer.json, tokenizer_config.json, config.json, dll)
models/xlmroberta_tokenizer/      (isi: tokenizer.json, tokenizer_config.json, config.json, dll)
```

**Catatan:** Jika model atau tokenizer belum ditaruh, backend otomatis berjalan dalam **mode mock** (simulasi prediksi berbasis rule sederhana) untuk model yang belum lengkap, sehingga UI tetap bisa didemonstrasikan.

### 3. Jalankan backend

```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

API akan berjalan di: `http://localhost:8000`
Dokumentasi otomatis tersedia di: `http://localhost:8000/docs`

### 4. Buka frontend

Cukup buka file `frontend/index.html` langsung di browser (klik dua kali), atau:

```bash
# Dari folder sms-spam-detector/
python -m http.server 5500 --directory frontend
# Buka http://localhost:5500
```

---

## API Endpoints

| Method | Endpoint   | Deskripsi                              |
|--------|------------|-----------------------------------------|
| GET    | `/`        | Status API                             |
| GET    | `/health`  | Cek status + model yang dimuat         |
| GET    | `/models`  | Daftar model tersedia                  |
| POST   | `/predict` | Prediksi label SMS                     |

### Contoh request `/predict`

```json
POST /predict
{
  "text": "Selamat! No Anda menang hadiah Rp50 juta.",
  "model": "both"
}
```

Parameter `model`: `"both"` | `"indobert"` | `"xlmroberta"`

### Contoh response

```json
{
  "text": "Selamat! No Anda menang hadiah Rp50 juta.",
  "text_clean": "Selamat! No Anda menang hadiah Rp50 juta.",
  "results": {
    "indobert": {
      "label": "penipuan",
      "confidence": 0.9231,
      "scores": { "normal": 0.0312, "promo": 0.0457, "penipuan": 0.9231 },
      "mode": "onnx",
      "emoji": "⚠️",
      "color": "#ef4444",
      "latency_ms": 142.3
    },
    "xlmroberta": {
      "label": "penipuan",
      "confidence": 0.8874,
      ...
    }
  }
}
```

---

## Cara Kerja

```
[Browser / index.html]
        │
        │  POST /predict  (JSON)
        ▼
[FastAPI — main.py]
        │
        ├── preprocess_text()   ← bersihkan URL, spasi, dll.
        │
        ├── AutoTokenizer       ← tokenisasi teks (didownload otomatis
        │                          dari HuggingFace, sesuai base model)
        │
        └── ONNX Runtime        ← inferensi model .onnx
                │
                └── softmax → label + confidence scores
```

**Tentang tokenizer:** tokenizer dibaca dari folder lokal (`models/indobert_tokenizer/`, `models/xlmroberta_tokenizer/`) hasil `save_pretrained()` saat fine-tuning — **tidak didownload dari HuggingFace**, dan **tidak membutuhkan PyTorch**, karena tokenizer dimuat dalam mode numpy (`return_tensors="np"`).

---

## Mengapa ONNX?

- **Lebih cepat** dari PyTorch pada CPU (graph optimization)
- **Ringan** — tidak perlu load PyTorch penuh di server
- **Portabel** — bisa dijalankan di berbagai environment
- **Mudah di-deploy** ke Hugging Face Spaces / Docker

---

## Tim Pengembang

Proyek Akhir — SINF6054 Pemrosesan Bahasa Alami Kelas A
Kelompok 14
- Tinsari Rauhana (2308107010038)
- Naufal Hanif (2308107010025)
- Haikal Aulia (2308107010063)
