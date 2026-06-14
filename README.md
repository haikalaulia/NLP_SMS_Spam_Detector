# SpamSense — SMS Spam Detector Indonesia

Aplikasi web detektor SMS spam berbahasa Indonesia berbasis **IndoBERT** dan **XLM-RoBERTa** dengan backend **FastAPI + ONNX Runtime**.

---

## Struktur Proyek

```
sms-spam-detector/
├── backend/
│   ├── main.py              ← FastAPI app (API endpoint)
│   ├── export_onnx.py       ← Script export model PyTorch → ONNX
│   └── requirements.txt
├── frontend/
│   └── index.html           ← Web UI (buka langsung di browser)
├── models/
│   ├── indobert.onnx        ← (diisi setelah export)
│   └── xlmroberta.onnx      ← (diisi setelah export)
└── README.md
```

---

## Setup & Cara Menjalankan

### 1. Install dependensi backend

```bash
cd backend
pip install -r requirements.txt
```

### 2. Export model ke ONNX (setelah fine-tuning selesai)

Jalankan dari folder `backend/`:

```bash
# Export IndoBERT
python export_onnx.py \
  --model indobert \
  --checkpoint /path/ke/folder/checkpoint/indobert

# Export XLM-RoBERTa
python export_onnx.py \
  --model xlmroberta \
  --checkpoint /path/ke/folder/checkpoint/xlmroberta
```

> Hasil export tersimpan otomatis ke `models/indobert.onnx` dan `models/xlmroberta.onnx`.

**Catatan:** Jika model belum siap, backend otomatis berjalan dalam **mode mock** (simulasi prediksi berbasis rule sederhana) sehingga UI tetap bisa didemonstrasikan.

### 3. Jalankan backend

```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

API akan berjalan di: `http://localhost:8000`

Dokumentasi otomatis tersedia di: `http://localhost:8000/docs`

### 4. Buka frontend

Cukup buka file `frontend/index.html` langsung di browser:

```
Klik dua kali → index.html
```

Atau via VS Code Live Server, atau:

```bash
# Dari folder sms-spam-detector/
python -m http.server 5500 --directory frontend
# Buka http://localhost:5500
```

---

## API Endpoints

| Method | Endpoint   | Deskripsi                              |
|--------|------------|----------------------------------------|
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
        ├── AutoTokenizer       ← tokenisasi teks
        │
        └── ONNX Runtime        ← inferensi model .onnx
                │
                └── softmax → label + confidence scores
```

---

## Mengapa ONNX?

- **Lebih cepat** dari PyTorch pada CPU (graph optimization)
- **Ringan** — tidak perlu load PyTorch penuh di server
- **Portabel** — bisa dijalankan di berbagai environment
- **Mudah di-deploy** ke Hugging Face Spaces / Docker

---

## Deployment ke Hugging Face Spaces (opsional)

1. Buat Space baru → pilih **Gradio** atau **Static** (untuk HTML murni)
2. Upload semua file
3. Untuk Space dengan backend FastAPI, gunakan template **Docker**:

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY backend/requirements.txt .
RUN pip install -r requirements.txt
COPY backend/ ./backend/
COPY models/ ./models/
COPY frontend/ ./frontend/
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]
```

---

## Tim Pengembang

Proyek Akhir — SINF6054 Pemrosesan Bahasa Alami Kelas A  
Kelompok 14  
- Tinsari Rauhana (2308107010038)  
- Naufal Hanif (2308107010025)  
- Haikal Aulia (2308107010063)
