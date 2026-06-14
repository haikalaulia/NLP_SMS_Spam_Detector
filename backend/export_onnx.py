"""
Export model PyTorch ke format ONNX.

Jalankan setelah fine-tuning selesai:
  python export_onnx.py --model indobert --checkpoint /path/to/checkpoint
  python export_onnx.py --model xlmroberta --checkpoint /path/to/checkpoint

Output: models/indobert.onnx  atau  models/xlmroberta.onnx
"""

import argparse
import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

LABEL2ID = {"normal": 0, "promo": 1, "penipuan": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
MAX_LEN = 128

MODEL_HF_NAMES = {
    "indobert": "indobenchmark/indobert-base-p1",
    "xlmroberta": "xlm-roberta-base",
}

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "../models")


def export(model_key: str, checkpoint_path: str):
    hf_name = MODEL_HF_NAMES[model_key]
    output_path = os.path.join(OUTPUT_DIR, f"{model_key}.onnx")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"[INFO] Memuat tokenizer: {hf_name}")
    tokenizer = AutoTokenizer.from_pretrained(hf_name)

    print(f"[INFO] Memuat model dari checkpoint: {checkpoint_path}")
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint_path,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    model.eval()

    # Dummy input untuk export
    dummy_text = "Selamat Anda menang hadiah"
    encoding = tokenizer(
        dummy_text,
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]

    dynamic_axes = {
        "input_ids": {0: "batch"},
        "attention_mask": {0: "batch"},
        "logits": {0: "batch"},
    }

    input_names = ["input_ids", "attention_mask"]
    inputs = (input_ids, attention_mask)

    # token_type_ids untuk BERT
    if "token_type_ids" in encoding:
        token_type_ids = encoding["token_type_ids"]
        input_names.append("token_type_ids")
        inputs = (input_ids, attention_mask, token_type_ids)
        dynamic_axes["token_type_ids"] = {0: "batch"}

    print(f"[INFO] Mengekspor ke: {output_path}")
    with torch.no_grad():
        torch.onnx.export(
            model,
            inputs,
            output_path,
            input_names=input_names,
            output_names=["logits"],
            dynamic_axes=dynamic_axes,
            opset_version=14,
            do_constant_folding=True,
        )

    print(f"[OK] Export selesai → {output_path}")

    # Verifikasi
    import onnxruntime as ort
    sess = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
    ort_inputs = {
        "input_ids": input_ids.numpy().astype(np.int64),
        "attention_mask": attention_mask.numpy().astype(np.int64),
    }
    if "token_type_ids" in encoding:
        ort_inputs["token_type_ids"] = encoding["token_type_ids"].numpy().astype(np.int64)

    ort_out = sess.run(None, ort_inputs)
    print(f"[OK] Verifikasi ONNX berhasil. Logits shape: {ort_out[0].shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["indobert", "xlmroberta"], required=True)
    parser.add_argument("--checkpoint", required=True, help="Path ke folder checkpoint PyTorch")
    args = parser.parse_args()
    export(args.model, args.checkpoint)
