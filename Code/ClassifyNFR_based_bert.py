# =========================================================
# predict_nfr_subtype.py
# Use a fine-tuned BERT classifier to label NEW (unlabeled) requirements.
#
# INPUT  : CSV with a column "Requirement" (no Type column needed)
# OUTPUT : predicted_nfr_subtypes.csv with Predicted_Type + Confidence + prob_*
# =========================================================

import os
import pandas as pd
import numpy as np
import joblib
import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax

# -----------------------------
# CONFIG (change these)
# -----------------------------
# Path to your unlabeled CSV (must contain "Requirement" column)
DATA_PATH = r"D:\path\to\your\unlabeled_requirements.csv"   # <-- CHANGE THIS

TEXT_COL = "Requirement"

# Where your trained artifacts are saved (from Train_Bert.py)
OUTPUT_DIR = "./bert_type_model"
MODEL_DIR = os.path.join(OUTPUT_DIR, "model")
TOKENIZER_DIR = os.path.join(OUTPUT_DIR, "tokenizer")
LABEL_ENCODER_PATH = os.path.join(OUTPUT_DIR, "label_encoder.joblib")

# Safer defaults (same style as your training)
MAX_LENGTH = 128
BATCH_SIZE = 16

# Output file
OUT_PATH = "predicted_nfr_subtypes.csv"


def main():
    # -----------------------------
    # Load input data
    # -----------------------------
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Input CSV not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH, encoding="utf-8")
    if TEXT_COL not in df.columns:
        raise ValueError(f"Input CSV must contain a '{TEXT_COL}' column. Found: {df.columns.tolist()}")

    df = df.dropna(subset=[TEXT_COL]).reset_index(drop=True)
    df[TEXT_COL] = df[TEXT_COL].astype(str)

    if len(df) == 0:
        raise ValueError("No valid rows found after dropping missing Requirement values.")

    # -----------------------------
    # Load trained model artifacts
    # -----------------------------
    if not os.path.exists(MODEL_DIR):
        raise FileNotFoundError(f"MODEL_DIR not found: {MODEL_DIR}")
    if not os.path.exists(TOKENIZER_DIR):
        raise FileNotFoundError(f"TOKENIZER_DIR not found: {TOKENIZER_DIR}")
    if not os.path.exists(LABEL_ENCODER_PATH):
        raise FileNotFoundError(f"LABEL_ENCODER not found: {LABEL_ENCODER_PATH}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # use_fast=False for Windows stability (matches your training approach)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)

    class_names = list(label_encoder.classes_)
    print("Loaded classes:", class_names)

    model.eval()

    # -----------------------------
    # Predict probabilities
    # -----------------------------
    texts = df[TEXT_COL].tolist()
    all_probs = []

    with torch.no_grad():
        for i in range(0, len(texts), BATCH_SIZE):
            batch_texts = texts[i:i + BATCH_SIZE]

            enc = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt"
            )
            enc = {k: v.to(device) for k, v in enc.items()}

            outputs = model(**enc)
            probs = softmax(outputs.logits, dim=-1)  # [batch, num_labels]
            all_probs.append(probs.cpu().numpy())

    probs = np.vstack(all_probs)  # [N, num_labels]

    # -----------------------------
    # Convert to labels + confidence
    # -----------------------------
    pred_ids = probs.argmax(axis=1)
    pred_labels = label_encoder.inverse_transform(pred_ids)
    confidence = probs.max(axis=1)

    out_df = df.copy()
    out_df["Predicted_Type"] = pred_labels
    out_df["Confidence"] = confidence

    # Add prob columns
    for idx, name in enumerate(class_names):
        out_df[f"prob_{name}"] = probs[:, idx]

    # -----------------------------
    # Save output
    # -----------------------------
    out_df.to_csv(OUT_PATH, index=False, encoding="utf-8")
    print(f"Saved predictions to: {OUT_PATH}")
    print("Rows predicted:", len(out_df))


if __name__ == "__main__":
    main()
