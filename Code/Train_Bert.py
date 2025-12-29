# =========================================================
# Train_Bert.py  (Windows + CPU safe version)
# - Uppercase 2-letter labels (e.g., CM, FR, MN, PE, PO, RL, SE, US)
# - Fixes common Windows 0xC0000005 crashes:
#     * force single-thread
#     * disable fast tokenizer
#     * smaller max_length + batch sizes (you can increase later)
# =========================================================

import os

# ---- MUST be set BEFORE importing torch/transformers ----
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import pandas as pd
import numpy as np
import joblib
import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    set_seed
)

# ---- Force single-thread inside PyTorch (Windows stability) ----
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# =========================
# CONFIGURATION
# =========================
DATA_PATH = r"D:\Masters\FH\Master_Thesis_2025\Thesis_Refinement\Dataset\Master_dataset\Real_Dataset\From_Agile_user_story_NFRs\10006_dataset_Multi.csv"

TEXT_COL  = "Requirement"
LABEL_COL = "Type"

OUTPUT_DIR = "./bert_type_model"
MODEL_DIR = os.path.join(OUTPUT_DIR, "model")
TOKENIZER_DIR = os.path.join(OUTPUT_DIR, "tokenizer")
LABEL_ENCODER_PATH = os.path.join(OUTPUT_DIR, "label_encoder.joblib")

MODEL_NAME = "bert-base-uncased"

TEST_SIZE = 0.2
RANDOM_STATE = 42

# ---- safer defaults for Windows/CPU; increase later if stable ----
MAX_LENGTH = 128
EPOCHS = 3
TRAIN_BATCH_SIZE = 2
EVAL_BATCH_SIZE = 4
LEARNING_RATE = 2e-5

# If you want to restrict labels to known codes:
ALLOWED_LABELS = {"CM", "FR", "MN", "PE", "PO", "RL", "SE", "US"}

os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

set_seed(RANDOM_STATE)

# =========================
# LOAD + CLEAN DATA
# =========================
df = pd.read_csv(DATA_PATH, encoding="utf-8")
df = df.dropna(subset=[TEXT_COL, LABEL_COL]).reset_index(drop=True)

print("\nLabel distribution BEFORE cleaning:")
print(df[LABEL_COL].value_counts(dropna=False))

# Minimal cleaning (you said labels are already uppercase)
df[LABEL_COL] = df[LABEL_COL].astype(str).str.strip()

# Drop blanks
df = df[df[LABEL_COL] != ""].reset_index(drop=True)

# OPTIONAL: drop unknown labels to prevent hidden noise
unknown = sorted(set(df[LABEL_COL].unique()) - ALLOWED_LABELS)
if unknown:
    print("\nWARNING: Unknown labels found and will be removed:", unknown)
df = df[df[LABEL_COL].isin(ALLOWED_LABELS)].reset_index(drop=True)

print("\nLabel distribution AFTER cleaning:")
print(df[LABEL_COL].value_counts(dropna=False))

# =========================
# ENCODE LABELS
# =========================
label_encoder = LabelEncoder()
df["label_id"] = label_encoder.fit_transform(df[LABEL_COL])

class_names = list(label_encoder.classes_)
num_labels = len(class_names)

print("\nClasses:", class_names)
print("Number of labels:", num_labels)

joblib.dump(label_encoder, LABEL_ENCODER_PATH)
print("Saved label encoder to:", LABEL_ENCODER_PATH)

# =========================
# TRAIN/VAL SPLIT
# =========================
train_df, val_df = train_test_split(
    df,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=df["label_id"]
)

print("\nTrain size:", len(train_df), "Val size:", len(val_df))

# =========================
# TOKENIZER + DATASET
# =========================
# IMPORTANT: use_fast=False (Windows crash fix)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

class ReqDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts.astype(str).tolist()
        self.labels = labels.astype(int).tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length
        )
        enc["labels"] = self.labels[idx]
        return {k: torch.tensor(v) for k, v in enc.items()}

train_dataset = ReqDataset(train_df[TEXT_COL], train_df["label_id"], tokenizer, MAX_LENGTH)
val_dataset   = ReqDataset(val_df[TEXT_COL],   val_df["label_id"],   tokenizer, MAX_LENGTH)

# =========================
# LOAD MODEL
# =========================
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels
).to(device)

# =========================
# METRICS
# =========================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, preds)}

# =========================
# TRAINING ARGS
# =========================
training_args = TrainingArguments(
    output_dir=os.path.join(OUTPUT_DIR, "training_output"),
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,

    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,

    # Windows/CPU stability:
    dataloader_num_workers=0,
    dataloader_pin_memory=False,

    logging_steps=50,
    seed=RANDOM_STATE,
    report_to="none",

    # disable fp16 on CPU
    fp16=torch.cuda.is_available(),
)

trainer = Trainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# =========================
# TRAIN
# =========================
print("\n>>> Starting trainer.train() now...")
trainer.train()

# =========================
# EVALUATE
# =========================
print("\n=== Evaluation on validation set ===")
eval_results = trainer.evaluate()
print("Eval results:", eval_results)

pred_output = trainer.predict(val_dataset)
logits = pred_output.predictions
y_pred = np.argmax(logits, axis=-1)
y_true = val_df["label_id"].values

print("\n=== Classification report (Type) ===")
print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

print("\n=== Confusion matrix ===")
print(confusion_matrix(y_true, y_pred))

# =========================
# SAVE MODEL + TOKENIZER + LABEL ENCODER
# =========================
trainer.save_model(MODEL_DIR)                 # best practice with Trainer
tokenizer.save_pretrained(TOKENIZER_DIR)      # save tokenizer
joblib.dump(label_encoder, LABEL_ENCODER_PATH)

print("\nSaved fine-tuned model to:", MODEL_DIR)
print("Saved tokenizer to:", TOKENIZER_DIR)
print("Saved label encoder to:", LABEL_ENCODER_PATH)

# =========================
# OPTIONAL: PREDICT ON FULL DATASET
# =========================
from torch.nn.functional import softmax

def predict_proba(texts, batch_size=16):
    all_probs = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            enc = tokenizer(
                list(batch_texts),
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt"
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            outputs = model(**enc)
            probs = softmax(outputs.logits, dim=-1)
            all_probs.append(probs.cpu().numpy())
    return np.vstack(all_probs)

print("\n=== Predicting on full dataset with fine-tuned BERT ===")
all_texts = df[TEXT_COL].astype(str).tolist()
all_probs = predict_proba(all_texts, batch_size=16)

all_pred_ids = np.argmax(all_probs, axis=1)
all_pred_labels = label_encoder.inverse_transform(all_pred_ids)

out_df = df.copy()
out_df["BERT_Type_pred"] = all_pred_labels

for idx, class_name in enumerate(class_names):
    out_df[f"prob_{class_name}"] = all_probs[:, idx]

full_out_path = os.path.join(OUTPUT_DIR, "dataset_with_bert_predictions.csv")
out_df.to_csv(full_out_path, index=False, encoding="utf-8")
print("Saved full dataset with BERT predictions to:", full_out_path)
