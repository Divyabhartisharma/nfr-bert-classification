import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# =========================================================
# 1) PATHS
# =========================================================
#model_path = r"D:\Masters\FH\Master_Thesis_2025\Thesis_Refinement\Classification_NFR\Test3_After_Dataset_Cleaning\with_iso25010_4nfrs\Iter1_chatgpt_iso25010.csv"
#gt_path    = r"D:\Masters\FH\Master_Thesis_2025\Thesis_Refinement\Classification_NFR\Dataset\data\output_csv\PURE_cleaned_strict.csv"
model_path = r"D:\Masters\FH\Master_Thesis_2025\Thesis_Refinement\Classification_NFR\Test3_After_Dataset_Cleaning\with_iso25010\iter1_Gemini_Requirements_Classified_Detailed_NFR.csv"
gt_path    = r"D:\Masters\FH\Master_Thesis_2025\Thesis_Refinement\Classification_NFR\Dataset\data\output_csv\PURE_cleaned_strict.csv"
# =========================================================
# 2) LOAD
# =========================================================
model_df = pd.read_csv(model_path)
gt_df    = pd.read_csv(gt_path)

print("Model columns:", model_df.columns.tolist())
print("GT columns:   ", gt_df.columns.tolist())

# Must contain Requirement + Type
for df, name in [(model_df, "Model"), (gt_df, "GT")]:
    if "Requirement" not in df.columns or "Type" not in df.columns:
        raise ValueError(f"{name} CSV must contain 'Requirement' and 'Type' columns.")

# =========================================================
# 3) MERGE
# =========================================================
merged = pd.merge(
    model_df[["Requirement", "Type"]],
    gt_df[["Requirement", "Type"]],
    on="Requirement",
    suffixes=("_model", "_gt")
)

print(f"\nTotal matched rows: {len(merged)}")
if len(merged) == 0:
    raise ValueError("No rows matched on 'Requirement'. Check text/encoding/whitespace.")

# =========================================================
# 4) NORMALIZE LABELS TO A COMMON ISO NAME SPACE
#    (DO NOT collapse valid ISO classes into Other)
# =========================================================

# Choose ONE standard representation: ISO full characteristic names
# This mapping only fixes formatting differences / synonyms.
# It does NOT change meaning and does NOT remove ISO categories.

ISO_NAME_MAP = {
    # Short codes -> ISO names (if GT uses codes)
    "PE": "Performance Efficiency",
    "SE": "Security",
    "US": "Usability",
    "O":  "Other",

    # Common variants -> ISO names
    "Performance": "Performance Efficiency",
    "Performance efficiency": "Performance Efficiency",
    "Security ": "Security",
    "Usability ": "Usability",
    "Other ": "Other",


    "Interaction Capability": "Usability",

   #(valid ISO names)
    "Functional Suitability": "Functional Suitability",
    "Compatibility": "Compatibility",
    "Reliability": "Reliability",
    "Maintainability": "Maintainability",
    "Portability": "Portability",
    "Safety": "Safety",
    "Flexibility": "Flexibility",
}

def normalize_iso_name(x):
    if pd.isna(x):
        return None
    x = str(x).strip()
    return ISO_NAME_MAP.get(x, x)  # keep unknown labels as-is (so you notice them)

merged["Type_model_norm"] = merged["Type_model"].apply(normalize_iso_name)
merged["Type_gt_norm"]    = merged["Type_gt"].apply(normalize_iso_name)

# =========================================================
# 5) SANITY CHECK: SEE LABELS ON BOTH SIDES
# =========================================================
print("\nUnique GT labels (normalized):", sorted(merged["Type_gt_norm"].dropna().unique()))
print("Unique Model labels (normalized):", sorted(merged["Type_model_norm"].dropna().unique()))

print("\nGT label counts:\n", merged["Type_gt_norm"].value_counts(dropna=False))
print("\nModel label counts:\n", merged["Type_model_norm"].value_counts(dropna=False))

# Optional: if you want to evaluate ONLY on labels that exist in GT:
gt_label_set = set(merged["Type_gt_norm"].dropna().unique())
eval_df = merged[merged["Type_gt_norm"].isin(gt_label_set) & merged["Type_model_norm"].notna()].copy()

y_true = eval_df["Type_gt_norm"]
y_pred = eval_df["Type_model_norm"]

# Define labels order for confusion matrix/report:
labels = sorted(gt_label_set)

# =========================================================
# 6) METRICS
# =========================================================
print("\n" + "="*60)
print("TYPE CLASSIFICATION METRICS (Based on Base Prompt )")
print("="*60)

acc = accuracy_score(y_true, y_pred)
print(f"Accuracy: {acc:.4f}\n")

print("Classification Report:")
print(classification_report(
    y_true, y_pred,
    labels=labels,
    zero_division=0
))

cm = confusion_matrix(y_true, y_pred, labels=labels)
print("Confusion Matrix [rows=true, cols=pred]:")
print("Labels order:", labels)
print(cm)

# =========================================================
# 7) SAVE MISMATCHES
# =========================================================
mismatches = eval_df[y_true != y_pred].copy()
mismatches.to_csv("mismatches_type.csv", index=False)

print("\nSaved mismatches to: mismatches_type.csv")
