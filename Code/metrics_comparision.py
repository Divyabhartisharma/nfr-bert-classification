import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# === 1. LOAD CSVs ===

#model_path = r"D:\Masters\FH\Master_Thesis_2025\Thesis_Refinement\Classification_NFR\test1\Prompt_ISO25010_with4nfrs\Iter1_chatgpt_iso25010.csv"      # model output
#gt_path    = r"D:\Masters\FH\Master_Thesis_2025\Thesis_Refinement\Dataset\Master_dataset\Real_Dataset\quantum_nlp_for_nfrs_dataset\PURE_Req_balanced.csv"           # ground truth
model_path = r"D:\Masters\FH\Master_Thesis_2025\Thesis_Refinement\Classification_NFR\Test3_After_Dataset_Cleaning\Base_Prompt\iter1_chatgpt_Requirements_Classified_Detailed_NFR.csv"      # model output
gt_path    = r"D:\Masters\FH\Master_Thesis_2025\Thesis_Refinement\Classification_NFR\Dataset\data\output_csv\PURE_cleaned_strict.csv"           # ground truth

model_df = pd.read_csv(model_path)
gt_df    = pd.read_csv(gt_path)

# Quick sanity check
print("Model columns:", model_df.columns.tolist())
print("GT columns:   ", gt_df.columns.tolist())


# === 2. MERGE ON REQUIREMENT TEXT ===
merged = pd.merge(
    model_df,
    gt_df,
    on="Requirement",
    suffixes=("_model", "_gt")
)

print(f"\nTotal matched rows: {len(merged)}")
if len(merged) == 0:
    raise ValueError("No rows matched on 'Requirement'. Check text or encoding.")


# === 3. FR vs NFR METRICS (Prediction column) ===
y_true_fr_nfr = merged["Prediction_gt"]
y_pred_fr_nfr = merged["Prediction_model"]

print("\n" + "="*60)
print("FR / NFR CLASSIFICATION METRICS")
print("="*60)

# Accuracy
acc_fr_nfr = accuracy_score(y_true_fr_nfr, y_pred_fr_nfr)
print(f"Accuracy (FR vs NFR): {acc_fr_nfr:.4f}\n")

# Classification report
print("Classification report (FR vs NFR):")
print(classification_report(y_true_fr_nfr, y_pred_fr_nfr))

# Confusion matrix
labels_fr_nfr = sorted(list(set(y_true_fr_nfr) | set(y_pred_fr_nfr)))
cm_fr_nfr = confusion_matrix(y_true_fr_nfr, y_pred_fr_nfr, labels=labels_fr_nfr)

print("Confusion matrix (FR vs NFR) [rows=true, cols=pred]:")
print("Labels order:", labels_fr_nfr)
print(cm_fr_nfr)


# === 4. NFR SUBTYPE METRICS (Type column) ===
# We usually only care about subtype when the requirement is NFR in ground truth
nfr_mask = merged["Prediction_gt"] == "NFR"
nfr_subset = merged[nfr_mask].copy()

print("\n" + "="*60)
print("NFR SUBTYPE METRICS (ISO 25010 categories)")
print("="*60)
print(f"Total NFR rows (from GT): {len(nfr_subset)}")

if len(nfr_subset) == 0:
    print("No NFR rows in ground truth – cannot compute subtype metrics.")
else:
    y_true_type = nfr_subset["Type_gt"]
    y_pred_type = nfr_subset["Type_model"]

    # Accuracy
    acc_type = accuracy_score(y_true_type, y_pred_type)
    print(f"\nAccuracy (NFR subtype / Type): {acc_type:.4f}\n")

    # Classification report
    print("Classification report (NFR subtype / Type):")
    print(classification_report(y_true_type, y_pred_type))

    # Confusion matrix
    labels_type = sorted(list(set(y_true_type) | set(y_pred_type)))
    cm_type = confusion_matrix(y_true_type, y_pred_type, labels=labels_type)

    print("Confusion matrix (NFR subtype) [rows=true, cols=pred]:")
    print("Labels order:", labels_type)
    print(cm_type)


# === 5. OPTIONAL: SAVE MISMATCHES TO CSV FOR INSPECTION ===
mismatch_fr_nfr = merged[y_true_fr_nfr != y_pred_fr_nfr]
mismatch_type = nfr_subset[y_true_type != y_pred_type] if len(nfr_subset) > 0 else pd.DataFrame()

mismatch_fr_nfr.to_csv("mismatches_fr_nfr.csv", index=False)
mismatch_type.to_csv("mismatches_nfr_type.csv", index=False)

print("\nSaved mismatches to:")
print(" - mismatches_fr_nfr.csv")
print(" - mismatches_nfr_type.csv")
