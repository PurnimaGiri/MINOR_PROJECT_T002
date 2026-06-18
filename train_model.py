"""
HOAMS - Step 2: Train Random Forest Models
Run AFTER generate_dataset.py: python train_model.py
Outputs:
  - models/slot_model.pkl
  - models/noshow_model.pkl
  - models/label_encoder.pkl
  - results_slot_classification_report.txt
  - results_noshow_comparison.txt
  - chart_feature_importance.png
  - chart_confusion_matrix.png
  - chart_roc_curve.png
  - chart_model_comparison.png
"""

import pandas as pd
import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings("ignore")

# ── Check dependencies
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import (classification_report, confusion_matrix,
                                  roc_auc_score, roc_curve, accuracy_score,
                                  precision_score, recall_score, f1_score)
    from sklearn.preprocessing import LabelEncoder
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.svm import SVC
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    print("All libraries loaded successfully.")
except ImportError as e:
    print(f"Missing library: {e}")
    print("Run: pip install scikit-learn imbalanced-learn matplotlib pandas numpy")
    exit(1)

os.makedirs("models", exist_ok=True)

# ════════════════════════════════════════════════
# 1. LOAD DATA
# ════════════════════════════════════════════════
print("\n[1/6] Loading dataset...")
df = pd.read_csv("appointments_dataset.csv")
print(f"     Loaded {len(df)} records, {df.shape[1]} columns")

FEATURES = [
    "patient_age", "patient_gender", "distance_km", "is_new_patient",
    "department_id", "visit_type", "appointment_hour", "day_of_week",
    "month", "lead_time_days", "is_holiday",
    "prior_noshow_count", "avg_past_duration", "appointments_90d"
]

X = df[FEATURES]
y_slot   = df["slot_duration_class"]
y_noshow = df["no_show"]

# ════════════════════════════════════════════════
# 2. TRAIN-TEST SPLIT
# ════════════════════════════════════════════════
print("\n[2/6] Splitting data (80% train / 20% test)...")
X_train, X_test, y_slot_train, y_slot_test = train_test_split(
    X, y_slot, test_size=0.2, random_state=42, stratify=y_slot)
_, _, y_noshow_train, y_noshow_test = train_test_split(
    X, y_noshow, test_size=0.2, random_state=42, stratify=y_noshow)
print(f"     Train: {len(X_train)} | Test: {len(X_test)}")

# ════════════════════════════════════════════════
# 3. HANDLE CLASS IMBALANCE (SMOTE for no-show)
# ════════════════════════════════════════════════
print("\n[3/6] Balancing no-show classes (random oversampling)...")
print(f"     Before balancing: {y_noshow_train.value_counts().to_dict()}")
# Manual oversampling of minority class (no-show=1)
np.random.seed(42)
X_train_arr = X_train.values
y_noshow_arr = y_noshow_train.values
minority_idx = np.where(y_noshow_arr == 1)[0]
majority_idx = np.where(y_noshow_arr == 0)[0]
n_oversample = len(majority_idx) - len(minority_idx)
oversample_idx = np.random.choice(minority_idx, size=n_oversample, replace=True)
X_train_sm = np.vstack([X_train_arr, X_train_arr[oversample_idx]])
y_noshow_train_sm = np.concatenate([y_noshow_arr, y_noshow_arr[oversample_idx]])
print(f"     After  balancing: {dict(zip(*np.unique(y_noshow_train_sm, return_counts=True)))}")

# ════════════════════════════════════════════════
# 4. TRAIN RANDOM FOREST MODELS
# ════════════════════════════════════════════════
print("\n[4/6] Training Random Forest models...")

# Model 1: Slot Duration
print("     Training Slot Duration Classifier (n=300 trees)...")
slot_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    min_samples_split=5,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1
)
slot_model.fit(X_train, y_slot_train)
slot_preds = slot_model.predict(X_test)
slot_acc = accuracy_score(y_slot_test, slot_preds)
print(f"     Slot Model Accuracy: {slot_acc*100:.2f}%")

# Model 2: No-Show
print("     Training No-Show Predictor (n=200 trees)...")
noshow_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
noshow_model.fit(X_train_sm, y_noshow_train_sm)
noshow_preds      = noshow_model.predict(X_test)
noshow_proba      = noshow_model.predict_proba(X_test)[:, 1]
noshow_auc        = roc_auc_score(y_noshow_test, noshow_proba)
noshow_acc        = accuracy_score(y_noshow_test, noshow_preds)
print(f"     No-Show Model AUC: {noshow_auc:.4f} | Accuracy: {noshow_acc*100:.2f}%")

# Save models
with open("models/slot_model.pkl",    "wb") as f: pickle.dump(slot_model,   f)
with open("models/noshow_model.pkl",  "wb") as f: pickle.dump(noshow_model, f)
with open("models/feature_names.pkl", "wb") as f: pickle.dump(FEATURES,     f)
print("     Models saved to /models/")

# ════════════════════════════════════════════════
# 5. RESULTS & REPORTS
# ════════════════════════════════════════════════
print("\n[5/6] Generating results and charts...")

# -- Classification Report (Slot)
slot_report = classification_report(
    y_slot_test, slot_preds,
    target_names=["Short (0-10min)", "Medium (10-20min)", "Long (20+min)"]
)
with open("results_slot_classification_report.txt", "w") as f:
    f.write("=== SLOT DURATION CLASSIFICATION REPORT ===\n\n")
    f.write(slot_report)
    f.write(f"\nOverall Accuracy: {slot_acc*100:.2f}%\n")
print("     Saved: results_slot_classification_report.txt")

# -- Baseline model comparison (No-Show)
models_compare = {
    "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
    "Naive Bayes":         GaussianNB(),
    "Decision Tree":       DecisionTreeClassifier(random_state=42),
    "SVM":                 SVC(probability=True, random_state=42, max_iter=1000),
}
comparison_results = []
print("     Training baseline models for comparison...")
for name, model in models_compare.items():
    model.fit(X_train_sm, y_noshow_train_sm)
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    comparison_results.append({
        "Model": name,
        "Accuracy": f"{accuracy_score(y_noshow_test, preds)*100:.1f}%",
        "Precision": f"{precision_score(y_noshow_test, preds):.2f}",
        "Recall":    f"{recall_score(y_noshow_test, preds):.2f}",
        "AUC-ROC":   f"{roc_auc_score(y_noshow_test, proba):.2f}",
    })
    print(f"       {name}: AUC={roc_auc_score(y_noshow_test, proba):.2f}")

comparison_results.append({
    "Model":     "Random Forest (Ours)",
    "Accuracy":  f"{noshow_acc*100:.1f}%",
    "Precision": f"{precision_score(y_noshow_test, noshow_preds):.2f}",
    "Recall":    f"{recall_score(y_noshow_test, noshow_preds):.2f}",
    "AUC-ROC":   f"{noshow_auc:.2f}",
})

comp_df = pd.DataFrame(comparison_results)
comp_df.to_csv("results_noshow_comparison.csv", index=False)
with open("results_noshow_comparison.txt", "w") as f:
    f.write("=== NO-SHOW MODEL COMPARISON ===\n\n")
    f.write(comp_df.to_string(index=False))
print("     Saved: results_noshow_comparison.txt")

# ════════════════════════════════════════════════
# 6. CHARTS
# ════════════════════════════════════════════════

colors = {
    "primary":   "#1F5C99",
    "secondary": "#2E75B6",
    "accent":    "#F4A261",
    "bg":        "#F8FAFC",
    "text":      "#1A1A2E",
}

# Chart 1: Feature Importance
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor(colors["bg"])
ax.set_facecolor(colors["bg"])

importances = slot_model.feature_importances_
feat_imp = pd.Series(importances, index=FEATURES).sort_values(ascending=True)
feat_labels = {
    "patient_age": "Patient Age", "patient_gender": "Patient Gender",
    "distance_km": "Distance (km)", "is_new_patient": "Is New Patient",
    "department_id": "Department", "visit_type": "Visit Type",
    "appointment_hour": "Appointment Hour", "day_of_week": "Day of Week",
    "month": "Month", "lead_time_days": "Lead Time (days)",
    "is_holiday": "Is Holiday", "prior_noshow_count": "Prior No-Shows",
    "avg_past_duration": "Avg Past Duration", "appointments_90d": "Appts (90 days)"
}
feat_imp.index = [feat_labels.get(x, x) for x in feat_imp.index]
bar_colors = [colors["accent"] if v > feat_imp.quantile(0.7) else colors["secondary"] for v in feat_imp.values]
bars = ax.barh(feat_imp.index, feat_imp.values, color=bar_colors, edgecolor="white", height=0.6)
ax.set_xlabel("Feature Importance (Gini)", fontsize=12, color=colors["text"])
ax.set_title("Feature Importance — Slot Duration Model\n(Random Forest)", fontsize=14, fontweight="bold", color=colors["text"], pad=15)
ax.tick_params(colors=colors["text"], labelsize=10)
for spine in ax.spines.values(): spine.set_visible(False)
ax.xaxis.grid(True, alpha=0.3, color="gray")
ax.set_axisbelow(True)
for bar, val in zip(bars, feat_imp.values):
    ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
            f"{val:.3f}", va="center", fontsize=9, color=colors["text"])
plt.tight_layout()
plt.savefig("chart_feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("     Saved: chart_feature_importance.png")

# Chart 2: Confusion Matrix (Slot)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.patch.set_facecolor(colors["bg"])

cm = confusion_matrix(y_slot_test, slot_preds)
im = axes[0].imshow(cm, cmap="Blues")
axes[0].set_xticks([0,1,2]); axes[0].set_yticks([0,1,2])
axes[0].set_xticklabels(["Short","Medium","Long"]); axes[0].set_yticklabels(["Short","Medium","Long"])
axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Actual")
axes[0].set_title("Confusion Matrix\n(Slot Duration)", fontweight="bold", color=colors["text"])
for i in range(3):
    for j in range(3):
        axes[0].text(j, i, str(cm[i,j]), ha="center", va="center",
                     color="white" if cm[i,j] > cm.max()/2 else colors["text"], fontsize=14, fontweight="bold")
axes[0].set_facecolor(colors["bg"])

# Chart 2b: No-show ROC
fpr, tpr, _ = roc_curve(y_noshow_test, noshow_proba)
axes[1].plot(fpr, tpr, color=colors["primary"], lw=2.5, label=f"Random Forest (AUC = {noshow_auc:.2f})")
axes[1].plot([0,1],[0,1], "k--", alpha=0.4, lw=1)
axes[1].fill_between(fpr, tpr, alpha=0.1, color=colors["primary"])
axes[1].set_xlabel("False Positive Rate"); axes[1].set_ylabel("True Positive Rate")
axes[1].set_title("ROC Curve — No-Show Prediction", fontweight="bold", color=colors["text"])
axes[1].legend(fontsize=11); axes[1].set_facecolor(colors["bg"])
for spine in axes[1].spines.values(): spine.set_color("lightgray")

plt.tight_layout()
plt.savefig("chart_roc_and_confusion.png", dpi=150, bbox_inches="tight")
plt.close()
print("     Saved: chart_roc_and_confusion.png")

# Chart 3: Model Comparison Bar Chart
fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor(colors["bg"])
ax.set_facecolor(colors["bg"])

model_names = [r["Model"] for r in comparison_results]
aucs        = [float(r["AUC-ROC"]) for r in comparison_results]
bar_cols    = [colors["accent"] if n == "Random Forest (Ours)" else colors["secondary"] for n in model_names]
bars = ax.bar(model_names, aucs, color=bar_cols, edgecolor="white", width=0.55)
ax.set_ylabel("AUC-ROC Score", fontsize=12, color=colors["text"])
ax.set_title("Model Comparison — No-Show Prediction\nAUC-ROC Scores", fontsize=14, fontweight="bold", color=colors["text"], pad=15)
ax.set_ylim(0.6, 0.95)
ax.tick_params(colors=colors["text"]); ax.set_xticklabels(model_names, rotation=15, ha="right", fontsize=10)
for spine in ax.spines.values(): spine.set_visible(False)
ax.yaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)
for bar, val in zip(bars, aucs):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.003, f"{val:.2f}",
            ha="center", va="bottom", fontsize=11, fontweight="bold", color=colors["text"])
rf_patch = mpatches.Patch(color=colors["accent"], label="Random Forest (Ours)")
ax.legend(handles=[rf_patch], fontsize=10)
plt.tight_layout()
plt.savefig("chart_model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("     Saved: chart_model_comparison.png")

# ════════════════════════════════════════════════
print("\n" + "="*55)
print("  TRAINING COMPLETE — SUMMARY")
print("="*55)
print(f"  Slot Duration Model Accuracy : {slot_acc*100:.2f}%")
print(f"  No-Show Model AUC-ROC        : {noshow_auc:.4f}")
print(f"  No-Show Model Accuracy       : {noshow_acc*100:.2f}%")
print("\n  Files generated:")
print("  models/slot_model.pkl")
print("  models/noshow_model.pkl")
print("  chart_feature_importance.png")
print("  chart_roc_and_confusion.png")
print("  chart_model_comparison.png")
print("  results_slot_classification_report.txt")
print("  results_noshow_comparison.txt")
print("="*55)
print("\n  Next step: python app.py")
