"""
train_model.py
--------------
Training pipeline for the Loan Approval Prediction System.
- Loads data from data/
- Engineers features
- Trains and compares SVM, Random Forest, XGBoost
- Saves the best model + metadata to model/

Run:
    python train_model.py
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, roc_auc_score, f1_score, roc_curve)

# Allow running from project root or any subdirectory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backend.model_utils import (
    load_raw_data, engineer_features_for_training,
    MODEL_DIR, MODEL_FILE, FEATURE_FILE, MAPPING_FILE,
    PREPROCESSING_MAPPINGS
)

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# DIRECTORIES
# ─────────────────────────────────────────────────────────────────────────────

ROOT_DIR         = os.path.dirname(os.path.abspath(__file__))
REPORT_IMGS_DIR  = os.path.join(ROOT_DIR, 'report_images')
os.makedirs(MODEL_DIR,        exist_ok=True)
os.makedirs(REPORT_IMGS_DIR,  exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("  LOAN APPROVAL PREDICTION — TRAINING PIPELINE")
print("="*60)

print("\n[1/5] Loading data...")
raw_df = load_raw_data()
print(f"      Dataset shape: {raw_df.shape}")
print(f"      Missing values:\n{raw_df.isnull().sum()[raw_df.isnull().sum() > 0]}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

print("\n[2/5] Feature engineering...")
X, Y = engineer_features_for_training(raw_df)
print(f"      Feature columns: {list(X.columns)}")
print(f"      Training samples: {len(X)}")
print(f"      Class distribution:\n{Y.value_counts().to_string()}")

# Train / test split
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, stratify=Y, random_state=42)
print(f"      Train: {X_train.shape}  |  Test: {X_test.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — DATA VISUALIZATIONS  (saved to report_images/)
# ─────────────────────────────────────────────────────────────────────────────

print("\n[3/5] Generating data visualizations...")

# Analytics copy (still has readable labels for plots)
analytics_df = raw_df.copy()
analytics_df = analytics_df.dropna()
analytics_df['Dependents'] = analytics_df['Dependents'].replace('3+', 4)
analytics_df['TotalIncome'] = (analytics_df['ApplicantIncome']
                                + analytics_df['CoapplicantIncome'])
analytics_df['DebtToIncomeRatio'] = (
    analytics_df['LoanAmount'] / analytics_df['TotalIncome'].replace(0, 1))

fig, axes = plt.subplots(2, 3, figsize=(16, 10), facecolor='#1e2130')
fig.suptitle('Loan Approval — Exploratory Data Analysis',
             fontsize=16, color='#90caf9', fontweight='bold', y=1.01)

plot_pairs = [
    ('Education',       'Loan Status vs Education'),
    ('Married',         'Loan Status vs Marital Status'),
    ('Property_Area',   'Loan Status vs Property Area'),
]
for ax, (col, title) in zip(axes[0], plot_pairs):
    ax.set_facecolor('#1e2130')
    sns.countplot(x=col, hue='Loan_Status', data=analytics_df, ax=ax,
                  palette={'Y': '#42a5f5', 'N': '#ef5350'})
    ax.set_title(title, color='#90caf9', fontsize=11)
    ax.tick_params(colors='#aaaaaa')
    ax.set_xlabel(col, color='#aaaaaa')
    ax.set_ylabel('Count', color='#aaaaaa')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333355')
    ax.legend(title='Loan Status', facecolor='#1e2130', labelcolor='#e0e0e0')

box_pairs = [
    ('TotalIncome',       'Total Income vs Loan Status'),
    ('DebtToIncomeRatio', 'Debt-to-Income vs Loan Status'),
    ('LoanAmount',        'Loan Amount vs Loan Status'),
]
for ax, (col, title) in zip(axes[1], box_pairs):
    ax.set_facecolor('#1e2130')
    plot_df = analytics_df.copy()
    if col == 'TotalIncome':
        plot_df[col] = plot_df[col].clip(upper=25000)
    if col == 'DebtToIncomeRatio':
        plot_df[col] = plot_df[col].clip(upper=0.15)
    sns.boxplot(x='Loan_Status', y=col, data=plot_df, ax=ax,
                palette={'Y': '#42a5f5', 'N': '#ef5350'})
    ax.set_title(title, color='#90caf9', fontsize=11)
    ax.tick_params(colors='#aaaaaa')
    ax.set_xlabel('Loan Status', color='#aaaaaa')
    ax.set_ylabel(col, color='#aaaaaa')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333355')

plt.tight_layout()
eda_path = os.path.join(REPORT_IMGS_DIR, 'data_visualization.png')
plt.savefig(eda_path, dpi=120, bbox_inches='tight', facecolor='#1e2130')
plt.close()
print(f"      Saved: {eda_path}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — MODEL TRAINING & COMPARISON
# ─────────────────────────────────────────────────────────────────────────────

print("\n[4/5] Training models...")

candidates = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM':           svm.SVC(kernel='linear', probability=True, random_state=42),
}
if XGBOOST_AVAILABLE:
    candidates['XGBoost'] = xgb.XGBClassifier(
        random_state=42, eval_metric='logloss', use_label_encoder=False)

results = {}
print(f"\n  {'Model':<20} {'Train Acc':>10} {'Test Acc':>10} {'F1':>8} {'ROC-AUC':>10} {'CV Mean':>10}")
print("  " + "-"*70)

for name, model in candidates.items():
    model.fit(X_train, Y_train)
    train_pred  = model.predict(X_train)
    test_pred   = model.predict(X_test)
    test_proba  = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

    train_acc = accuracy_score(Y_train, train_pred)
    test_acc  = accuracy_score(Y_test,  test_pred)
    f1        = f1_score(Y_test, test_pred)
    roc_auc   = roc_auc_score(Y_test, test_proba) if test_proba is not None else float('nan')
    cv_scores = cross_val_score(model, X_train, Y_train, cv=5, scoring='accuracy')

    results[name] = {
        'model': model, 'train_acc': train_acc, 'test_acc': test_acc,
        'f1': f1, 'roc_auc': roc_auc, 'cv_mean': cv_scores.mean(),
        'test_pred': test_pred, 'test_proba': test_proba,
    }
    print(f"  {name:<20} {train_acc:>10.4f} {test_acc:>10.4f} {f1:>8.4f} {roc_auc:>10.4f} {cv_scores.mean():>10.4f}")

# Best model
best_name  = max(results, key=lambda n: results[n]['test_acc'])
best       = results[best_name]
best_model = best['model']
print(f"\n  Best model: {best_name}  (Test Accuracy: {best['test_acc']:.4f})")

# Classification report
print(f"\n  Classification Report — {best_name}")
print("  " + "-"*50)
print(classification_report(Y_test, best['test_pred'],
                             target_names=['Rejected', 'Approved']))

# ─── Model comparison chart ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor='#1e2130')
fig.suptitle('Model Comparison', fontsize=15, color='#90caf9',
             fontweight='bold')

metric_cols = ['Test Acc', 'F1 Score', 'ROC-AUC']
metric_vals = {
    name: [
        results[name]['test_acc'],
        results[name]['f1'],
        results[name]['roc_auc'],
    ]
    for name in results
}

colors_map = {'XGBoost': '#42a5f5', 'Random Forest': '#66bb6a', 'SVM': '#ffa726'}
for ax, metric_idx, metric_label in zip(axes, range(3), metric_cols):
    ax.set_facecolor('#1e2130')
    names  = list(metric_vals.keys())
    vals   = [metric_vals[n][metric_idx] for n in names]
    colors = [colors_map.get(n, '#90caf9') for n in names]
    bars   = ax.bar(names, vals, color=colors)
    ax.set_ylim(0.5, 1.0)
    ax.set_title(metric_label, color='#90caf9', fontsize=12)
    ax.tick_params(colors='#aaaaaa')
    ax.set_ylabel('Score', color='#aaaaaa')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333355')
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', va='bottom',
                color='#e0e0e0', fontsize=9)

plt.tight_layout()
cmp_path = os.path.join(REPORT_IMGS_DIR, 'model_comparison.png')
plt.savefig(cmp_path, dpi=120, bbox_inches='tight', facecolor='#1e2130')
plt.close()
print(f"\n      Saved: {cmp_path}")

# ─── Confusion matrix for best model ─────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor='#1e2130')
fig.suptitle(f'Evaluation — {best_name}', fontsize=14,
             color='#90caf9', fontweight='bold')

# Confusion matrix
ax = axes[0]
ax.set_facecolor('#1e2130')
cm = confusion_matrix(Y_test, best['test_pred'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Rejected', 'Approved'],
            yticklabels=['Rejected', 'Approved'])
ax.set_title('Confusion Matrix', color='#90caf9')
ax.tick_params(colors='#aaaaaa')
ax.set_xlabel('Predicted', color='#aaaaaa')
ax.set_ylabel('Actual', color='#aaaaaa')

# ROC Curve
ax = axes[1]
ax.set_facecolor('#1e2130')
if best['test_proba'] is not None:
    fpr, tpr, _ = roc_curve(Y_test, best['test_proba'])
    ax.plot(fpr, tpr, color='#42a5f5', linewidth=2,
            label=f"AUC = {best['roc_auc']:.3f}")
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
    ax.set_xlabel('False Positive Rate', color='#aaaaaa')
    ax.set_ylabel('True Positive Rate', color='#aaaaaa')
    ax.legend(facecolor='#1e2130', labelcolor='#e0e0e0')
ax.set_title('ROC Curve', color='#90caf9')
ax.tick_params(colors='#aaaaaa')
for spine in ax.spines.values():
    spine.set_edgecolor('#333355')

# Feature importance
ax = axes[2]
ax.set_facecolor('#1e2130')
if hasattr(best_model, 'feature_importances_'):
    fi = pd.Series(best_model.feature_importances_, index=X.columns).sort_values()
    fi.tail(10).plot.barh(ax=ax, color='#42a5f5')
    ax.set_title('Top 10 Feature Importances', color='#90caf9')
elif hasattr(best_model, 'coef_'):
    fi = pd.Series(abs(best_model.coef_[0]), index=X.columns).sort_values()
    fi.tail(10).plot.barh(ax=ax, color='#ffa726')
    ax.set_title('Top 10 Feature Coefficients', color='#90caf9')
else:
    ax.text(0.5, 0.5, 'Not available', ha='center', va='center',
            transform=ax.transAxes, color='#aaaaaa')
    ax.set_title('Feature Importance', color='#90caf9')
ax.tick_params(colors='#aaaaaa')
ax.set_xlabel('Importance', color='#aaaaaa')
for spine in ax.spines.values():
    spine.set_edgecolor('#333355')

plt.tight_layout()
eval_path = os.path.join(REPORT_IMGS_DIR, 'model_evaluation.png')
plt.savefig(eval_path, dpi=120, bbox_inches='tight', facecolor='#1e2130')
plt.close()
print(f"      Saved: {eval_path}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — SAVE MODEL
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n[5/5] Saving model to model/ ...")
joblib.dump(best_model,            MODEL_FILE)
joblib.dump(X.columns.tolist(),    FEATURE_FILE)
joblib.dump(PREPROCESSING_MAPPINGS, MAPPING_FILE)

print(f"      Saved: {MODEL_FILE}")
print(f"      Saved: {FEATURE_FILE}")
print(f"      Saved: {MAPPING_FILE}")

print("\n" + "="*60)
print("  Training complete. Run the dashboard:")
print("  streamlit run frontend/app.py")
print("="*60 + "\n")
