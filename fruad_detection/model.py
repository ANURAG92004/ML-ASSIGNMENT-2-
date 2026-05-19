# ============================================================
#   CREDIT CARD FRAUD DETECTION - ML MODEL COMPARISON
#   Run this file in VS Code terminal: python model.py
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pickle
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, f1_score,
    precision_score, recall_score, accuracy_score
)

# ============================================================
# STEP 1 - LOAD DATASET
# ============================================================
print("\n" + "="*55)
print("  CREDIT CARD FRAUD DETECTION - ML PROJECT")
print("="*55)
print("\n[1/6] Loading dataset...")

df = pd.read_csv('../archive/creditcard.csv')
print(f"      Dataset shape: {df.shape}")
print(f"      Total transactions: {len(df):,}")
print(f"      Fraud cases: {df['Class'].sum():,} ({df['Class'].mean()*100:.3f}%)")
print(f"      Legitimate cases: {(df['Class']==0).sum():,}")

# ============================================================
# STEP 2 - EDA & VISUALIZATIONS
# ============================================================
print("\n[2/6] Generating EDA visualizations...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Credit Card Fraud Detection - EDA', fontsize=16, fontweight='bold')

# Class distribution
axes[0].bar(['Legitimate', 'Fraud'],
            [df['Class'].value_counts()[0], df['Class'].value_counts()[1]],
            color=['#1D9E75', '#D85A30'])
axes[0].set_title('Class Distribution (Imbalanced)')
axes[0].set_ylabel('Count')
for i, v in enumerate([df['Class'].value_counts()[0], df['Class'].value_counts()[1]]):
    axes[0].text(i, v + 100, f'{v:,}', ha='center', fontweight='bold')

# Transaction amount distribution
axes[1].hist(df[df['Class']==0]['Amount'], bins=50, alpha=0.7,
             color='#1D9E75', label='Legitimate')
axes[1].hist(df[df['Class']==1]['Amount'], bins=50, alpha=0.7,
             color='#D85A30', label='Fraud')
axes[1].set_title('Transaction Amount Distribution')
axes[1].set_xlabel('Amount')
axes[1].set_ylabel('Frequency')
axes[1].legend()

# Correlation heatmap (top features)
top_features = ['V1','V2','V3','V4','V10','V11','V12','V14','V17','Amount','Class']
corr = df[top_features].corr()
sns.heatmap(corr, ax=axes[2], cmap='coolwarm', center=0,
            annot=False, linewidths=0.5)
axes[2].set_title('Feature Correlation Heatmap')

plt.tight_layout()
plt.savefig('eda_plots.png', dpi=150, bbox_inches='tight')
plt.show()
print("      EDA plots saved as 'eda_plots.png'")

# ============================================================
# STEP 3 - PREPROCESSING
# ============================================================
print("\n[3/6] Preprocessing data...")

# Feature scaling
scaler = StandardScaler()
df['Amount'] = scaler.fit_transform(df[['Amount']])
df['Time'] = scaler.fit_transform(df[['Time']])

X = df.drop('Class', axis=1)
y = df['Class']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"      Training set: {X_train.shape[0]:,} samples")
print(f"      Test set: {X_test.shape[0]:,} samples")

# Apply SMOTE to handle class imbalance
print("      Applying SMOTE for class balancing...")
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
print(f"      After SMOTE - Fraud: {y_train_sm.sum():,} | Legitimate: {(y_train_sm==0).sum():,}")

# ============================================================
# STEP 4 - TRAIN ALL 5 MODELS
# ============================================================
print("\n[4/6] Training 5 ML models (this may take a few minutes)...")

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree':       DecisionTreeClassifier(max_depth=10, random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'XGBoost':             XGBClassifier(n_estimators=100, random_state=42,
                                         eval_metric='logloss', verbosity=0),
    # 'SVM':                 SVC(kernel='rbf', probability=True, random_state=42)
}

results = {}
trained_models = {}

for name, model in models.items():
    print(f"      Training {name}...", end=' ')
    model.fit(X_train_sm, y_train_sm)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    results[name] = {
        'Accuracy':  round(accuracy_score(y_test, y_pred) * 100, 2),
        'Precision': round(precision_score(y_test, y_pred) * 100, 2),
        'Recall':    round(recall_score(y_test, y_pred) * 100, 2),
        'F1-Score':  round(f1_score(y_test, y_pred) * 100, 2),
        'ROC-AUC':   round(roc_auc_score(y_test, y_prob) * 100, 2),
    }
    trained_models[name] = (model, y_pred, y_prob)
    print(f"Done! F1={results[name]['F1-Score']}%")

# ============================================================
# STEP 5 - RESULTS & COMPARISON
# ============================================================
print("\n[5/6] Generating results and comparison plots...")

results_df = pd.DataFrame(results).T
print("\n" + "="*55)
print("  MODEL COMPARISON RESULTS")
print("="*55)
print(results_df.to_string())
print("="*55)

best_model_name = results_df['F1-Score'].idxmax()
print(f"\n  BEST MODEL: {best_model_name}")
print(f"  F1-Score:   {results_df.loc[best_model_name, 'F1-Score']}%")
print(f"  ROC-AUC:    {results_df.loc[best_model_name, 'ROC-AUC']}%")

# Plot 1 — Model comparison bar chart
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Model Comparison — Credit Card Fraud Detection', fontsize=14, fontweight='bold')

metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
x = np.arange(len(metrics))
width = 0.15
colors = ['#378ADD', '#1D9E75', '#BA7517', '#7F77DD', '#D85A30']

for i, (name, res) in enumerate(results.items()):
    vals = [res[m] for m in metrics]
    bars = axes[0].bar(x + i * width, vals, width, label=name, color=colors[i], alpha=0.85)

axes[0].set_xlabel('Metrics')
axes[0].set_ylabel('Score (%)')
axes[0].set_title('All Metrics Comparison')
axes[0].set_xticks(x + width * 2)
axes[0].set_xticklabels(metrics, rotation=10)
axes[0].legend(fontsize=8)
axes[0].set_ylim(60, 105)

# Plot 2 — ROC curves
for name, (model, y_pred, y_prob) in trained_models.items():
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = results[name]['ROC-AUC']
    axes[1].plot(fpr, tpr, label=f'{name} (AUC={auc}%)', linewidth=2)

axes[1].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].set_title('ROC-AUC Curves Comparison')
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("      Comparison chart saved as 'model_comparison.png'")

# Plot 3 — Confusion matrix for best model
best_model_obj, best_pred, _ = trained_models[best_model_name]
cm = confusion_matrix(y_test, best_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Legitimate', 'Fraud'],
            yticklabels=['Legitimate', 'Fraud'])
plt.title(f'Confusion Matrix — {best_model_name} (Best Model)', fontweight='bold')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()
print("      Confusion matrix saved as 'confusion_matrix.png'")

# ============================================================
# STEP 6 - SAVE BEST MODEL
# ============================================================
print(f"\n[6/6] Saving best model ({best_model_name})...")

with open('model.pkl', 'wb') as f:
    pickle.dump(best_model_obj, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("      model.pkl saved successfully!")
print("      scaler.pkl saved successfully!")

print("\n" + "="*55)
print("  ALL DONE! Project files generated:")
print("    model.pkl         ← trained model for web app")
print("    scaler.pkl        ← scaler for web app")
print("    eda_plots.png     ← for report")
print("    model_comparison.png ← for report")
print("    confusion_matrix.png ← for report")
print("="*55)
print(f"\n  Next step: Run the web app with:")
print(f"  streamlit run app.py\n")
