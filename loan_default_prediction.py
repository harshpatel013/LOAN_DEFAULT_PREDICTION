import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    classification_report, roc_auc_score, roc_curve
)


sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.dpi'] = 100

print("=" * 55)
print("   LOAN DEFAULT PREDICTION - ML PROJECT")
print("=" * 55)

# Load the dataset
df = pd.read_csv("credit_risk_dataset.csv")

print("\n[Step 1] Dataset Loaded Successfully!")
print(f"   Rows    : {df.shape[0]}")
print(f"   Columns : {df.shape[1]}")
print("\nFirst 5 rows:")
print(df.head())

print("\n" + "-" * 55)
print("[Step 2] Exploratory Data Analysis (EDA)")
print("-" * 55)

# Inspect the data and its distribution
print("\nDataset Info:")
print(df.dtypes)

print("\nBasic Statistics:")
print(df.describe())

print("\nMissing Values per Column:")
print(df.isnull().sum())

print(f"\nDefault Distribution:")
vc = df['Default'].value_counts()
print(f"   No Default (0) : {vc[0]}  ({vc[0]/len(df)*100:.1f}%)")
print(f"   Default    (1) : {vc[1]}  ({vc[1]/len(df)*100:.1f}%)")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Loan Default Prediction – EDA Visualisations", fontsize=15, fontweight='bold')

axes[0, 0].bar(['No Default', 'Default'], df['Default'].value_counts().values,
               color=['steelblue', 'tomato'], edgecolor='black', width=0.5)
axes[0, 0].set_title("Class Distribution")
axes[0, 0].set_ylabel("Count")
for i, v in enumerate(df['Default'].value_counts().values):
    axes[0, 0].text(i, v + 5, str(v), ha='center', fontweight='bold')

df[df['Default'] == 0]['Age'].hist(ax=axes[0, 1], bins=20,
                                   color='steelblue', alpha=0.7, label='No Default')
df[df['Default'] == 1]['Age'].hist(ax=axes[0, 1], bins=20,
                                   color='tomato', alpha=0.7, label='Default')
axes[0, 1].set_title("Age Distribution by Default")
axes[0, 1].set_xlabel("Age")
axes[0, 1].legend()

df.boxplot(column='Credit_Score', by='Default', ax=axes[0, 2],
           patch_artist=True,
           boxprops=dict(facecolor='lightblue'),
           medianprops=dict(color='red', linewidth=2))
axes[0, 2].set_title("Credit Score by Default Status")
axes[0, 2].set_xlabel("Default (0 = No, 1 = Yes)")
plt.sca(axes[0, 2])
plt.title("Credit Score by Default Status")

colors = df['Default'].map({0: 'steelblue', 1: 'tomato'})
axes[1, 0].scatter(df['Income'], df['Loan_Amount'],
                   c=colors, alpha=0.4, s=20)
axes[1, 0].set_title("Income vs Loan Amount")
axes[1, 0].set_xlabel("Income")
axes[1, 0].set_ylabel("Loan Amount")

edu_default = df.groupby('Education_Level')['Default'].mean() * 100
edu_default.sort_values().plot(kind='bar', ax=axes[1, 1],
                               color='mediumpurple', edgecolor='black')
axes[1, 1].set_title("Default Rate by Education Level")
axes[1, 1].set_ylabel("Default Rate (%)")
axes[1, 1].tick_params(axis='x', rotation=20)

num_cols = df.select_dtypes(include='number').columns
corr = df[num_cols].corr()
sns.heatmap(corr, ax=axes[1, 2], annot=True, fmt=".2f",
            cmap='coolwarm', linewidths=0.5)
axes[1, 2].set_title("Correlation Heatmap")

plt.tight_layout()
plt.savefig("eda_visualisations.png", bbox_inches='tight')
plt.close()
print("\n   [Saved] eda_visualisations.png")

# Prepare the data for modeling
print("\n" + "-" * 55)
print("[Step 3] Data Preprocessing")
print("-" * 55)

median_income = df['Income'].median()
df['Income'] = df['Income'].fillna(median_income)
print(f"   Filled 15 missing Income values with median = {median_income:,.0f}")

le = LabelEncoder()
df['Education_Level_enc'] = le.fit_transform(df['Education_Level'])
df['Housing_Status_enc']  = le.fit_transform(df['Housing_Status'])
print("   Label-encoded: Education_Level, Housing_Status")

df['Debt_to_Income'] = df['Loan_Amount'] / df['Income']
print("   Created new feature: Debt_to_Income = Loan_Amount / Income")

feature_cols = ['Age', 'Income', 'Loan_Amount', 'Credit_Score',
                'Employment_Years', 'Education_Level_enc',
                'Housing_Status_enc', 'Debt_to_Income']
X = df[feature_cols]
y = df['Default']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print(f"\n   Train size : {X_train.shape[0]} samples")
print(f"   Test  size : {X_test.shape[0]} samples")

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)
print("   Applied StandardScaler for Logistic Regression")

print("\n" + "-" * 55)
print("[Step 4] Model Training")
print("-" * 55)

# Train multiple classifiers
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_sc, y_train)
print("   ✓ Logistic Regression trained")

dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_model.fit(X_train, y_train)
print("   ✓ Decision Tree trained  (max_depth=5)")

rf_model = RandomForestClassifier(n_estimators=100, max_depth=7,
                                  random_state=42)
rf_model.fit(X_train, y_train)
print("   ✓ Random Forest trained  (100 trees, max_depth=7)")

print("\n" + "-" * 55)
print("[Step 5] Model Evaluation")
print("-" * 55)

# Evaluate model performance
def evaluate(name, model, X_t, y_t):
    y_pred = model.predict(X_t)
    y_prob = model.predict_proba(X_t)[:, 1]
    acc  = accuracy_score(y_t, y_pred)
    auc  = roc_auc_score(y_t, y_prob)
    cm   = confusion_matrix(y_t, y_pred)
    print(f"\n  ── {name} ──")
    print(f"   Accuracy : {acc*100:.2f}%")
    print(f"   ROC-AUC  : {auc:.4f}")
    print(f"\n   Confusion Matrix:\n{cm}")
    print(f"\n   Classification Report:\n{classification_report(y_t, y_pred, target_names=['No Default','Default'])}")
    return acc, auc, y_prob

lr_acc, lr_auc, lr_prob = evaluate("Logistic Regression", lr_model, X_test_sc, y_test)
dt_acc, dt_auc, dt_prob = evaluate("Decision Tree",       dt_model,  X_test,    y_test)
rf_acc, rf_auc, rf_prob = evaluate("Random Forest",       rf_model,  X_test,    y_test)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Model Evaluation Visualisations", fontsize=14, fontweight='bold')

models = ['Logistic\nRegression', 'Decision\nTree', 'Random\nForest']
accs   = [lr_acc*100, dt_acc*100, rf_acc*100]
bars   = axes[0].bar(models, accs, color=['steelblue','orange','seagreen'],
                     edgecolor='black', width=0.5)
axes[0].set_ylim(80, 100)
axes[0].set_title("Model Accuracy Comparison")
axes[0].set_ylabel("Accuracy (%)")
for bar, acc in zip(bars, accs):
    axes[0].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.2,
                 f"{acc:.2f}%", ha='center', fontweight='bold', fontsize=10)

for prob, label, color in [
    (lr_prob, f'Logistic Reg (AUC={lr_auc:.3f})', 'steelblue'),
    (dt_prob, f'Decision Tree (AUC={dt_auc:.3f})', 'orange'),
    (rf_prob, f'Random Forest (AUC={rf_auc:.3f})', 'seagreen'),
]:
    fpr, tpr, _ = roc_curve(y_test, prob)
    axes[1].plot(fpr, tpr, label=label, lw=2, color=color)
axes[1].plot([0,1],[0,1], 'k--', lw=1)
axes[1].set_title("ROC Curves")
axes[1].set_xlabel("False Positive Rate")
axes[1].set_ylabel("True Positive Rate")
axes[1].legend(fontsize=8)

importances = rf_model.feature_importances_
feat_df = pd.DataFrame({'Feature': feature_cols, 'Importance': importances})
feat_df = feat_df.sort_values('Importance', ascending=True)
axes[2].barh(feat_df['Feature'], feat_df['Importance'],
             color='seagreen', edgecolor='black')
axes[2].set_title("Feature Importance (Random Forest)")
axes[2].set_xlabel("Importance Score")

plt.tight_layout()
plt.savefig("model_evaluation.png", bbox_inches='tight')
plt.close()
print("\n   [Saved] model_evaluation.png")

# Plot confusion matrices for each model
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("Confusion Matrices", fontsize=13, fontweight='bold')

for ax, model, name, X_t, sc in [
    (axes[0], lr_model, "Logistic Regression", X_test_sc, True),
    (axes[1], dt_model, "Decision Tree",        X_test,    False),
    (axes[2], rf_model, "Random Forest",        X_test,    False),
]:
    cm = confusion_matrix(y_test, model.predict(X_t))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['No Default','Default'],
                yticklabels=['No Default','Default'])
    ax.set_title(name)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

plt.tight_layout()
plt.savefig("confusion_matrices.png", bbox_inches='tight')
plt.close()
print("   [Saved] confusion_matrices.png")

print("\n" + "=" * 55)
print("   PROJECT SUMMARY")
print("=" * 55)
print(f"\n   {'Model':<22} {'Accuracy':>10} {'ROC-AUC':>10}")
print(f"   {'-'*44}")
print(f"   {'Logistic Regression':<22} {lr_acc*100:>9.2f}% {lr_auc:>10.4f}")
print(f"   {'Decision Tree':<22} {dt_acc*100:>9.2f}% {dt_auc:>10.4f}")
print(f"   {'Random Forest':<22} {rf_acc*100:>9.2f}% {rf_auc:>10.4f}")

best = max([('Logistic Regression', lr_acc), ('Decision Tree', dt_acc),
            ('Random Forest', rf_acc)], key=lambda x: x[1])
print(f"\n   Best Model : {best[0]}  ({best[1]*100:.2f}% accuracy)")

print("=" * 55)
