import os
import sys
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve, classification_report
import matplotlib.pyplot as plt
import joblib
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, '..', 'data', 'churn.csv')
sys.path.append(os.path.join(BASE_DIR, '..', 'src'))

from preprocess import load_and_preprocess_data

X_train, X_test, y_train, y_test, preprocessor = load_and_preprocess_data(csv_path)

models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

results = {}
metrics = {}

plt.figure(figsize=(8, 6))

for name, clf in models.items():
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', clf)
    ])

    model.fit(X_train, y_train)
    print(f"\n Model {name} trained successfully.")

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print(f"\n Classification Report for {name}:\n{classification_report(y_test, y_pred)}")
    acc = round(accuracy_score(y_test, y_pred), 3)
    auc = round(roc_auc_score(y_test, y_prob), 3)

    print(f"{name} — Accuracy: {acc}, ROC AUC: {auc}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc})")

    results[name] = {
        "pipeline": model,
        "auc": auc,
        "acc": acc
    }

    metrics[name] = {
        "accuracy": acc,
        "roc_auc": auc,
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
    }

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Different Models')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, '..', 'plots', 'all_models_roc.png'))
plt.show()

# Сохраняем лучший пайплайн
best_model_name = max(results, key=lambda name: results[name]['auc'])
print(f"\n Best model: {best_model_name} with AUC: {results[best_model_name]['auc']}")

best_model = results[best_model_name]['pipeline']
pkl_path = os.path.join(BASE_DIR, '..', 'models', 'churn_model.pkl')
joblib.dump(best_model, pkl_path)
print(f" Best model saved to {pkl_path}")

# Сохраняем метрики в JSON
metrics_path = os.path.join(BASE_DIR, '..', 'models', 'metrics.json')
with open(metrics_path, 'w') as f:
    json.dump(metrics, f, indent=4)
print(f" Metrics saved to {metrics_path}")