"""Breast cancer classification: Logistic Regression, Decision Tree, SVM, K-NN
tuned with GridSearchCV on the Wisconsin Diagnostic Breast Cancer dataset."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

SEED = 42

data = load_breast_cancer(as_frame=True)
X, y = data.data, data.target  # target: 0 = malignant, 1 = benign
print(f"Samples: {X.shape[0]}, features: {X.shape[1]}")
print(y.map({0: "malignant", 1: "benign"}).value_counts().to_string())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED)

def pipe(model):
    return Pipeline([("scale", StandardScaler()), ("clf", model)])

models = {
    "Logistic Regression": (pipe(LogisticRegression(max_iter=5000)),
                            {"clf__C": [0.01, 0.1, 1, 10, 100]}),
    "Decision Tree": (pipe(DecisionTreeClassifier(random_state=SEED)),
                      {"clf__max_depth": [2, 3, 4, 5, None],
                       "clf__min_samples_split": [2, 5, 10]}),
    "SVM": (pipe(SVC()),
            {"clf__C": [0.1, 1, 10, 100],
             "clf__gamma": ["scale", 0.01, 0.1],
             "clf__kernel": ["linear", "rbf"]}),
    "K-NN": (pipe(KNeighborsClassifier()),
             {"clf__n_neighbors": [3, 5, 7, 9, 11],
              "clf__weights": ["uniform", "distance"]}),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
rows, best = [], {}
for name, (est, grid) in models.items():
    gs = GridSearchCV(est, grid, cv=cv, scoring="accuracy", n_jobs=-1)
    gs.fit(X_train, y_train)
    pred = gs.predict(X_test)
    best[name] = (gs, pred)
    rows.append({
        "Model": name,
        "CV accuracy": round(gs.best_score_, 4),
        "Test accuracy": round(accuracy_score(y_test, pred), 4),
        "Precision": round(precision_score(y_test, pred), 4),
        "Recall": round(recall_score(y_test, pred), 4),
        "F1": round(f1_score(y_test, pred), 4),
        "Best params": {k.replace("clf__", ""): v for k, v in gs.best_params_.items()},
    })

res = pd.DataFrame(rows).sort_values("Test accuracy", ascending=False)
print("\n", res.drop(columns="Best params").to_string(index=False))
res.to_csv("outputs/gridsearch_comparison.csv", index=False)
with open("outputs/gridsearch_best_params.json", "w") as f:
    json.dump({r["Model"]: r["Best params"] for r in rows}, f, indent=2, default=str)

# Accuracy chart
plt.figure(figsize=(7, 4))
sns.barplot(data=res, x="Model", y="Test accuracy", color="#4c72b0")
plt.ylim(0.85, 1.0)
plt.title("Test accuracy by model")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("outputs/gridsearch_accuracy.png", dpi=150)
plt.close()

# SVM detail
svm_gs, svm_pred = best["SVM"]
print("\nSVM classification report")
print(classification_report(y_test, svm_pred, target_names=["malignant", "benign"]))
cm = confusion_matrix(y_test, svm_pred)
plt.figure(figsize=(4.5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["malignant", "benign"], yticklabels=["malignant", "benign"])
plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title("SVM confusion matrix")
plt.tight_layout()
plt.savefig("outputs/svm_confusion_matrix.png", dpi=150)
