"""
Breast Cancer Classification: compares four ML models and tunes an SVM
with GridSearchCV to classify tumors as malignant or benign.
"""

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def load_data():
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="diagnosis")
    return X, y, data.target_names


def build_baseline_models():
    return {
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000)
        ),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "SVM": make_pipeline(StandardScaler(), SVC()),
        "K-NN": make_pipeline(StandardScaler(), KNeighborsClassifier()),
    }


def main():
    X, y, class_names = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 1. Baseline comparison
    print("=== Baseline models ===")
    baseline_scores = {}
    for name, model in build_baseline_models().items():
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        baseline_scores[name] = acc
        print(f"{name:<22} accuracy: {acc:.3f}")

    # 2. Hyperparameter tuning for SVM
    print("\n=== Tuning SVM with GridSearchCV ===")
    param_grid = {
        "svc__kernel": ["linear", "rbf"],
        "svc__C": [0.1, 1, 10, 100],
        "svc__gamma": ["scale", 0.01, 0.001],
    }
    svm_pipe = make_pipeline(StandardScaler(), SVC())
    grid = GridSearchCV(svm_pipe, param_grid, cv=5, scoring="f1", n_jobs=-1)
    grid.fit(X_train, y_train)

    print(f"Best parameters: {grid.best_params_}")
    best_svm = grid.best_estimator_
    y_pred = best_svm.predict(X_test)
    tuned_acc = accuracy_score(y_test, y_pred)
    print(f"Tuned SVM accuracy: {tuned_acc:.3f}")

    # 3. Evaluation
    print("\nConfusion matrix (rows = actual, columns = predicted):")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    # 4. Improvement over the best baseline
    best_baseline = max(baseline_scores.values())
    print(f"Improvement over best baseline: {(tuned_acc - best_baseline) * 100:.1f} points")


if __name__ == "__main__":
    main()
