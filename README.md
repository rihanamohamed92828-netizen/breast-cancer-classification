# Breast Cancer Classification Model

Binary classification of breast tumors as **malignant** or **benign** using the Breast Cancer Wisconsin (Diagnostic) dataset from scikit-learn.

## Goal

Compare several machine learning classifiers and tune the best one to minimize false negatives, since missing a malignant tumor is the costliest error in diagnosis.

## Dataset

- Breast Cancer Wisconsin (Diagnostic), built into `sklearn.datasets.load_breast_cancer`
- 569 samples, 30 numeric features computed from cell nucleus images
- Target: malignant (0) or benign (1)

## Models Compared

1. Logistic Regression
2. Decision Tree
3. Support Vector Machine (SVM)
4. K-Nearest Neighbors (K-NN)

## Method

1. Split data into training and test sets, with feature scaling.
2. Train each baseline model and compare results.
3. Tune the SVM with `GridSearchCV` over the kernel, `C`, and `gamma`.
4. Evaluate with a confusion matrix, precision, recall, and F1-score.

## Results

- Baseline accuracy on the test set: Logistic Regression 98.2%, SVM 98.2%, K-NN 95.6%, Decision Tree 91.2%.
- Tuned SVM (RBF kernel, C=10, gamma=0.01) reached **98.2% accuracy** with 1 false negative and 1 false positive on 114 test samples.
- Recall and F1-score were prioritized to reduce false negatives.

## Project Structure

```
breast-cancer-classification/
├── breast_cancer_classification.py
├── requirements.txt
└── README.md
```

## How to Run

```bash
pip install -r requirements.txt
python breast_cancer_classification.py
```

## Tech Stack

Python, scikit-learn, pandas, NumPy, Matplotlib
