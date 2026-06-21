"""Train and evaluate multiple classifiers on the UCI Heart Disease dataset."""

# pylint: disable=invalid-name

import joblib
import numpy as np
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    AdaBoostClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# ---------------------------
# LOAD DATASET — UCI Heart Disease (Cleveland)
# ---------------------------
columns = [
    'age', 'sex', 'cp', 'trestbps', 'chol',
    'fbs', 'restecg', 'thalach', 'exang',
    'oldpeak', 'slope', 'ca', 'thal', 'target'
]

df = pd.read_csv("processed.cleveland.data", names=columns)

# Convert target from multi-class (0-4) to binary (0 = no disease, 1 = disease present)
df['target'] = df['target'].apply(lambda x: 0 if x == 0 else 1)

# Handle missing values: '?' denotes missing, coerce to NaN, then impute with median
df.replace('?', np.nan, inplace=True)
df = df.apply(pd.to_numeric, errors='coerce')
df.fillna(df.median(), inplace=True)

# ---------------------------
# SPLIT DATA — 80/20 train/test with stratified sampling
# ---------------------------
X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ---------------------------
# SCALING — standardize features (fit only on train to avoid data leakage)
# Distance-based models (KNN, SVM, LR, etc.) are sensitive to feature scale
# ---------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------
# MODELS — diverse set covering key families:
# probabilistic (NB), tree-based (DT, RF, AdaBoost),
# distance-based (KNN, SVM), linear (LR), and ensembles (Voting, Stacking)
# ---------------------------
nb = GaussianNB()
dt = DecisionTreeClassifier(random_state=42)
rf = RandomForestClassifier(n_estimators=200, random_state=42)
knn = KNeighborsClassifier(n_neighbors=5)

lr = LogisticRegression(max_iter=2000, solver='lbfgs', random_state=42)

# SVC wrapped in CalibratedClassifierCV to enable predict_proba for soft voting
svm = CalibratedClassifierCV(
    SVC(random_state=42),
    ensemble=False
)

voting = VotingClassifier(
    estimators=[
        ('lr', lr),
        ('rf', rf),
        ('svm', svm),
        ('nb', nb)
    ],
    voting='soft'
)

ada = AdaBoostClassifier(n_estimators=100, random_state=42)

stack = StackingClassifier(
    estimators=[
        ('nb', nb),
        ('dt', dt),
        ('knn', knn),
        ('rf', rf),
        ('svm', svm)
    ],
    final_estimator=LogisticRegression(max_iter=2000)
)

models = {
    "Naive Bayes": nb,
    "Decision Tree": dt,
    "Random Forest": rf,
    "KNN": knn,
    "Logistic Regression": lr,
    "SVM": svm,
    "Voting": voting,
    "AdaBoost": ada,
    "Stacking": stack
}

# ---------------------------
# TRAIN + EVALUATE — each model on its appropriate input format
# Distance-based (KNN, LR, SVM) and ensembles containing them (Voting, Stacking)
# use scaled features; tree-based and naive Bayes use original unscaled features
# ---------------------------
accuracies = {}

for name, model in models.items():

    if name in ["KNN", "Logistic Regression", "SVM", "Voting", "Stacking"]:
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    accuracies[name] = acc
    print(f"{name} Accuracy: {acc:.4f}")

# ---------------------------
# SELECT BEST MODEL — highest accuracy across all nine
# ---------------------------
best_model_name = max(accuracies, key=accuracies.get)

best_model = models[best_model_name]

print("\nBest Model:", best_model_name)
print("Best Accuracy:", accuracies[best_model_name])

# ---------------------------
# SAVE BEST MODEL & SCALER — for reuse in inference without retraining
# ---------------------------
joblib.dump(best_model, "heart_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\nModel and scaler saved successfully!")

# ---------------------------
# METRICS FUNCTION — compute accuracy, precision, recall, F1 from trained model
# ---------------------------
def get_metrics(estimator, features, labels):
    """Return accuracy, precision, recall, F1 for estimator on given data."""
    pred = estimator.predict(features)
    return {
        "accuracy": accuracy_score(labels, pred),
        "precision": precision_score(labels, pred),
        "recall": recall_score(labels, pred),
        "f1": f1_score(labels, pred)
    }

# ---------------------------
# COMPUTE & SAVE METRICS FOR ALL MODELS — each uses the same data variant
# (scaled or unscaled) that was used during training to ensure consistency
# ---------------------------
metrics = {
    "Logistic Regression": get_metrics(lr, X_test_scaled, y_test),
    "SVM": get_metrics(svm, X_test_scaled, y_test),
    "KNN": get_metrics(knn, X_test_scaled, y_test),

    "Naive Bayes": get_metrics(nb, X_test, y_test),
    "Decision Tree": get_metrics(dt, X_test, y_test),
    "Random Forest": get_metrics(rf, X_test, y_test),

    "Voting Classifier": get_metrics(voting, X_test_scaled, y_test),
    "AdaBoost": get_metrics(ada, X_test, y_test),
    "Stacking Classifier": get_metrics(stack, X_test_scaled, y_test)
}

joblib.dump(metrics, "model_metrics.pkl")

print("All model metrics saved successfully!")
