import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    VotingClassifier,
    AdaBoostClassifier,
    StackingClassifier
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

import joblib

# ---------------------------
# LOAD DATASET
# ---------------------------
columns = [
    'age', 'sex', 'cp', 'trestbps', 'chol',
    'fbs', 'restecg', 'thalach', 'exang',
    'oldpeak', 'slope', 'ca', 'thal', 'target'
]

df = pd.read_csv("processed.cleveland.data", names=columns)

# Convert target to binary
df['target'] = df['target'].apply(lambda x: 0 if x == 0 else 1)

# Handle missing values
df.replace('?', np.nan, inplace=True)
df = df.apply(pd.to_numeric, errors='coerce')
df.fillna(df.median(), inplace=True)

# ---------------------------
# SPLIT DATA
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
# SCALING
# ---------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------
# MODELS
# ---------------------------
nb = GaussianNB()
dt = DecisionTreeClassifier(random_state=42)
rf = RandomForestClassifier(n_estimators=200, random_state=42)
knn = KNeighborsClassifier(n_neighbors=5)

lr = LogisticRegression(max_iter=2000, solver='lbfgs', random_state=42)

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
# TRAIN + EVALUATE
# ---------------------------
accuracies = {}

for name, model in models.items():

    # choose correct dataset
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
# BEST MODEL
# ---------------------------
best_model_name = max(accuracies, key=accuracies.get)

best_model = models[best_model_name]

print("\nBest Model:", best_model_name)
print("Best Accuracy:", accuracies[best_model_name])

# ---------------------------
# SAVE MODEL + SCALER
# ---------------------------
joblib.dump(best_model, "heart_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\nModel and scaler saved successfully!")

# ---------------------------
# METRICS FUNCTION
# ---------------------------
def get_metrics(model, X, y):
    pred = model.predict(X)
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred),
        "recall": recall_score(y, pred),
        "f1": f1_score(y, pred)
    }

# ---------------------------
# FINAL METRICS (FIXED)
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