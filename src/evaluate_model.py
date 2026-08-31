import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ==================================================
# 1. Load processed data
# ==================================================

X_train = pd.read_csv("data/processed/X_train.csv")
X_test = pd.read_csv("data/processed/X_test.csv")

y_train = pd.read_csv("data/processed/y_train.csv").squeeze()
y_test = pd.read_csv("data/processed/y_test.csv").squeeze()


print("=" * 50)
print("DATASET INFORMATION")
print("=" * 50)

print("X_train shape:", X_train.shape)
print("X_test shape :", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape :", y_test.shape)


# ==================================================
# 2. Check class distribution
# ==================================================

print("\n" + "=" * 50)
print("TRAINING CLASS DISTRIBUTION")
print("=" * 50)

print(y_train.value_counts().sort_index())


print("\n" + "=" * 50)
print("TEST CLASS DISTRIBUTION")
print("=" * 50)

print(y_test.value_counts().sort_index())


# ==================================================
# 3. Baseline Random Forest
# ==================================================

baseline_rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

baseline_rf.fit(X_train, y_train)

baseline_pred = baseline_rf.predict(X_test)


print("\n" + "=" * 50)
print("BASELINE RANDOM FOREST")
print("=" * 50)

print(
    "Accuracy:",
    accuracy_score(y_test, baseline_pred)
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        baseline_pred,
        zero_division=0
    )
)


# ==================================================
# 4. Balanced Random Forest
# ==================================================

balanced_rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)

balanced_rf.fit(X_train, y_train)

balanced_pred = balanced_rf.predict(X_test)


print("\n" + "=" * 50)
print("BALANCED RANDOM FOREST")
print("=" * 50)

print(
    "Accuracy:",
    accuracy_score(y_test, balanced_pred)
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        balanced_pred,
        zero_division=0
    )
)


# ==================================================
# 5. Confusion Matrix
# ==================================================

print("\n" + "=" * 50)
print("BALANCED RANDOM FOREST - CONFUSION MATRIX")
print("=" * 50)

print(
    confusion_matrix(
        y_test,
        balanced_pred
    )
)