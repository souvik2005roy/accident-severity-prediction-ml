import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

import joblib


# ============================================================
# 1. LOAD PROCESSED DATA
# ============================================================

X_train = pd.read_csv("data/processed/X_train.csv")
X_test = pd.read_csv("data/processed/X_test.csv")

y_train = pd.read_csv("data/processed/y_train.csv").squeeze()
y_test = pd.read_csv("data/processed/y_test.csv").squeeze()


print("=" * 60)
print("FINAL ACCIDENT SEVERITY PREDICTION MODEL")
print("=" * 60)

print("X_train:", X_train.shape)
print("X_test :", X_test.shape)


# ============================================================
# 2. CREATE FINAL RANDOM FOREST MODEL
# ============================================================

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=2,
    min_samples_leaf=1,
    class_weight=None,
    random_state=42,
    n_jobs=-1
)


# ============================================================
# 3. TRAIN MODEL
# ============================================================

print("\nTraining final Random Forest...")

model.fit(
    X_train,
    y_train
)

print("Training completed.")


# ============================================================
# 4. MAKE PREDICTIONS
# ============================================================

y_pred = model.predict(X_test)


# ============================================================
# 5. MODEL EVALUATION
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n" + "=" * 60)
print("FINAL MODEL RESULTS")
print("=" * 60)

print("\nAccuracy:")
print(f"{accuracy:.4f}")


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# ============================================================
# 6. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print(cm)


# ============================================================
# 7. CONFUSION MATRIX VISUALIZATION
# ============================================================

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "Severity 1",
        "Severity 2",
        "Severity 3",
        "Severity 4"
    ]
)

fig, ax = plt.subplots(figsize=(8, 6))

disp.plot(
    ax=ax,
    values_format="d"
)

plt.title("Final Random Forest - Confusion Matrix")

plt.tight_layout()

plt.savefig(
    "outputs/figures/final_confusion_matrix.png",
    dpi=300
)

plt.show()


# ============================================================
# 8. FEATURE IMPORTANCE
# ============================================================

feature_importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)


print("\n" + "=" * 60)
print("FINAL MODEL - FEATURE IMPORTANCE")
print("=" * 60)

print(
    feature_importance.to_string(
        index=False
    )
)


# ============================================================
# 9. FEATURE IMPORTANCE VISUALIZATION
# ============================================================

plt.figure(figsize=(10, 7))

plt.barh(
    feature_importance["Feature"],
    feature_importance["Importance"]
)

plt.xlabel("Importance")
plt.ylabel("Feature")

plt.title(
    "Final Random Forest - Feature Importance"
)

plt.gca().invert_yaxis()

plt.tight_layout()

plt.savefig(
    "outputs/figures/final_feature_importance.png",
    dpi=300
)

plt.show()


# ============================================================
# 10. SAVE FEATURE IMPORTANCE
# ============================================================

feature_importance.to_csv(
    "outputs/final_feature_importance.csv",
    index=False
)


# ============================================================
# 11. SAVE FINAL MODEL
# ============================================================

joblib.dump(
    model,
    "outputs/final_accident_severity_model.pkl"
)

print("\n" + "=" * 60)
print("FINAL MODEL SAVED")
print("=" * 60)

print(
    "Model: outputs/final_accident_severity_model.pkl"
)

print(
    "Feature importance: outputs/final_feature_importance.csv"
)

print(
    "Confusion matrix: outputs/figures/final_confusion_matrix.png"
)

print(
    "Feature importance plot: "
    "outputs/figures/final_feature_importance.png"
)

print("\nFinal model pipeline completed successfully.")