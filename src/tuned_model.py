import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
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


print("=" * 60)
print("HYPERPARAMETER TUNING - RANDOM FOREST")
print("=" * 60)

print("X_train:", X_train.shape)
print("X_test :", X_test.shape)


# ==================================================
# 2. Define Random Forest
# ==================================================

rf = RandomForestClassifier(
    random_state=42
)


# ==================================================
# 3. Define parameters to test
# ==================================================

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
    "class_weight": [None, "balanced"]
}


# ==================================================
# 4. Grid Search
# ==================================================

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    scoring="accuracy",
    cv=5,
    n_jobs=-1,
    verbose=1
)


print("\nStarting Grid Search...")
print("This may take some time.")


grid_search.fit(X_train, y_train)


# ==================================================
# 5. Best parameters
# ==================================================

print("\n" + "=" * 60)
print("BEST PARAMETERS")
print("=" * 60)

print(grid_search.best_params_)


print("\nBest Cross-Validation Macro F1:")
print(grid_search.best_score_)


# ==================================================
# 6. Best model
# ==================================================

best_rf = grid_search.best_estimator_

y_pred = best_rf.predict(X_test)


# ==================================================
# 7. Test Set Evaluation
# ==================================================

print("\n" + "=" * 60)
print("TUNED RANDOM FOREST - TEST RESULTS")
print("=" * 60)


accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# ==================================================
# 8. Confusion Matrix
# ==================================================

print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)
# ==================================================
# 9. Feature Importance
# ==================================================

feature_importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": best_rf.feature_importances_
})

# Sort from most important to least important
feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n" + "=" * 60)
print("FEATURE IMPORTANCE")
print("=" * 60)

print(feature_importance.to_string(index=False))


# ==================================================
# 10. Feature Importance Visualization
# ==================================================

plt.figure(figsize=(10, 7))

plt.barh(
    feature_importance["Feature"],
    feature_importance["Importance"]
)

plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("Random Forest Feature Importance")

# Put most important feature at the top
plt.gca().invert_yaxis()

plt.tight_layout()

plt.savefig(
    "outputs/figures/feature_importance.png",
    dpi=300
)

plt.show()


print("\nFeature importance visualization saved successfully.")