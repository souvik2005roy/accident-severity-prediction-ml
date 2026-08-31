import pandas as pd

from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ==================================================
# 1. Load data
# ==================================================

X_train = pd.read_csv("data/processed/X_train.csv")
X_test = pd.read_csv("data/processed/X_test.csv")

y_train = pd.read_csv("data/processed/y_train.csv").squeeze()
y_test = pd.read_csv("data/processed/y_test.csv").squeeze()


print("=" * 60)
print("TUNED XGBOOST")
print("=" * 60)

print("X_train:", X_train.shape)
print("X_test :", X_test.shape)


# ==================================================
# 2. Convert labels 1-4 → 0-3
# ==================================================

y_train_xgb = y_train - 1
y_test_xgb = y_test - 1


# ==================================================
# 3. Base XGBoost model
# ==================================================

xgb = XGBClassifier(
    objective="multi:softmax",
    num_class=4,
    eval_metric="mlogloss",
    random_state=42
)


# ==================================================
# 4. Parameters to test
# ==================================================

param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.03, 0.05, 0.1],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0]
}


# ==================================================
# 5. Grid Search
# ==================================================

grid_search = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    scoring="accuracy",
    cv=5,
    n_jobs=-1,
    verbose=1
)


print("\nStarting XGBoost Grid Search...")
print("This may take some time.")


grid_search.fit(
    X_train,
    y_train_xgb
)


# ==================================================
# 6. Best parameters
# ==================================================

print("\n" + "=" * 60)
print("BEST PARAMETERS")
print("=" * 60)

print(grid_search.best_params_)

print("\nBest Cross-Validation Accuracy:")
print(grid_search.best_score_)


# ==================================================
# 7. Best model prediction
# ==================================================

best_xgb = grid_search.best_estimator_

y_pred_xgb = best_xgb.predict(X_test)

# Convert 0-3 back to 1-4
y_pred = y_pred_xgb + 1


# ==================================================
# 8. Evaluation
# ==================================================

print("\n" + "=" * 60)
print("TUNED XGBOOST - TEST RESULTS")
print("=" * 60)

accuracy = accuracy_score(
    y_test,
    y_pred
)

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
# 9. Confusion Matrix
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