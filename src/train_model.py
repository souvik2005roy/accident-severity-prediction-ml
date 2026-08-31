import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the processed data
X_train = pd.read_csv("data/processed/X_train.csv")
X_test = pd.read_csv("data/processed/X_test.csv")

y_train = pd.read_csv("data/processed/y_train.csv").squeeze()
y_test = pd.read_csv("data/processed/y_test.csv").squeeze()

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)


# --------------------------------------------------
# 1. Logistic Regression
# --------------------------------------------------

logistic_model = LogisticRegression(
    max_iter=1000
)

logistic_model.fit(X_train, y_train)

logistic_prediction = logistic_model.predict(X_test)

logistic_accuracy = accuracy_score(
    y_test,
    logistic_prediction
)

print("\nLogistic Regression Accuracy:",
      round(logistic_accuracy, 4))


# --------------------------------------------------
# 2. Random Forest
# --------------------------------------------------

random_forest = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

random_forest.fit(X_train, y_train)

random_forest_prediction = random_forest.predict(X_test)

random_forest_accuracy = accuracy_score(
    y_test,
    random_forest_prediction
)

print("Random Forest Accuracy:",
      round(random_forest_accuracy, 4))


# --------------------------------------------------
# Compare models
# --------------------------------------------------

print("\nModel Comparison")
print("------------------------")

print(
    "Logistic Regression:",
    round(logistic_accuracy, 4)
)

print(
    "Random Forest:",
    round(random_forest_accuracy, 4)
)


# Find the better model
if random_forest_accuracy > logistic_accuracy:
    print("\nRandom Forest performed better.")
else:
    print("\nLogistic Regression performed better.")
