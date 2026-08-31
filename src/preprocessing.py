import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv("data/ETP_4_New_Data_Accidents.csv")

print("Dataset shape:", data.shape)
print("\nFirst 5 rows:")
print(data.head())

# Remove duplicate rows
data = data.drop_duplicates()

# Remove rows where accident severity is missing
data = data.dropna(subset=["Accident_Severity_C"])

# Convert date into datetime
data["Date"] = pd.to_datetime(data["Date"], errors="coerce")

# Create some useful features from date
data["Year"] = data["Date"].dt.year
data["Month"] = data["Date"].dt.month
data["Day"] = data["Date"].dt.day

# Convert accident time into hour
data["Time_of_Accident"] = pd.to_datetime(
    data["Time_of_Accident"],
    format="%H:%M",
    errors="coerce"
)

data["Hour"] = data["Time_of_Accident"].dt.hour

# Drop original date and time columns
data = data.drop(
    columns=["Date", "Time_of_Accident"]
)

# Separate input variables and target variable
X = data.drop(columns=["Accident_Severity_C"])
y = data["Accident_Severity_C"]

# Convert categorical columns into numbers
X = pd.get_dummies(X, drop_first=True)

# Fill missing values
X = X.fillna(X.median(numeric_only=True))
X = X.fillna(0)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Save processed data
X_train.to_csv("data/processed/X_train.csv", index=False)
X_test.to_csv("data/processed/X_test.csv", index=False)
y_train.to_csv("data/processed/y_train.csv", index=False)
y_test.to_csv("data/processed/y_test.csv", index=False)

print("\nPreprocessing completed.")
print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

print("\nAccident severity distribution:")
print(y.value_counts())
