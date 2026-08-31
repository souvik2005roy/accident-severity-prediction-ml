import pandas as pd
import joblib


# ============================================================
# 1. LOAD SAVED MODEL
# ============================================================

model = joblib.load(
    "outputs/final_accident_severity_model.pkl"
)

print("=" * 60)
print("ACCIDENT SEVERITY PREDICTION")
print("=" * 60)


# ============================================================
# 2. LOAD ORIGINAL DATASET
# ============================================================
# We use the original dataset to reproduce the same
# categorical encoding used during model training.

data = pd.read_csv(
    "data/ETP_4_New_Data_Accidents.csv"
)


# ============================================================
# 3. GET MODEL FEATURES
# ============================================================

model_features = model.feature_names_in_


# ============================================================
# 4. ASK USER FOR ACCIDENT INFORMATION
# ============================================================

print("\nEnter accident details.")
print("----------------------------------------")


date_input = input(
    "Date (DD/MM/YYYY): "
)

time_input = input(
    "Time of accident (HH:MM): "
)


# Show available values for categorical variables
print("\nAvailable Accident Locations:")
print(
    data["Accident_Location_A"].dropna()
    .unique()
)

location = input(
    "\nAccident Location: "
)


print("\nAvailable Road Conditions:")
print(
    data["Road_Condition_F"].dropna()
    .unique()
)

road_condition = input(
    "\nRoad Condition: "
)


print("\nAvailable Road Features:")
print(
    data["Road_Feature_E"].dropna()
    .unique()
)

road_feature = input(
    "\nRoad Feature: "
)


print("\nAvailable Causes:")
print(
    data["Causes_D"].dropna()
    .unique()
)

cause = input(
    "\nCause of Accident: "
)


print("\nAvailable Weather Conditions:")
print(
    data["Weather_Conditions_H"].dropna()
    .unique()
)

weather = input(
    "\nWeather Condition: "
)


print("\nAvailable Vehicle Type 1:")
print(
    data["Vehicle_Type_Involved_J_V1"].dropna()
    .unique()
)

vehicle_1 = input(
    "\nVehicle Type 1: "
)


print("\nAvailable Vehicle Type 2:")
print(
    data["Vehicle_Type_Involved_J_V2"].dropna()
    .unique()
)

vehicle_2 = input(
    "\nVehicle Type 2: "
)


chainage = float(
    input(
        "\nAccident Location Chainage (km): "
    )
)


chainage_roadside = float(
    input(
        "Accident Location Chainage RoadSide: "
    )
)


# ============================================================
# 5. CREATE INPUT DATAFRAME
# ============================================================

new_data = pd.DataFrame({
    "Date": [date_input],
    "Day_of_Week": [None],
    "Time_of_Accident": [time_input],
    "Accident_Location_A": [location],
    "Accident_Location_A_Chainage_km": [chainage],
    "Accident_Location_A_Chainage_km_RoadSide": [
        chainage_roadside
    ],
    "Road_Condition_F": [road_condition],
    "Road_Feature_E": [road_feature],
    "Causes_D": [cause],
    "Weather_Conditions_H": [weather],
    "Vehicle_Type_Involved_J_V1": [vehicle_1],
    "Vehicle_Type_Involved_J_V2": [vehicle_2]
})


# ============================================================
# 6. DATE PROCESSING
# ============================================================

new_data["Date"] = pd.to_datetime(
    new_data["Date"],
    errors="coerce",
    dayfirst=True
)

new_data["Year"] = new_data["Date"].dt.year
new_data["Month"] = new_data["Date"].dt.month
new_data["Day"] = new_data["Date"].dt.day


# ============================================================
# 7. DAY OF WEEK
# ============================================================

new_data["Day_of_Week"] = (
    new_data["Date"].dt.dayofweek + 1
)


# ============================================================
# 8. TIME PROCESSING
# ============================================================

new_data["Time_of_Accident"] = pd.to_datetime(
    new_data["Time_of_Accident"],
    format="%H:%M",
    errors="coerce"
)

new_data["Hour"] = (
    new_data["Time_of_Accident"].dt.hour
)


# ============================================================
# 9. REMOVE ORIGINAL DATE AND TIME
# ============================================================

new_data = new_data.drop(
    columns=[
        "Date",
        "Time_of_Accident"
    ]
)


# ============================================================
# 10. ONE-HOT ENCODING
# ============================================================

new_data = pd.get_dummies(
    new_data,
    drop_first=True
)


# ============================================================
# 11. ALIGN WITH TRAINING FEATURES
# ============================================================
# This is VERY important.
#
# The model expects exactly the same 14 columns that
# existed during training.

new_data = new_data.reindex(
    columns=model_features,
    fill_value=0
)


# ============================================================
# 12. HANDLE MISSING VALUES
# ============================================================

new_data = new_data.fillna(0)


# ============================================================
# 13. MAKE PREDICTION
# ============================================================

prediction = model.predict(
    new_data
)[0]


# ============================================================
# 14. DISPLAY RESULT
# ============================================================

severity_names = {
    1: "Severity 1",
    2: "Severity 2",
    3: "Severity 3",
    4: "Severity 4"
}


print("\n" + "=" * 60)
print("PREDICTION RESULT")
print("=" * 60)

print(
    "\nPredicted Accident Severity:",
    prediction
)

print(
    "Severity Category:",
    severity_names.get(
        prediction,
        "Unknown"
    )
)


# ============================================================
# 15. PREDICTION PROBABILITIES
# ============================================================

probabilities = model.predict_proba(
    new_data
)[0]


print("\nPrediction Probabilities:")

for severity, probability in zip(
    model.classes_,
    probabilities
):

    print(
        f"Severity {severity}: "
        f"{probability * 100:.2f}%"
    )


print("\nPrediction completed successfully.")