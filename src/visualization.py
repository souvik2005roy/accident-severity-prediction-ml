import pandas as pd
import matplotlib.pyplot as plt
import os


# Load the accident data
data = pd.read_csv("data\\ETP_4_New_Data_Accidents.csv")

# Create output folder if it does not exist
os.makedirs("outputs/figures", exist_ok=True)


# --------------------------------------------------
# 1. Accident Severity Distribution
# --------------------------------------------------

severity_labels = {
    1: "Fatal",
    2: "Grievous Injury",
    3: "Minor Injury",
    4: "No Injury"
}

severity_count = (
    data["Accident_Severity_C"]
    .value_counts()
    .sort_index()
)

plt.figure(figsize=(9, 5))

severity_count.plot(
    kind="bar"
)

plt.title("Accident Severity Distribution")
plt.xlabel("Accident Severity")
plt.ylabel("Number of Accidents")

# Replace numerical codes with actual severity names
plt.xticks(
    ticks=range(len(severity_count)),
    labels=[severity_labels[i] for i in severity_count.index],
    rotation=0
)

plt.tight_layout()

plt.savefig(
    "outputs/figures/accident_severity.png",
    dpi=300
)

plt.show()


# --------------------------------------------------
# 2. Accidents by Day of Week
# --------------------------------------------------

day_count = data["Day_of_Week"].value_counts().sort_index()

plt.figure(figsize=(7, 5))

day_count.plot(
    kind="bar"
)

plt.title("Accidents by Day of Week")
plt.xlabel("Day of Week")
plt.ylabel("Number of Accidents")

plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig(
    "outputs/figures/accidents_by_day.png",
    dpi=300
)

plt.show()


# --------------------------------------------------
# 3. Accidents by Hour
# --------------------------------------------------

time_data = pd.to_datetime(
    data["Time_of_Accident"],
    format="%H:%M",
    errors="coerce"
)

data["Hour"] = time_data.dt.hour

hour_count = data["Hour"].value_counts().sort_index()

plt.figure(figsize=(9, 5))

hour_count.plot(
    kind="line",
    marker="o"
)

plt.title("Accidents by Hour of Day")
plt.xlabel("Hour of Day")
plt.ylabel("Number of Accidents")

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "outputs/figures/accidents_by_hour.png",
    dpi=300
)

plt.show()


# --------------------------------------------------
# 4. Accidents by Weather Condition
# --------------------------------------------------

# Weather condition code mapping
weather_labels = {
    1: "Fine",
    2: "Mist / Fog",
    3: "Cloud",
    4: "Light Rain",
    5: "Heavy Rain",
    6: "Hail / Sleet",
    7: "Snow",
    8: "Strong Wind",
    9: "Dust Storm",
    10: "Very Hot",
    11: "Very Cold",
    12: "Other"
}

weather_count = (
    data["Weather_Conditions_H"]
    .value_counts()
    .sort_index()
)

plt.figure(figsize=(10, 6))

weather_count.plot(
    kind="bar"
)

plt.title("Accidents by Weather Condition")
plt.xlabel("Weather Condition")
plt.ylabel("Number of Accidents")

# Replace numerical codes with actual weather names
plt.xticks(
    ticks=range(len(weather_count)),
    labels=[weather_labels[i] for i in weather_count.index],
    rotation=45,
    ha="right"
)

plt.tight_layout()

plt.savefig(
    "outputs/figures/accidents_by_weather.png",
    dpi=300
)

plt.show()