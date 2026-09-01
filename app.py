import streamlit as st
import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Accident Severity Prediction",
    page_icon="🚗",
    layout="wide"
)


# --------------------------------------------------
# BASE DIRECTORY
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# PATHS
# ============================================================

MODEL_PATH = os.path.join(
    BASE_DIR,
    "outputs",
    "final_accident_severity_model.pkl"
)
DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "ETP_4_New_Data_Accidents.csv"
)
FEATURE_IMPORTANCE_PATH = os.path.join(
    BASE_DIR,
    "outputs",
    "final_feature_importance.csv"
)
CONFUSION_MATRIX_PATH = os.path.join(
    BASE_DIR,
    "outputs",
    "random_forest_confusion_matrix.png"
)

# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = joblib.load(MODEL_PATH)

    return model


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    data = pd.read_csv(DATA_PATH)

    return data


# ============================================================
# LOAD RESOURCES
# ============================================================

model = load_model()
data = load_data()


# ============================================================
# TITLE
# ============================================================

st.title("🚗 Accident Severity Prediction")

st.markdown(
    """
    ### Machine Learning Based Accident Severity Analysis

    This application uses a **Random Forest Classifier** to predict
    accident severity based on accident location, road conditions,
    weather, vehicle information, accident cause and time-related
    features.
    """
)


st.divider()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select a page:",
    [
        "🏠 Dashboard",
        "🔮 Prediction",
        "📊 Model Insights"
    ]
)

# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.header("📊 Dataset Overview")

    # --------------------------------------------------------
    # Dataset metrics
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Accidents",
            f"{len(data):,}"
        )

    with col2:
        st.metric(
            "Features",
            "14"
        )

    with col3:
        st.metric(
            "Training Samples",
            "3,242"
        )

    with col4:
        st.metric(
            "Test Samples",
            "811"
        )

    st.divider()

    # --------------------------------------------------------
    # Accident severity distribution
    # --------------------------------------------------------

    st.subheader("Accident Severity Distribution")

    severity_count = (
        data["Accident_Severity_C"]
        .value_counts()
        .sort_index()
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(
        severity_count.index.astype(str),
        severity_count.values
    )

    ax.set_xlabel("Accident Severity")
    ax.set_ylabel("Number of Accidents")
    ax.set_title("Accident Severity Distribution")

    st.pyplot(fig)

    # --------------------------------------------------------
    # Weather condition distribution
    # --------------------------------------------------------

    st.subheader("Accidents by Weather Condition")

    weather_count = (
        data["Weather_Conditions_H"]
        .value_counts()
        .sort_index()
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(
        weather_count.index.astype(str),
        weather_count.values
    )

    ax.set_xlabel("Weather Condition Code")
    ax.set_ylabel("Number of Accidents")
    ax.set_title("Accidents by Weather Condition")

    st.pyplot(fig)

    st.info(
        "The categorical values shown in the dataset are encoded "
        "as numerical codes. These codes are used directly by the "
        "trained model."
    )


# ============================================================
# PREDICTION PAGE
# ============================================================

elif page == "🔮 Prediction":

    st.header("🔮 Predict Accident Severity")

    st.write(
        "Enter the accident characteristics below and the trained "
        "Random Forest model will estimate the accident severity."
    )

    st.divider()

    # --------------------------------------------------------
    # DATE AND TIME
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        date_input = st.date_input(
            "Date of Accident"
        )

    with col2:

        time_input = st.time_input(
            "Time of Accident"
        )

    # --------------------------------------------------------
    # ACCIDENT LOCATION
    # --------------------------------------------------------

    st.subheader("📍 Location Information")

    col1, col2 = st.columns(2)

    with col1:

        location_values = sorted(
            data["Accident_Location_A"]
            .dropna()
            .unique()
            .tolist()
        )

        location = st.selectbox(
            "Accident Location",
            location_values
        )

    with col2:

        chainage = st.number_input(
            "Accident Location Chainage (km)",
            min_value=0.0,
            value=1.0,
            step=0.1
        )

    chainage_roadside = st.number_input(
        "Accident Location Chainage RoadSide",
        min_value=0.0,
        value=1.0,
        step=0.1
    )

    # --------------------------------------------------------
    # ROAD INFORMATION
    # --------------------------------------------------------

    st.subheader("🛣️ Road Information")

    col1, col2 = st.columns(2)

    with col1:

        road_condition_values = sorted(
            data["Road_Condition_F"]
            .dropna()
            .unique()
            .tolist()
        )

        road_condition = st.selectbox(
            "Road Condition",
            road_condition_values
        )

    with col2:

        road_feature_values = sorted(
            data["Road_Feature_E"]
            .dropna()
            .unique()
            .tolist()
        )

        road_feature = st.selectbox(
            "Road Feature",
            road_feature_values
        )

    # --------------------------------------------------------
    # ACCIDENT CAUSE
    # --------------------------------------------------------

    st.subheader("⚠️ Accident Information")

    col1, col2 = st.columns(2)

    with col1:

        cause_values = sorted(
            data["Causes_D"]
            .dropna()
            .unique()
            .tolist()
        )

        cause = st.selectbox(
            "Cause of Accident",
            cause_values
        )

    with col2:

        weather_values = sorted(
            data["Weather_Conditions_H"]
            .dropna()
            .unique()
            .tolist()
        )

        weather = st.selectbox(
            "Weather Condition",
            weather_values
        )

    # --------------------------------------------------------
    # VEHICLE INFORMATION
    # --------------------------------------------------------

    st.subheader("🚙 Vehicle Information")

    col1, col2 = st.columns(2)

    with col1:

        vehicle_1_values = sorted(
            data["Vehicle_Type_Involved_J_V1"]
            .dropna()
            .unique()
            .tolist()
        )

        vehicle_1 = st.selectbox(
            "Vehicle Type 1",
            vehicle_1_values
        )

    with col2:

        vehicle_2_values = sorted(
            data["Vehicle_Type_Involved_J_V2"]
            .dropna()
            .unique()
            .tolist()
        )

        vehicle_2 = st.selectbox(
            "Vehicle Type 2",
            vehicle_2_values
        )

    st.divider()

    # ========================================================
    # PREDICT BUTTON
    # ========================================================

    if st.button(
        "🚨 Predict Accident Severity",
        type="primary",
        use_container_width=True
    ):

        # ----------------------------------------------------
        # CREATE INPUT DATAFRAME
        # ----------------------------------------------------

        new_data = pd.DataFrame({

            "Date": [date_input],

            "Day_of_Week": [
                date_input.weekday() + 1
            ],

            "Time_of_Accident": [
                time_input.strftime("%H:%M")
            ],

            "Accident_Location_A": [
                location
            ],

            "Accident_Location_A_Chainage_km": [
                chainage
            ],

            "Accident_Location_A_Chainage_km_RoadSide": [
                chainage_roadside
            ],

            "Road_Condition_F": [
                road_condition
            ],

            "Road_Feature_E": [
                road_feature
            ],

            "Causes_D": [
                cause
            ],

            "Weather_Conditions_H": [
                weather
            ],

            "Vehicle_Type_Involved_J_V1": [
                vehicle_1
            ],

            "Vehicle_Type_Involved_J_V2": [
                vehicle_2
            ]
        })


        # ----------------------------------------------------
        # DATE FEATURES
        # ----------------------------------------------------

        new_data["Date"] = pd.to_datetime(
            new_data["Date"]
        )

        new_data["Year"] = (
            new_data["Date"].dt.year
        )

        new_data["Month"] = (
            new_data["Date"].dt.month
        )

        new_data["Day"] = (
            new_data["Date"].dt.day
        )


        # ----------------------------------------------------
        # TIME FEATURE
        # ----------------------------------------------------

        new_data["Time_of_Accident"] = pd.to_datetime(
            new_data["Time_of_Accident"],
            format="%H:%M"
        )

        new_data["Hour"] = (
            new_data["Time_of_Accident"].dt.hour
        )


        # ----------------------------------------------------
        # REMOVE ORIGINAL DATE AND TIME
        # ----------------------------------------------------

        new_data = new_data.drop(
            columns=[
                "Date",
                "Time_of_Accident"
            ]
        )


        # ----------------------------------------------------
        # ONE-HOT ENCODING
        # ----------------------------------------------------

        new_data = pd.get_dummies(
            new_data,
            drop_first=True
        )


        # ----------------------------------------------------
        # ALIGN WITH MODEL FEATURES
        # ----------------------------------------------------

        model_features = model.feature_names_in_

        new_data = new_data.reindex(
            columns=model_features,
            fill_value=0
        )


        # ----------------------------------------------------
        # HANDLE MISSING VALUES
        # ----------------------------------------------------

        new_data = new_data.fillna(0)


        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        prediction = model.predict(
            new_data
        )[0]


        probabilities = model.predict_proba(
            new_data
        )[0]


        # ----------------------------------------------------
        # DISPLAY RESULT
        # ----------------------------------------------------

        st.success("Prediction completed successfully!")

        st.subheader("Prediction Result")

        severity_names = {

            1: "Severity 1",

            2: "Severity 2",

            3: "Severity 3",

            4: "Severity 4"
        }

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Predicted Severity",
                severity_names.get(
                    prediction,
                    str(prediction)
                )
            )

        with col2:

            confidence = (
                probabilities[
                    list(model.classes_).index(
                        prediction
                    )
                ] * 100
            )

            st.metric(
                "Prediction Confidence",
                f"{confidence:.2f}%"
            )


        # ----------------------------------------------------
        # PROBABILITY CHART
        # ----------------------------------------------------

        st.subheader(
            "Prediction Probabilities"
        )

        probability_df = pd.DataFrame({

            "Severity": [
                f"Severity {x}"
                for x in model.classes_
            ],

            "Probability": probabilities * 100

        })

        st.bar_chart(
            probability_df.set_index(
                "Severity"
            )
        )


        # ----------------------------------------------------
        # PROBABILITY TABLE
        # ----------------------------------------------------

        probability_df["Probability"] = (
            probability_df["Probability"]
            .round(2)
        )

        probability_df["Probability"] = (
            probability_df["Probability"]
            .astype(str)
            + "%"
        )

        st.dataframe(
            probability_df,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# MODEL INSIGHTS PAGE
# ============================================================

elif page == "📊 Model Insights":

    st.header("📊 Model Insights")

    # --------------------------------------------------------
    # MODEL PERFORMANCE
    # --------------------------------------------------------

    st.subheader("Final Model Performance")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Model",
            "Random Forest"
        )

    with col2:

        st.metric(
            "Test Accuracy",
            "50.68%"
        )

    with col3:

        st.metric(
            "Test Samples",
            "811"
        )


    st.divider()


    # --------------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------------

    st.subheader(
        "Feature Importance"
    )

    if os.path.exists(
        FEATURE_IMPORTANCE_PATH
    ):

        importance_df = pd.read_csv(
            FEATURE_IMPORTANCE_PATH
        )

        importance_df = (
            importance_df
            .sort_values(
                "Importance",
                ascending=False
            )
        )

        st.dataframe(
            importance_df,
            use_container_width=True,
            hide_index=True
        )


        # Top 10 features

        top_features = (
            importance_df
            .head(10)
            .sort_values(
                "Importance"
            )
        )

        fig, ax = plt.subplots(
            figsize=(9, 6)
        )

        ax.barh(
            top_features["Feature"],
            top_features["Importance"]
        )

        ax.set_xlabel(
            "Importance"
        )

        ax.set_title(
            "Top 10 Feature Importance"
        )

        plt.tight_layout()

        st.pyplot(fig)
    # --------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------

    st.subheader("Confusion Matrix")

    if os.path.exists(CONFUSION_MATRIX_PATH):
        st.image(
            CONFUSION_MATRIX_PATH,
            caption="Final Random Forest Confusion Matrix",
            use_container_width=True
        )
    else:
        st.warning("Confusion matrix image not found.")
    # --------------------------------------------------------
    # MODEL DESCRIPTION
    # --------------------------------------------------------

    st.divider()

    st.subheader(
        "About the Model"
    )

    st.write(
        """
        The final model is a Random Forest Classifier trained on
        accident-related road, vehicle, weather, location and
        temporal features.

        The model uses 200 decision trees with a maximum depth of
        10 and a random state of 42.

        The strongest features include accident location chainage,
        accident hour, vehicle type, accident cause, day of week,
        month and weather condition.
        """
    )