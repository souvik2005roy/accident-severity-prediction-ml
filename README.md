# Accident Severity Prediction

A Machine Learning project for predicting accident severity using
road, vehicle, weather, location, time, and accident-cause information.

## Objective

The objective of this project is to develop a classification model
that predicts the severity category of a road accident based on
available accident characteristics.

## Dataset

The dataset contains accident records with information including:

- Accident date and time
- Accident location
- Location chainage
- Road condition
- Road features
- Accident causes
- Weather conditions
- Vehicle types
- Accident severity

## Data Preprocessing

The following preprocessing steps were performed:

1. Removed duplicate records.
2. Removed records with missing accident severity.
3. Converted accident date into Year, Month and Day.
4. Extracted Hour from accident time.
5. Converted categorical variables using one-hot encoding.
6. Handled missing values.
7. Split the dataset into training and testing sets using stratified sampling.

## Models Evaluated

Several machine learning algorithms were evaluated:

- Logistic Regression
- Random Forest
- Tuned Random Forest
- XGBoost
- Tuned XGBoost
- Extra Trees Classifier

Random Forest produced the best overall test accuracy.

## Final Model

The final model is a Random Forest Classifier with:

- Number of trees: 200
- Maximum depth: 10
- Minimum samples split: 2
- Minimum samples leaf: 1
- Random state: 42

## Results

Final Random Forest performance:

| Metric | Score |
|---|---:|
| Accuracy | 50.68% |
| Test Samples | 811 |
| Training Samples | 3242 |

The model achieved the strongest prediction performance for
Severity Classes 2 and 3, while performance for the minority
Severity Class 1 remained limited.

## Feature Importance

The most important features identified by the Random Forest were:

1. Accident Location Chainage
2. Hour
3. Vehicle Type
4. Accident Cause
5. Day of Week
6. Month
7. Day
8. Weather Condition

## Visualization

The project generates:

- Accident severity distribution
- Accidents by weather condition
- Confusion matrix
- Feature importance visualization

## Prediction

A prediction script is included that loads the trained model and
produces an accident severity prediction along with prediction
probabilities.

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib
- Joblib
