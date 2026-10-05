Bike Sharing Demand Prediction

A machine learning regression project using the UCI Bike Sharing dataset to predict the number of bike rentals (cnt) based on weather, time, and calendar-related features.

Project Overview

The goal of this project is to predict hourly bike rental demand using two different regression models:

Linear Regression

Random Forest Regressor

The project also explores preprocessing with ColumnTransformer, categorical encoding, feature scaling, pipelines, time-based train/test splitting, time-series cross-validation, and Random Forest feature importance.

Dataset

The project uses the Bike Sharing Dataset, specifically hour.csv.

The target variable is:

cnt — total number of bike rentals

The features used include:

season

yr

mnth

hr

holiday

weekday

workingday

weathersit

temp

atemp

hum

windspeed

Approach

1. Train/Test Split

Because the dataset is time-based, the data was split chronologically rather than randomly.

The first 80% of the data was used for training and the remaining 20% was used as the test set.

split_index = int(len(df) * 0.8)

train = df.iloc[:split_index]
test = df.iloc[split_index:]

2. Preprocessing

A ColumnTransformer was used to apply different preprocessing steps to numerical and categorical features.

Numerical features:

Standardized using StandardScaler

Categorical features:

Encoded using OneHotEncoder

handle_unknown="ignore" was used to prevent errors when a category appears in a validation/test split but was not present during fitting.

3. Pipelines

The preprocessing and model steps were combined using Pipeline.

This was done separately for Linear Regression and Random Forest.

4. Models

Two regression models were compared:

LinearRegression

RandomForestRegressor

The Random Forest model was configured with 100 trees.

5. Evaluation

The models were evaluated using:

MAE (Mean Absolute Error)

RMSE (Root Mean Squared Error)

R² (R-squared)

For the Random Forest model, TimeSeriesSplit with 5 splits was also used for cross-validation.

6. Feature Importance

Random Forest feature importance was extracted after training.

The top 10 transformed features were visualized to understand which features contributed most to the model's predictions.

Results

Test Set Performance

Model

MAE

RMSE

R²

Linear Regression

98.80

133.84

0.632

Random Forest

55.41

83.42

0.857

The Random Forest model performed substantially better than the Linear Regression model on the held-out test set.

Its test-set R² was approximately 0.857, compared with approximately 0.632 for Linear Regression.

Time-Series Cross-Validation

The Random Forest model was evaluated using 5 sequential time-series splits:

[0.4027, 0.7639, 0.6468, 0.6466, 0.8811]

Average R²:

0.6682

The cross-validation scores vary considerably between splits. This shows that model performance is not consistent across all time periods in the training data.

The final test-set score should therefore be interpreted separately from the cross-validation results.

Feature Importance

The Random Forest model's most important transformed features included:

atemp

hr_17

hr_18

hum

hr_8

workingday_1

temp

workingday_0

yr_0

yr_1

The results suggest that perceived temperature, humidity, hour of the day, working-day status, and year-related features were important for predicting bike rental demand.

Technologies Used

Python

pandas

scikit-learn

matplotlib

Scikit-learn techniques used:

LinearRegression

RandomForestRegressor

ColumnTransformer

StandardScaler

OneHotEncoder

Pipeline

TimeSeriesSplit

cross_val_score

MAE

RMSE

R²

Random Forest feature importance

What I Learned

Through this project, I practiced:

Preparing a regression dataset for machine learning

Chronological train/test splitting for time-based data

Separating numerical and categorical preprocessing

One-hot encoding categorical variables

Feature scaling

Building preprocessing/model pipelines

Comparing different regression models

Evaluating regression models with multiple metrics

Applying time-series cross-validation

Interpreting Random Forest feature importance

Project Structure

bike-sharing-demand/
│
├── hour.csv
├── main.py
├── README.md
└── requirements.txt

Conclusion

Random Forest performed better than Linear Regression on the held-out test set, achieving an R² of approximately 0.857 and lower MAE and RMSE.

The project also showed that evaluation results can vary significantly across different time-based validation splits, highlighting the importance of using time-aware validation when working with chronological data.