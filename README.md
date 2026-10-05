# Bike Sharing Demand Prediction

This repository contains a machine learning project focused on predicting hourly bike rental demand. Using the UCI Bike Sharing dataset, the objective is to estimate the number of rentals (`cnt`) based on weather conditions, time, and calendar data. 

I developed this project to practically apply data preprocessing techniques, build machine learning pipelines, and compare regression models on time-series data.

## Dataset
The project utilizes the `hour.csv` file from the Bike Sharing Dataset.
- **Target Variable:** `cnt` (total number of bike rentals)
- **Features Used:** `season`, `yr`, `mnth`, `hr`, `holiday`, `weekday`, `workingday`, `weathersit`, `temp`, `atemp`, `hum`, `windspeed`.

## Methodology

### 1. Chronological Train/Test Split
Since the data is time-dependent, using a standard random split would cause data leakage. To evaluate the models realistically, I split the dataset chronologically: the first 80% of the records were used for training, and the final 20% were kept as the hold-out test set.

### 2. Preprocessing with Pipelines
To maintain clean code and prevent leakage, I integrated preprocessing steps into a `Pipeline` using `ColumnTransformer`:
- **Numerical Features:** Scaled using `StandardScaler`.
- **Categorical Features:** Encoded using `OneHotEncoder` with `handle_unknown="ignore"` to handle any unseen categories in the test set.

### 3. Model Training
I trained and compared two different models:
- **Linear Regression:** Used as a baseline model.
- **Random Forest Regressor:** A more complex ensemble model configured with 100 trees.

## Results

Below is the performance comparison of the models on the 20% test set:

| Model | MAE | RMSE | R² Score |
| Linear Regression | 98.80 | 133.84 | 0.632 |
| Random Forest | 55.41 | 83.42 | 0.857 |

The Random Forest model outperformed the baseline Linear Regression, achieving an R² score of 0.857. 

**Cross-Validation Insights:** 
I also evaluated the Random Forest model using `TimeSeriesSplit` (5 splits). The cross-validation R² scores varied between 0.40 and 0.88 (average ~0.66). This variance indicates that the model's performance fluctuates depending on the specific time frame, highlighting the importance of proper validation techniques for chronological data.

**Feature Importance:**
According to the Random Forest model, the top drivers of rental demand were:
1. `atemp` (Perceived temperature)
2. `hr_17` and `hr_18` (5 PM and 6 PM rush hours)
3. `hum` (Humidity)
4. `hr_8` (8 AM rush hour)

## Installation and Usage

**Prerequisites:** I developed and tested this project using **Python 3.14.7**.

1. Clone this repository.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
3. Run the main script:
   python main.py

Key Takeaways
Through building this project, I gained hands-on experience in:
Handling time-based datasets without introducing data leakage.
Structuring preprocessing steps using scikit-learn Pipelines.
Evaluating regression models with multiple metrics (MAE, RMSE, R²).

Interpreting the underlying feature importance of ensemble models.
