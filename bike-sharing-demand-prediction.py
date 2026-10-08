import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import r2_score

df = pd.read_csv("hour.csv")

split_index = int(len(df) * 0.8)

train = df.iloc[:split_index]
test = df.iloc[split_index:]

features = [
    "season",
    "yr",
    "mnth",
    "hr",
    "holiday",
    "weekday",
    "workingday",
    "weathersit",
    "temp",
    "atemp",
    "hum",
    "windspeed"
]

X_train = train[features]
y_train = train["cnt"]

X_test = test[features]
y_test = test["cnt"]

preprocessor = ColumnTransformer([
    ("numerical", StandardScaler(), ["temp", "atemp", "hum", "windspeed"]),
    ("categorical", OneHotEncoder(handle_unknown="ignore"), ["season", "yr", "mnth", "hr", "holiday", "weekday", "workingday", "weathersit"])
])

linear_model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

forest_model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=100, random_state=1))
])

linear_model.fit(X_train, y_train)
forest_model.fit(X_train, y_train)

forest_prediction = forest_model.predict(X_test)
linear_prediction = linear_model.predict(X_test)

linear_mae = mean_absolute_error(y_test, linear_prediction)
linear_rmse = root_mean_squared_error(y_test, linear_prediction)
linear_r2_test_score = r2_score(y_test, linear_prediction)

forest_mae = mean_absolute_error(y_test, forest_prediction)
forest_rmse = root_mean_squared_error(y_test, forest_prediction)
forest_r2_test_score = r2_score(y_test, forest_prediction)

print("Linear Model's mean absolute error = ", linear_mae)
print("Linear Model's root mean squared error = ", linear_rmse)
print("Linear Model's r2 score = ", linear_r2_test_score)

print("Forest Model's mean absolute error = ", forest_mae)
print("Forest Model's root mean squared error = ", forest_rmse)
print("Forest Model's r2 score = ", forest_r2_test_score)

forest_cv = TimeSeriesSplit(n_splits=5)
scores = cross_val_score(forest_model, X_train, y_train, cv=forest_cv)

print("Forest Model's cross validation scores: ", scores)
print("Forest Model's average score = ", scores.mean())

feature_names = forest_model.named_steps["preprocessor"].get_feature_names_out()

importances = forest_model.named_steps["regressor"].feature_importances_

importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
})

importance_df = importance_df.sort_values(
    by="importance",
    ascending=False
)

top_features = importance_df.head(10)

plt.barh(top_features["feature"], top_features["importance"])
plt.xlabel("Importance")
plt.ylabel("Features")
plt.title("Most Important Features")
plt.show()