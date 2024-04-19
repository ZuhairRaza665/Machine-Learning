# Date: 2024-03-12
# CSC354 – Assignment2 – ML – Concept Learning
# Zuhair Raza
# FA20-BCS-085
# Brief Description: This code uses a dataset about cars to predict their selling prices. It first prepares the data, splitting it into training and testing sets. Then, it creates a basic model to predict prices. After that, it tries to improve the model by testing different combinations of settings using two methods: random search and grid search. Finally, it evaluates the best models from each method to see which one performs better at predicting car prices.
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

data = pd.read_csv("/Users/zuhair/Desktop/ABC/cars-dataset.csv")


data = pd.get_dummies(data)


X = data.drop(columns=['selling_price'])
y = data['selling_price']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)


baseline_model = DecisionTreeRegressor(random_state=42)
baseline_model.fit(X_train, y_train)


baseline_predictions = baseline_model.predict(X_test)
baseline_mse = mean_squared_error(y_test, baseline_predictions)
print("Baseline Mean Squared Error:", baseline_mse)


random_param_dist = {
    "max_depth": [None, 5, 10, 15],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}
random_search = RandomizedSearchCV(DecisionTreeRegressor(
    random_state=42), param_distributions=random_param_dist, n_iter=50, cv=5, random_state=42)
random_search.fit(X_train, y_train)
print("Random Search Best Parameters:", random_search.best_params_)

grid_param_grid = {
    "max_depth": [None, 5, 10, 15],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}
grid_search = GridSearchCV(DecisionTreeRegressor(
    random_state=42), param_grid=grid_param_grid, cv=5)
grid_search.fit(X_train, y_train)
print("Grid Search Best Parameters:", grid_search.best_params_)


random_search_best_model = random_search.best_estimator_
grid_search_best_model = grid_search.best_estimator_

random_search_predictions = random_search_best_model.predict(X_test)
grid_search_predictions = grid_search_best_model.predict(X_test)

random_search_mse = mean_squared_error(y_test, random_search_predictions)
grid_search_mse = mean_squared_error(y_test, grid_search_predictions)

print("Random Search Best Model Mean Squared Error:", random_search_mse)
print("Grid Search Best Model Mean Squared Error:", grid_search_mse)
