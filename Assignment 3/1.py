import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import pandas as pd

data = pd.read_csv("/Users/zuhair/Desktop/ABC/datasaurus.csv")

X = data.drop(columns=['dataset'])
y = data['dataset']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

j48_baseline = DecisionTreeClassifier()
j48_baseline.fit(X_train, y_train)

rf_baseline = RandomForestClassifier()
rf_baseline.fit(X_train, y_train)

print("J48 Baseline Evaluation:")
print(classification_report(y_test, j48_baseline.predict(X_test)))

print("Random Forest Baseline Evaluation:")
print(classification_report(y_test, rf_baseline.predict(X_test)))

j48_param_dist = {
    "max_depth": [None, 5, 10, 15],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "criterion": ["gini", "entropy"]
}

rf_param_dist = {
    "n_estimators": [50, 100, 150],
    "max_depth": [None, 5, 10, 15],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "criterion": ["gini", "entropy"]
}

j48_random_search = RandomizedSearchCV(
    j48_baseline, param_distributions=j48_param_dist, n_iter=50, cv=5)
j48_random_search.fit(X_train, y_train)

rf_grid_search = GridSearchCV(rf_baseline, param_grid=rf_param_dist, cv=5)
rf_grid_search.fit(X_train, y_train)

print("J48 Random Search Evaluation:")
print(classification_report(
    y_test, j48_random_search.best_estimator_.predict(X_test)))

print("Random Forest Grid Search Evaluation:")
print(classification_report(y_test, rf_grid_search.best_estimator_.predict(X_test)))
