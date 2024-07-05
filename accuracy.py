import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()
X = iris.data
y = iris.target


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=200)

param_grid = {
    'C': [0.1, 1, 10, 100, 1000],
    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
    'penalty': ['l2']
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

best_grid_params = grid_search.best_params_
best_grid_score = grid_search.best_score_

param_dist = {
    'C': np.logspace(-4, 4, 20),
    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
    'penalty': ['l2'] 
}

random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=100, cv=5, n_jobs=-1, random_state=42)
random_search.fit(X_train, y_train)


best_random_params = random_search.best_params_
best_random_score = random_search.best_score_


print("Best parameters from GridSearchCV:", best_grid_params)
print("Best cross-validation accuracy from GridSearchCV:", best_grid_score)

print("Best parameters from RandomizedSearchCV:", best_random_params)
print("Best cross-validation accuracy from RandomizedSearchCV:", best_random_score)

best_grid_model = grid_search.best_estimator_
y_pred_grid = best_grid_model.predict(X_test)
test_accuracy_grid = accuracy_score(y_test, y_pred_grid)
print("Test set accuracy with GridSearchCV best parameters:", test_accuracy_grid)

best_random_model = random_search.best_estimator_
y_pred_random = best_random_model.predict(X_test)
test_accuracy_random = accuracy_score(y_test, y_pred_random)
print("Test set accuracy with RandomizedSearchCV best parameters:", test_accuracy_random)
