# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score


# read the dataset
data = pd.read_csv('C:/Users/hsri2/OneDrive/Desktop/Data sheet/Advertising.csv')

# display the first 5 rows of the dataset
print(data.head())

# display summary statistics of the dataset
print(data.describe())

# define the features and target variable
X = data[['TV', 'Radio', 'Newspaper']]
y = data['Sales']

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Linear Regression model
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# print the mean squared error and coefficient of determination for linear regression model
print('Mean squared error (Linear Regression): {:.2f}'.format(mean_squared_error(y_test, y_pred_lr)))
print('Coefficient of determination (Linear Regression): {:.2f}'.format(r2_score(y_test, y_pred_lr)))

# Ridge Regression model
ridge = Ridge(alpha=0.1)
ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_test)

# print the mean squared error and coefficient of determination for ridge regression model
print('Mean squared error (Ridge Regression): {:.2f}'.format(mean_squared_error(y_test, y_pred_ridge)))
print('Coefficient of determination (Ridge Regression): {:.2f}'.format(r2_score(y_test, y_pred_ridge)))

# Lasso Regression model
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
y_pred_lasso = lasso.predict(X_test)

# print the mean squared error and coefficient of determination for lasso regression model
print('Mean squared error (Lasso Regression): {:.2f}'.format(mean_squared_error(y_test, y_pred_lasso)))
print('Coefficient of determination (Lasso Regression): {:.2f}'.format(r2_score(y_test, y_pred_lasso)))

# Decision Tree Regression model
dt = DecisionTreeRegressor(max_depth=3, random_state=0)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

# print the mean squared error and coefficient of determination for decision tree regression model
print('Mean squared error (Decision Tree Regression): {:.2f}'.format(mean_squared_error(y_test, y_pred_dt)))
print('Coefficient of determination (Decision Tree Regression): {:.2f}'.format(r2_score(y_test, y_pred_dt)))

# Random Forest Regression model
rf = RandomForestRegressor(n_estimators=100, max_depth=3, random_state=0)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# print the mean squared error and coefficient of determination for random forest regression model
print('Mean squared error (Random Forest Regression): {:.2f}'.format(mean_squared_error(y_test, y_pred_rf)))
print('Coefficient of determination (Random Forest Regression): {:.2f}'.format(r2_score(y_test, y_pred_rf)))

# Gradient Boosting Regression model
gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=0)
gb.fit(X_train, y_train)
y_pred_gb = gb.predict(X_test)

#print the mean squared error and coefficient of determination for gradient boosting regression model
print('Mean squared error (Gradient Boosting): %.2f' % mean_squared_error(y_test, y_pred_gb))
print('Coefficient of determination (Gradient Boosting): %.2f' % r2_score(y_test, y_pred_gb))

#plot feature importance for gradient boosting regression model
feature_importance_gb = gb.feature_importances_
sorted_idx_gb = feature_importance_gb.argsort()
pos_gb = np.arange(sorted_idx_gb.shape[0]) + .5
plt.subplot(1, 2, 2)
plt.barh(pos_gb, feature_importance_gb[sorted_idx_gb], align='center')
plt.yticks(pos_gb, X.columns[sorted_idx_gb])
plt.xlabel('Feature Importance')
plt.title('Gradient Boosting Regression')
plt.show()
