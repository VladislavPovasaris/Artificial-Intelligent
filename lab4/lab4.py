import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv('Housing.csv')

# Handle missing values
imputer = SimpleImputer(strategy='mean')
data[['area', 'bedrooms', 'bathrooms', 'stories']] = imputer.fit_transform(data[['area', 'bedrooms', 'bathrooms', 'stories']])

# Convert categorical variables
categorical_features = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'parking', 'prefarea', 'furnishingstatus']
data = pd.get_dummies(data, columns=categorical_features, drop_first=True)

# Normalize numerical data
scaler = StandardScaler()
data[['area', 'bedrooms', 'bathrooms', 'stories']] = scaler.fit_transform(data[['area', 'bedrooms', 'bathrooms', 'stories']])

# Split data into training and testing sets
X = data.drop('price', axis=1)
y = data['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1481)

# Linear Regression
linear_regression = LinearRegression()
linear_regression.fit(X_train, y_train)
y_pred_lr = linear_regression.predict(X_test)

# Decision Tree
decision_tree = DecisionTreeRegressor(random_state=1481)
decision_tree.fit(X_train, y_train)
y_pred_dt = decision_tree.predict(X_test)

# Random Forest
random_forest = RandomForestRegressor(n_estimators=100, random_state=1481)
random_forest.fit(X_train, y_train)
y_pred_rf = random_forest.predict(X_test)

# Polynomial regression
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly_features.fit_transform(X_train)
X_test_poly = poly_features.transform(X_test)

# Separate feature scaling for polynomial features
poly_scaler = StandardScaler()
X_train_poly_scaled = poly_scaler.fit_transform(X_train_poly)
X_test_poly_scaled = poly_scaler.transform(X_test_poly)

# Ridge Polynomial Regression with regularization
ridge_poly_regression = Ridge(alpha=1.0)  # You can adjust the regularization strength
ridge_poly_regression.fit(X_train_poly_scaled, y_train)
y_pred_ridge_pr = ridge_poly_regression.predict(X_test_poly_scaled)

# Evaluate the models
models = {
    'Linear Regression': (y_pred_lr, mean_absolute_error(y_test, y_pred_lr), mean_squared_error(y_test, y_pred_lr)),
    'Decision Tree': (y_pred_dt, mean_absolute_error(y_test, y_pred_dt), mean_squared_error(y_test, y_pred_dt)),
    'Random Forest': (y_pred_rf, mean_absolute_error(y_test, y_pred_rf), mean_squared_error(y_test, y_pred_rf)),
    'Ridge Polynomial Regression': (y_pred_ridge_pr, mean_absolute_error(y_test, y_pred_ridge_pr), mean_squared_error(y_test, y_pred_ridge_pr))
}

# Print evaluation metrics
for name, (_, mae, mse) in models.items():
    print(f'{name} MAE: {mae}')
    print(f'{name} MSE: {mse}')

# Visualization of predictions
plt.figure(figsize=(12, 8))
for i, (name, (y_pred, _, _)) in enumerate(models.items()):
    plt.subplot(2, 2, i+1)
    plt.scatter(y_test, y_pred)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.xlabel('Actual Prices')
    plt.ylabel('Predicted Prices')
    plt.title(name)

plt.tight_layout()
plt.show()

# Find the best MAE and MSE among all models
best_mae = float('inf')
best_mae_model = None

best_mse = float('inf')
best_mse_model = None

for name, (_, mae, mse) in models.items():
    if mae < best_mae:
        best_mae = mae
        best_mae_model = name
    
    if mse < best_mse:
        best_mse = mse
        best_mse_model = name

print(f'Best MAE: {best_mae} from {best_mae_model}')
print(f'Best MSE: {best_mse} from {best_mse_model}')
