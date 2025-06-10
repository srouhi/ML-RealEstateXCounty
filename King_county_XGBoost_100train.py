import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = 'kc_real_estate_python(in).csv'
data = pd.read_csv(file_path)

for column in data.columns:
    if data[column].dtype in ['int64', 'float64']:
        data[column] = data[column].fillna(data[column].mean())
    else:
        data[column] = data[column].fillna(data[column].mode()[0])

average_price = data['price'].mean()
print(f"Average house price: ${average_price:,.2f}")

# Convert any object columns to numeric if possible
for column in data.columns:
    if data[column].dtype == 'object':
        data[column] = pd.to_numeric(data[column], errors='coerce').fillna(0)

features = [col for col in data.columns if col != 'price']
X = data[features]
y = data['price']

# Normalizing the target variable 'price'
scaler_y = MinMaxScaler()
y = scaler_y.fit_transform(y.values.reshape(-1, 1)).ravel()

scaler = StandardScaler()
X = scaler.fit_transform(X)

params = {
    'objective': 'reg:squarederror',
    'learning_rate': 0.05,
    'max_depth': 6,
    'alpha': 10,
    'n_estimators': 1000
}

model = xgb.XGBRegressor(**params)
model.fit(X, y, verbose=False)

# Hyperparameter Tuning 
param_grid = {
    'max_depth': [4, 6, 8],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [500, 1000, 1500],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

grid_search = GridSearchCV(estimator=xgb.XGBRegressor(objective='reg:squarederror'),
                           param_grid=param_grid,
                           cv=3,
                           n_jobs=-1,
                           scoring='neg_root_mean_squared_error',
                           verbose=0)

grid_search.fit(X, y)

best_model = grid_search.best_estimator_

rmse_best = -grid_search.best_score_

print(f"Improved RMSE from Cross-Validation: {rmse_best}")

importances = best_model.feature_importances_

feature_importances = pd.Series(importances, index=features)

feature_importances = feature_importances.sort_values(ascending=False)

print("\nFeature Importances from the Best Model:")
print(feature_importances)

plt.figure(figsize=(12, 8))
sns.barplot(x=feature_importances.values, y=feature_importances.index)
plt.title('Feature Importances from the Best XGBoost Model')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.tight_layout()
plt.show()
