import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the datasets
file_path_A = 'KC_clean.csv'
file_path_B = 'Lancaster Data - Cleaned.csv'
dataset_A = pd.read_csv(file_path_A)
dataset_B = pd.read_csv(file_path_B)

for dataset in [dataset_A, dataset_B]:
    for column in dataset.columns:
        if dataset[column].dtype in ['int64', 'float64']:
            dataset[column] = dataset[column].fillna(dataset[column].median())
        else:
            dataset[column] = dataset[column].fillna(dataset[column].mode()[0])

# Increase the price in dataset_A by 9% (Inflation?)
dataset_A['price'] = dataset_A['price'] * 1.09

# Splitting features and target variable for both datasets
features_A = [col for col in dataset_A.columns if col != 'price']
features_B = [col for col in dataset_B.columns if col != 'price']

common_features = list(set(features_A).intersection(features_B))

X_A = dataset_A[common_features]
y_A = dataset_A['price']

X_B = dataset_B[common_features]
y_B = dataset_B['price']

# Normilized
y_A_median, y_A_std = y_A.median(), y_A.std()
y_A_normalized = (y_A - y_A_median) / y_A_std

y_B_median, y_B_std = y_B.median(), y_B.std()
y_B_normalized = (y_B - y_B_median) / y_B_std

# Splitting dataset B into 20% train, 10% validation, 70% test
X_B_train, X_B_temp, y_B_train, y_B_temp = train_test_split(X_B, y_B_normalized, test_size=0.80, random_state=42)
X_B_val, X_B_test, y_B_val, y_B_test = train_test_split(X_B_temp, y_B_temp, test_size=0.875, random_state=42)

# Training set consists of 100% of dataset A and 20% of dataset B
X_train = pd.concat([X_A, X_B_train], axis=0)
y_train = pd.concat([y_A_normalized, y_B_train], axis=0)

scaler_A = StandardScaler()
X_A_scaled = scaler_A.fit_transform(X_A)

scaler_B = StandardScaler()
X_B_train_scaled = scaler_B.fit_transform(X_B_train)
X_B_val_scaled = scaler_B.transform(X_B_val)
X_B_test_scaled = scaler_B.transform(X_B_test)

X_train_scaled = np.vstack((X_A_scaled, X_B_train_scaled))

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='linear')
])

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])

history = model.fit(X_train_scaled, y_train, validation_data=(X_B_val_scaled, y_val), epochs=100, batch_size=32, verbose=1)

y_B_test_pred = model.predict(X_B_test_scaled)

y_B_test_pred_actual = y_B_median + (y_B_std * y_B_test_pred.ravel())
y_B_test_actual = y_B_median + (y_B_std * y_B_test.ravel())

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(history.history['mean_squared_error'], label='Training MSE')
plt.plot(history.history['val_mean_squared_error'], label='Validation MSE')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.title('Training and Validation MSE over Epochs')
plt.legend()
plt.show()

print("\nPredicted vs Actual Prices for Test Set:")
for i in range(10):
    print(f"Sample {i + 1}: Predicted Price: ${y_B_test_pred_actual[i]:,.2f}, Actual Price: ${y_B_test_actual[i]:,.2f}")
