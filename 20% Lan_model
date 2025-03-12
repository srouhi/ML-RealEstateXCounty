import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

file_path = 'Lancaster Data - Cleaned.csv'
dataset = pd.read_csv(file_path)

dataset.columns = dataset.columns.str.strip()

categorical_columns = ['CDU', 'Qual', 'date']

label_encoder = LabelEncoder()
for column in categorical_columns:
    if dataset[column].dtype == 'object' or dataset[column].dtype == 'category':
        dataset[column] = label_encoder.fit_transform(dataset[column].astype(str))

dataset['floors'] = pd.to_numeric(dataset['floors'], errors='coerce')

for column in dataset.columns:
    if dataset[column].dtype in ['int64', 'float64']:
        dataset[column] = dataset[column].fillna(dataset[column].median())
    else:
        dataset[column] = dataset[column].fillna(dataset[column].mode()[0])

features = [col for col in dataset.columns if col != 'price']
X = dataset[features]
y = dataset['price']

y_median, y_std = y.median(), y.std()
y_normalized = (y - y_median) / y_std

# Splitting the dataset into 20% train, 10% validation, 70% test
X_train, X_temp, y_train, y_temp = train_test_split(X, y_normalized, test_size=0.80, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.875, random_state=42)  # 10% validation, 70% testing

# Normalized
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_val_scaled = scaler_X.transform(X_val)
X_test_scaled = scaler_X.transform(X_test)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='linear')
])

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])

history = model.fit(X_train_scaled, y_train, validation_data=(X_val_scaled, y_val), epochs=100, batch_size=32, verbose=1)

y_test_pred = model.predict(X_test_scaled)

y_test_pred_actual = y_median + (y_std * y_test_pred.ravel())
y_test_actual = y_median + (y_std * y_test.ravel())

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
    print(f"Sample {i + 1}: Predicted Price: ${y_test_pred_actual[i]:,.2f}, Actual Price: ${y_test_actual[i]:,.2f}")
