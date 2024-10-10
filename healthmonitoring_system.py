import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

# Load the health monitoring data
health_data = pd.read_csv("C:/Users/swathiga/Downloads/healthmonitoring (1).csv")

# Data cleaning
health_data['Timestamp'] = pd.to_datetime(health_data['Timestamp'])
health_data['HeartRate'] = pd.to_numeric(health_data['HeartRate'], errors='coerce')
health_data['BloodPressure'] = health_data['BloodPressure'].str.split('/')
health_data[['Systolic', 'Diastolic']] = pd.DataFrame(health_data['BloodPressure'].tolist(), index=health_data.index)
health_data['Systolic'] = pd.to_numeric(health_data['Systolic'], errors='coerce')
health_data['Diastolic'] = pd.to_numeric(health_data['Diastolic'], errors='coerce')

# Drop rows with missing values in critical columns
health_data_clean = health_data.dropna(subset=['HeartRate', 'Systolic', 'Diastolic'])

# Prepare data for LSTM model - use heart rate, systolic, and diastolic for prediction
data = health_data_clean[['HeartRate', 'Systolic', 'Diastolic']].values

# Scale the data between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)

# Create sequences for LSTM (timesteps)
def create_sequences(data, time_steps=10):
    X, y = [], []
    for i in range(time_steps, len(data)):
        X.append(data[i-time_steps:i])
        y.append(data[i])
    return np.array(X), np.array(y)

time_steps = 10
X, y = create_sequences(data_scaled, time_steps)

# Split data into training and testing sets
split = int(0.8 * len(X))
X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=3))  # 3 outputs: heart rate, systolic, diastolic

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test), verbose=2)

# Predict on the test set
predictions = model.predict(X_test)

# Inverse scale the predictions back to original values
predictions_rescaled = scaler.inverse_transform(predictions)

# Inverse scale the actual test data
actual_rescaled = scaler.inverse_transform(y_test)

# Compare predictions and actual values (anomalies will be evident where there's a significant difference)
for i in range(5):
    print(f"Prediction {i + 1}: {predictions_rescaled[i]}")
    print(f"Actual {i + 1}: {actual_rescaled[i]}")
    print("-" * 50)

# Visualize the comparison between predicted and actual values for heart rate, systolic and diastolic
plt.figure(figsize=(14, 7))

plt.subplot(3, 1, 1)
plt.plot(actual_rescaled[:, 0], color='blue', label='Actual Heart Rate')
plt.plot(predictions_rescaled[:, 0], color='red', label='Predicted Heart Rate')
plt.title('Heart Rate - Actual vs Predicted')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(actual_rescaled[:, 1], color='blue', label='Actual Systolic BP')
plt.plot(predictions_rescaled[:, 1], color='red', label='Predicted Systolic BP')
plt.title('Systolic Blood Pressure - Actual vs Predicted')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(actual_rescaled[:, 2], color='blue', label='Actual Diastolic BP')
plt.plot(predictions_rescaled[:, 2], color='red', label='Predicted Diastolic BP')
plt.title('Diastolic Blood Pressure - Actual vs Predicted')
plt.legend()

plt.tight_layout()
plt.show()
