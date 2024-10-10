import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

# Load health monitoring data (CSV file)
data = pd.read_csv('C:/Users/swathiga/Downloads/healthmonitoring (1).csv')

# Print DataFrame and its columns for debugging
print(data.head())
print("Columns in the CSV:", data.columns.tolist())

# Split BloodPressure into systolic and diastolic (assuming the format is '120/80')
data[['Systolic', 'Diastolic']] = data['BloodPressure'].str.split('/', expand=True)

# Convert systolic and diastolic columns to numeric
data['Systolic'] = pd.to_numeric(data['Systolic'], errors='coerce')
data['Diastolic'] = pd.to_numeric(data['Diastolic'], errors='coerce')

# Preprocess data (scaling, splitting for LSTM)
def preprocess_data(data):
    scaler = MinMaxScaler()
    # Scaling the relevant columns
    scaled_data = scaler.fit_transform(data[['HeartRate', 'Systolic', 'Diastolic']])

    X, y = [], []
    seq_length = 10  # Sequence length for LSTM input
    for i in range(seq_length, len(scaled_data)):
        X.append(scaled_data[i-seq_length:i])
        y.append(scaled_data[i])

    X, y = np.array(X), np.array(y)
    return X, y, scaler

# Run the preprocessing
X, y, scaler = preprocess_data(data)

# Define the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1]))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Fit the model
model.fit(X, y, epochs=20, batch_size=32)

# Anomaly detection function
def anomaly_detection():
    # Using the same preprocessing for the test data
    X_test, y_test, _ = preprocess_data(data)
    predictions = model.predict(X_test)
    
    # Reverse scaling to get actual values
    predictions = scaler.inverse_transform(predictions)
    y_test = scaler.inverse_transform(y_test)

    for i in range(5):  # Displaying the first 5 predictions
        print(f"Prediction {i+1}: {predictions[i]}")
        print(f"Actual {i+1}: {y_test[i]}")
        print("--------------------------------------------------")

# Call anomaly detection
anomaly_detection()
