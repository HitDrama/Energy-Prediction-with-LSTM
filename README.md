# Energy Consumption Prediction with LSTM

This project uses an **LSTM (Long Short-Term Memory)** model to predict energy consumption based on historical hourly data. The model is trained using the **DOM_hourly.csv** dataset, which contains historical energy consumption data.

## Dataset

The dataset used for training the model can be found at the following link: 

[Hourly Energy Consumption Dataset - Kaggle](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption/data?select=DOM_hourly.csv)

## Model Overview

The model used for prediction is a **Sequential LSTM model**, consisting of:

- **LSTM Layer**: 50 units with `tanh` activation function.
- **Dense Layer**: 1 unit for the final output (predicted energy consumption).

The model is compiled using the **Adam optimizer** and **Mean Squared Error (MSE)** loss function.

### Model Architecture

```python
model = Sequential([  # Initialize the Sequential model
    LSTM(50, activation='tanh', input_shape=(X_train.shape[1], 1)),  # Add LSTM layer with 50 units, tanh activation, and input shape (time steps, 1)
    Dense(1)  # Dense layer with 1 output unit
])
model.compile(optimizer='adam', loss='mse')  # Compile the model with Adam optimizer and MSE loss function
```
---
## Data Preprocessing

The data is scaled using the MinMaxScaler to normalize the features to the range [0, 1].
```python
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df)
```

The dataset is then transformed into a supervised learning format with input features (X) and target values (y) using a sliding window approach of size 24 hours.
```python
def create_dataset(data, window_size=24):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i - window_size:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

X, y = create_dataset(data_scaled, window_size=24)
X = X.reshape((X.shape[0], X.shape[1], 1))
```

The data is then split into training and testing sets:
```python
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
```

---

## Results

Here are the results of the LSTM model testing and evaluation:

### 1. Comparison Chart between **Actual** and **Predicted** Values:

The chart below shows the comparison between the **actual energy consumption values** and the **predicted values** by the LSTM model. The model is able to predict the trends quite accurately, although there are some slight discrepancies at certain points.

<img src="https://github.com/HitDrama/Energy-Prediction-with-LSTM/blob/main/static/img/test-energy.png" alt="Prediction vs Actual" width="600"/>

### 2. Test Result Image:

Below is another image showing the test results of the LSTM model, providing additional evaluation of the model's performance on the test data.

<img src="https://github.com/HitDrama/Energy-Prediction-with-LSTM/blob/main/static/img/comparison-plot.png" alt="Test Image" width="600"/>

---

## Developer

This model was developed by **Dang To Nhan**.

