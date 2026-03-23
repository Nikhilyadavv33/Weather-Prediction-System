import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.linear_model import LogisticRegression
import joblib
import plotly.graph_objects as go

# ------------------------------
# LOAD DATASET
# ------------------------------
data = pd.read_csv("weatherAUS.csv")

data = data[['MinTemp','MaxTemp','Humidity3pm','Pressure3pm','WindSpeed3pm','RainTomorrow']]
data = data.dropna()

# Convert Yes/No to 1/0
data['RainTomorrow'] = data['RainTomorrow'].map({'Yes':1, 'No':0})

# Inputs & Output
X = data[['MinTemp','MaxTemp','Humidity3pm','Pressure3pm','WindSpeed3pm']]
y = data['RainTomorrow']

# ------------------------------
# DNN MODEL
# ------------------------------
model = Sequential()
model.add(Dense(16, activation='relu', input_dim=5))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=5)

model.save("model.h5")
print("DNN model trained & saved!")

# ------------------------------
# LOGISTIC REGRESSION
# ------------------------------
lr_model = LogisticRegression()
lr_model.fit(X, y)

joblib.dump(lr_model, "lr_model.pkl")
print("Logistic Regression model saved!")

# ------------------------------
# LSTM MODEL
# ------------------------------
X_lstm = np.array(X)
X_lstm = X_lstm.reshape((X_lstm.shape[0], 1, X_lstm.shape[1]))

lstm_model = Sequential()
lstm_model.add(LSTM(32, activation='relu', input_shape=(1, X.shape[1])))
lstm_model.add(Dense(1, activation='sigmoid'))

lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
lstm_model.fit(X_lstm, y, epochs=5)

lstm_model.save("lstm_model.h5")
print("LSTM model trained & saved!")

# ------------------------------
# CALCULATE ACCURACY
# ------------------------------
dnn_acc = model.evaluate(X, y, verbose=0)[1]
lstm_acc = lstm_model.evaluate(X_lstm, y, verbose=0)[1]

print(f"DNN Accuracy: {dnn_acc*100:.2f}%")
print(f"LSTM Accuracy: {lstm_acc*100:.2f}%")

# ------------------------------
# CREATE GRAPH (PLOTLY)
# ------------------------------
models = ['DNN', 'LSTM']
accuracies = [dnn_acc, lstm_acc]

fig = go.Figure(data=[
    go.Bar(
        x=models,
        y=accuracies,
        text=[f"{acc*100:.2f}%" for acc in accuracies],
        textposition='auto'
    )
])

fig.update_layout(
    title="Model Accuracy Comparison",
    xaxis_title="Models",
    yaxis_title="Accuracy"
)

# Save graph
fig.write_html("accuracy.html")

print("Graph created successfully! (accuracy.html)")