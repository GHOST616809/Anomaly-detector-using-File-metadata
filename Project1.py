# TensorFlow and related imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense,Input
from tensorflow.keras.optimizers import Adam


# Data handling and manipulation
import numpy as np
import pandas as pd

# Visualization tools
import matplotlib.pyplot as plt
#import seaborn as sns

# scikit-learn (for data preprocessing, metrics, etc.)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

#import kagglehub

# Download latest version
#path = kagglehub.dataset_download("blueblushed/hospital-dataset-for-practice")
file_path="/Users/tanising/Downloads/hospital_data_analysis.csv"
df = pd.read_csv(file_path)
#print(df.head(10))
scaler = StandardScaler()

columns=["Procedure","Length_of_Stay"]
df_update=df[columns]
#print(df_update["Length_of_Stay"][0])
#unique_procedures = df_update['Procedure'].unique()
#print(type(unique_procedures))
output_df=df["Cost"]
#print(df_update)
updated_input=df_update.to_numpy()
updated_output=output_df.to_numpy()
print(updated_input[0])
procedures = updated_input[:, 0].reshape(-1, 1)
encoder = OneHotEncoder(sparse_output=False)  # sparse_output=False gives a NumPy array

encoded_procedures = encoder.fit_transform(procedures)
#print(encoded_procedures[1].shape)
#print(encoder.categories_)

input_encoded = np.hstack((updated_input[:, 1:], encoded_procedures))


X_train, X_test = train_test_split(input_encoded, test_size=0.2, random_state=42)
Y_train, Y_test = train_test_split(updated_output, test_size=0.2, random_state=42)
#print(Y_train.dtype)
  # Reshaping for OneHotEncoder
X_train = X_train.astype(np.float64)
X_test = X_test.astype(np.float64)
#print(X_train[0])

X_train[:, 0] = scaler.fit_transform(X_train[:, 0].reshape(-1, 1)).flatten()  # Assuming column 1 is 'days'
X_test[:, 0] = scaler.transform(X_test[:, 0].reshape(-1, 1)).flatten()
#print(X_train[0])
#print(X_train.shape)
#print(X_train[:10][0].dtype)

model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(64),  # Input layer (10 neurons)
    Dense(50),  # Hidden layer (8 neurons)
    Dense(1)  # Output layer (1 neuron)
])


model.compile(optimizer=Adam(learning_rate=0.0005), loss='mean_squared_error', metrics=['mse'])

history = model.fit(X_train, Y_train, epochs=1000, batch_size=32, validation_split=0.2, verbose=1)
predictions=model.predict(X_test)
print(predictions[98], "actual cost is",Y_test[98])
#loss, mse = model.evaluate(X_test, Y_test, verbose=1)
#print(f"\nFinal Model Performance:\nLoss: {loss}\nMean Squared Error: {mse}")


