import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import math
import pickle
import joblib

train_data = pd.read_csv("X_train.csv")
X_train = []
Y_train = []
baseX = None

for line in train_data.values:
   
    if line[0] == 0:
        baseX = [line[0], line[1], line[2], line[5], line[6], line[9], line[10]]
    X_train.append((line[0], baseX[1], baseX[2], baseX[3], baseX[4], baseX[5], baseX[6]))
    Y_train.append((line[1], line[2], line[5], line[6], line[9], line[10]))

X_train = np.array(X_train)
Y_train = np.array(Y_train)

# Combine and shuffle the batches
combined = list(zip(X_train, Y_train))
np.random.shuffle(combined)
X_train, Y_train = zip(*combined)

# Create scalers for X and Y separately
x_scaler = StandardScaler()
y_scaler = StandardScaler()

# Fit and transform the scalers on the training data
X_train = x_scaler.fit_transform(X_train)
Y_train = y_scaler.fit_transform(Y_train)

joblib.dump(x_scaler, 'x_scaler.pkl')  # Save the X scaler
joblib.dump(y_scaler, 'y_scaler.pkl')  # Save the Y scaler

# Split the data into training and validation sets
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.3)
print(len(X_train), len(Y_train), len(X_val), len(Y_val))

BATCH_SIZE=100000

count = 0

N = 10000
best_model = None
best_model_mse = 99
count = 1

x_train_batches = np.array_split(X_train, math.ceil(len(X_train) / BATCH_SIZE))
y_train_batches = np.array_split(Y_train, math.ceil(len(Y_train) / BATCH_SIZE))

for i in range(N):
    predictions = []
    degree = i + 1  # degree of polynomial
    model = make_pipeline(PolynomialFeatures(degree), Ridge(alpha=0.95))
    #model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    for x_train, y_train in zip(x_train_batches, y_train_batches):
        model.fit(x_train, y_train)
            
        prediction = model.predict(X_val)
        mse = mean_squared_error(Y_val, prediction)
    
        if mse < best_model_mse:
            best_model_mse = mse
            best_model = model
            filename = f'models/best_model.plt'
            pickle.dump(model, open(filename, 'wb'))
                    
        print(f"Step {count-1} with Mean Squared Error: {mse}")
    
    # filename = f'models/checkpoint_step_{str(count).zfill(6)}_loss_{mse}.plt'
    # pickle.dump(model, open(filename, 'wb'))
    count += 1
