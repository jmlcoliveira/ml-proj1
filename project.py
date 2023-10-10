import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import random as r
import math
import pickle

train_data = pd.read_csv("X_train.csv")
# train_data = load_train_values('X_train.csv')
# test_data = pd.read_csv("X_test.csv")
# test_data = load_test_values('X_test.csv')
X_train = []
Y_train = []
baseX = None
values = train_data.values
# r.shuffle(values)

# scaler = StandardScaler()
# values=scaler.fit(values).transform(values)
# print(values.min(), values.max())

x_block = []
y_block = []
for line in values:
    if line[0] == 0:
        baseX = [line[0], line[1], line[2], line[5], line[6], line[9], line[10]]
    if line[0] == 10:
        X_train.append(x_block)
        x_block = []
        Y_train.append(y_block)
        y_block = []
    x_block.append([line[0], baseX[1], baseX[2], baseX[3], baseX[4], baseX[5], baseX[6]])
    y_block.append([line[1], line[2], line[5], line[6], line[9], line[10]])
    #X_train.append([line[0], baseX[1], baseX[2], baseX[3], baseX[4], baseX[5], baseX[6]])
    #Y_train.append([line[1], line[2], line[5], line[6], line[9], line[10]])

#plt.figure()
print(x_block[0], y_block[0])

# X_train = train_data[:, 1:]  # Features (columns 1 onwards)
# y_train = train_data[:, 0]   # Target variable (timestep, column 0)

# X_test = test_data[:, 1:]    # Features for test data
combined = list(zip(X_train, Y_train))
r.shuffle(combined)
X_train, Y_train = zip(*combined)

X_val = X_train[math.floor(len(X_train)*0.8):]
Y_val = Y_train[math.floor(len(Y_train)*0.8):]

X_train = X_train[:math.floor(len(X_train)*0.8)]
Y_train = Y_train[:math.floor(len(Y_train)*0.8)]

X_val = [item for sublist in X_val for item in sublist]
Y_val = [item for sublist in Y_val for item in sublist]

X_train = [item for sublist in X_train for item in sublist]
Y_train = [item for sublist in Y_train for item in sublist]

N=10000
degree_labels = []
best_model = None
best_model_mse = 99
count = 1
for i in range(0, N):
    degree=i+1 #degree of polynomial
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    # model = LinearRegression()
    model.fit(X_train, Y_train)
    y_pred = model.predict(X_val)
    mse = mean_squared_error(Y_val, y_pred)
    if mse < best_model_mse:
        best_model_mse = mse
        best_model = model
        filename = f'models/best_model.plt'
        pickle.dump(model, open(filename, 'wb'))
        filename = f'models/best_model_{str(count).zfill(6)}_loss_{mse}.plt'
        pickle.dump(model, open(filename, 'wb'))
    filename = f'models/checkpoint_step_{str(count).zfill(6)}_loss_{mse}.plt'
    pickle.dump(model, open(filename, 'wb'))
    count = count+1
    print(f"Step {count-1} with Mean Squared Error: {mse}")
    
    #TODO load by batches