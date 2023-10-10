import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
import math
import pickle

train_data = pd.read_csv("X_train.csv")
X_train = []
Y_train = []
baseX = None
values = train_data.values

# Define batch size and initialize empty batch lists
x_block = []
y_block = []

for line in values:
    if line[0] == 0:
        baseX = [line[0], line[1], line[2], line[5], line[6], line[9], line[10]]
    x_block.append([line[0], baseX[1], baseX[2], baseX[3], baseX[4], baseX[5], baseX[6]])
    y_block.append([line[1], line[2], line[5], line[6], line[9], line[10]])

    if line[0] == 10:
        X_train.append(x_block)
        Y_train.append(y_block)
        x_block = []
        y_block = []

# Combine and shuffle the batches
combined = list(zip(X_train, Y_train))
np.random.shuffle(combined)
X_train, Y_train = zip(*combined)

# Split data into training and validation sets
split_idx = math.floor(len(X_train) * 0.8)
X_val = X_train[split_idx:]
Y_val = Y_train[split_idx:]
X_train = X_train[:split_idx]
Y_train = Y_train[:split_idx]

# Flatten the data within each batch
batch_x=[]
batch_y=[]
temp_x_train=[]
temp_y_train=[]
BATCH_SIZE=512
count = 0
for i in range(len(X_train)):
    if count % BATCH_SIZE == 0:
        batch_x=[item for sublist in batch_x for item in sublist]
        temp_x_train.append(batch_x)
        batch_x=[]
        batch_y=[item for sublist in batch_y for item in sublist]
        temp_y_train.append(batch_y)
        batch_y=[]
    count = count + 1
    batch_x.append(X_train[i])
    batch_y.append(Y_train[i])
    
batch_x=[item for sublist in batch_x for item in sublist]
temp_x_train.append(batch_x)
batch_y=[item for sublist in batch_y for item in sublist]
temp_y_train.append(batch_y)

X_train = temp_x_train[1:]
Y_train = temp_y_train[1:]
    
X_val = [item for sublist in X_val for item in sublist]
Y_val = [item for sublist in Y_val for item in sublist]

N = 10000
best_model = None
best_model_mse = 99
count = 1

print(len(X_train), len(Y_train))

for i in range(N):
    predictions = []
    degree = i + 1  # degree of polynomial
    model = make_pipeline(PolynomialFeatures(degree), Ridge(alpha=0.9, normalize=True))
    #model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    for batch_x, batch_y in zip(X_train, Y_train):
        model.fit(batch_x, batch_y)
        predictions.append(model.predict(X_val))
    #model.fit(X_train, Y_train)
    #y_pred = model.predict(X_val)
    final_prediction = np.mean(predictions, axis=0)
    mse = mean_squared_error(Y_val, final_prediction)
    
    if mse < best_model_mse:
        best_model_mse = mse
        best_model = model
        filename = f'models/best_model.plt'
        pickle.dump(model, open(filename, 'wb'))
        filename = f'models/best_model_{str(count).zfill(6)}_loss_{mse}.plt'
        pickle.dump(model, open(filename, 'wb'))
    
    filename = f'models/checkpoint_step_{str(count).zfill(6)}_loss_{mse}.plt'
    pickle.dump(model, open(filename, 'wb'))
    count += 1
    print(f"Step {count-1} with Mean Squared Error: {mse}")
