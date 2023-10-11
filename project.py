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
values = train_data.values

# Define batch size and initialize empty batch lists
x_block = []
y_block = []

# t_min = values[:, 0].min()
# t_max = values[:, 0].max()
# x1_min = values[:, 1].min()
# x1_max = values[:, 1].max()
# y1_min = values[:, 2].min()
# y1_max = values[:, 2].max()
# x2_min = values[:, 5].min()
# x2_max = values[:, 5].max()
# y2_min = values[:, 6].min()
# y2_max = values[:, 6].max()
# x3_min = values[:, 9].min()
# x3_max = values[:, 9].max()
# y3_min = values[:, 10].min()
# y3_max = values[:, 10].max()

# print(t_min, t_max, x1_min, x1_max, y1_min, y1_max, x2_min, x2_max, y2_min, y2_max, x3_min, x3_max, y3_min, y3_max)

for line in values:
    # t_normalized = (line[0] - t_min) / (t_max - t_min)
    # x1_normalized = (line[1] - x1_min) / (x1_max - x1_min)
    # y1_normalized = (line[2] - y1_min) / (y1_max - y1_min)
    # x2_normalized = (line[5] - x2_min) / (x2_max - x2_min)
    # y2_normalized = (line[6] - y2_min) / (y2_max - y2_min)
    # x3_normalized = (line[9] - x3_min) / (x3_max - x3_min)
    # y3_normalized = (line[10] - y3_min) / (y3_max - y3_min)
    
    #TODO: add colisison tests
    
    if line[0] == 0:
        baseX = [line[0], line[1], line[2], line[5], line[6], line[9], line[10]]
    X_train.append((line[0], baseX[1], baseX[2], baseX[3], baseX[4], baseX[5], baseX[6]))
    Y_train.append((line[1], line[2], line[5], line[6], line[9], line[10]))
    
#        baseX = [t_normalized, x1_normalized, y1_normalized, x2_normalized, y2_normalized, x3_normalized, y3_normalized]
#    x_block.append([t_normalized, baseX[1], baseX[2], baseX[3], baseX[4], baseX[5], baseX[6]])
#    y_block.append([t_normalized, x1_normalized, y1_normalized, x2_normalized, y2_normalized, x3_normalized, y3_normalized])

    # if line[0] == 10:
    #     X_train.append(x_block)
    #     Y_train.append(y_block)
    #     x_block = []
    #     y_block = []
    #     baseX = []

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
print(X_train[0], Y_train[0], X_val[0], Y_val[0])

# Split data into training and validation sets
# split_idx = math.floor(len(X_train) * 0.8)
# X_val = X_train[split_idx:]
# Y_val = Y_train[split_idx:]
# X_train = X_train[:split_idx]
# Y_train = Y_train[:split_idx]

# Flatten the data within each batch
batch_x=[]
batch_y=[]
temp_x_train=[]
temp_y_train=[]

BATCH_SIZE=5000

count = 0
#scaler = StandardScaler()
#X_val=scaler.fit(X_val).transform(X_val)
#Y_val=scaler.fit(Y_val).transform(Y_val)
# for i in range(len(X_train)):
#     if count % BATCH_SIZE == 0:
#         batch_x=[item for sublist in batch_x for item in sublist]
#         temp_x_train.append(batch_x)
#         batch_x=[]
#         batch_y=[item for sublist in batch_y for item in sublist]
#         temp_y_train.append(batch_y)
#         batch_y=[]
#     count = count + 1
#     batch_x.append(X_train[i])
#     batch_y.append(Y_train[i])
    
# batch_x=[item for sublist in batch_x for item in sublist]
# temp_x_train.append(batch_x)
# batch_y=[item for sublist in batch_y for item in sublist]
# temp_y_train.append(batch_y)

# X_train = temp_x_train[1:]
# Y_train = temp_y_train[1:]
    
# X_val = [item for sublist in X_val for item in sublist]
# Y_val = [item for sublist in Y_val for item in sublist]

#scaler = StandardScaler()
#X_val=scaler.fit(X_val).transform(X_val)
#Y_val=scaler.fit(Y_val).transform(Y_val)

#print(X_train)

N = 10000
best_model = None
best_model_mse = 99
count = 1

for i in range(N):
    predictions = []
    degree = i + 1  # degree of polynomial
    model = make_pipeline(PolynomialFeatures(degree), Ridge(alpha=0.99))
    #model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    #for batch_x, batch_y in zip(X_train, Y_train):
        #model.fit(batch_x, batch_y)
    model.fit(X_train, Y_train)
        
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
