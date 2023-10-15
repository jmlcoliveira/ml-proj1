import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import math
import pickle
import joblib
import matplotlib.pyplot as plt

train_data = pd.read_csv("X_train.csv")
X_data = []
Y_data = []
baseX = None

for line in train_data.values:
    # if line[0] == 0 and line[1] == 0 and line[2] == 0 and line[5] == 0 and line[6] == 0 and line[9] == 0 and line[10] == 0:
    #     continue
    
    b1_b2 = math.sqrt((line[1] - line[5])**2 + (line[2] - line[6])**2)
    b1_b3 = math.sqrt((line[1] - line[9])**2 + (line[2] - line[10])**2)
    b2_b3 = math.sqrt((line[5] - line[9])**2 + (line[6] - line[10])**2)
    Ox = (line[1] + line[5] + line[9]) / 3
    Oy = (line[2] + line[6] + line[10]) / 3
    if line[0] == 0:
        #            x1,     y1,      vx1        vy1     x2      y2      vx2         vy2     x3      y3          vx3     vy3
        # baseX = [line[1], line[2], line[3], line[4], line[5], line[6], line[7], line[8], line[9], line[10], line[11], line[12], b1_b2, b1_b3, b2_b3, Ox, Oy]
        baseX = [line[1], line[2], line[3], line[4], line[5], line[6], line[7], line[8], line[9], line[10], line[11], line[12], Ox, Oy]
    # X_data.append((line[0], baseX[0], baseX[1], baseX[2], baseX[3], baseX[4], baseX[5], baseX[6], baseX[7], baseX[8], baseX[9], baseX[10], baseX[11], baseX[12], baseX[13], baseX[14], baseX[15], baseX[16]))
    X_data.append((line[0], baseX[0], baseX[1], baseX[4], baseX[5], baseX[8], baseX[9], baseX[12], baseX[13]))
    distance = math.sqrt((Ox - baseX[12])**2 + (Oy - baseX[13])**2)
    # Y_data.append((line[1], line[2], line[3], line[4], line[5], line[6], line[7], line[8], line[9], line[10], line[11], line[12], b1_b2, b1_b3, b2_b3, distance))
    Y_data.append((line[1], line[2], line[5], line[6], line[9], line[10], distance))

X_data = np.array(X_data)
Y_data = np.array(Y_data)

# Combine and shuffle the batches
combined = list(zip(X_data, Y_data))
np.random.shuffle(combined)
X_train, Y_train = zip(*combined)

# Create scalers for X and Y separately
x_scaler = StandardScaler()
y_scaler = StandardScaler()

# Fit and transform the scalers on the training data
X_train = x_scaler.fit_transform(X_data)
Y_train = y_scaler.fit_transform(Y_data)

joblib.dump(x_scaler, 'x_scaler.pkl')  # Save the X scaler
joblib.dump(y_scaler, 'y_scaler.pkl')  # Save the Y scaler

# Split the data into training and validation sets
X_train, X_val, Y_train, Y_val = train_test_split(X_data, Y_data, test_size=0.5, shuffle=False)

BATCH_SIZE=257*100

count = 0

N = 10000
best_model = None
best_model_mse = 99
count = 0

x_train_batches = np.array_split(X_train, math.ceil(len(X_train) / BATCH_SIZE))
y_train_batches = np.array_split(Y_train, math.ceil(len(Y_train) / BATCH_SIZE))

def show_plot(prediction, Y_val):
    plt.figure()
    plt.scatter(Y_val[:,0], prediction[:,0], color='blue')
    plt.scatter(Y_val[:,1], prediction[:,1], color='green')
    plt.scatter(Y_val[:,2], prediction[:,2], color='red')
    plt.scatter(Y_val[:,3], prediction[:,3], color='yellow')
    plt.scatter(Y_val[:,4], prediction[:,4], color='purple')
    plt.scatter(Y_val[:,5], prediction[:,5], color='orange')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'Step {count} with Squared Mean Squared Error: {rmse}')
    plt.savefig(f'plots/step_{str(count).zfill(6)}_loss_{rmse}.png')
    #plt.show()
    plt.close()

for i in range(count, N):
    predictions = []
    degree = i + 1  # degree of polynomial
    #model = make_pipeline(PolynomialFeatures(degree), Ridge(alpha=0.95))
    model = make_pipeline(PolynomialFeatures(degree), KNeighborsRegressor(n_neighbors=3))
    #model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    for x_train, y_train in zip(x_train_batches, y_train_batches):
        model.fit(x_train, y_train)
            
        prediction = model.predict(X_val)
        rmse = math.sqrt(mean_squared_error(Y_val, prediction))
            
        if rmse < best_model_mse:
            show_plot(prediction, Y_val)
            best_model_mse = rmse
            best_model = model
            filename = f'models/best_model.plt'
            pickle.dump(model, open(filename, 'wb'))
            print(f"Step {count} with Mean Squared Error: {rmse}")
    
    # filename = f'models/checkpoint_step_{str(count).zfill(6)}_loss_{mse}.plt'
    # pickle.dump(model, open(filename, 'wb'))
    count += 1
