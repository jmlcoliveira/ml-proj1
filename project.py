import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import math
import pickle
import matplotlib.pyplot as plt
from datetime import datetime
import os
import shutil

# Read the train data
train_data = pd.read_csv("X_train.csv")
X_data = []
Y_data = []
baseX = None

# Get the current time
current_time = datetime.now()
# Format it as a string with seconds
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

FOLDER_NAME = f'models/models-{current_time}'

source_file = 'project.py'

print(f"Saving to {FOLDER_NAME}")

os.makedirs(os.path.join(os.getcwd(), FOLDER_NAME), exist_ok=True)

shutil.copy(source_file, FOLDER_NAME)

for line in train_data.values:
    if line[0] == 0 and line[1] == 0 and line[2] == 0 and line[5] == 0 and line[6] == 0 and line[9] == 0 and line[10] == 0:
        continue
    
    b1_b2 = math.sqrt((line[1] - line[5])**2 + (line[2] - line[6])**2)
    b1_b3 = math.sqrt((line[1] - line[9])**2 + (line[2] - line[10])**2)
    b2_b3 = math.sqrt((line[5] - line[9])**2 + (line[6] - line[10])**2)

    if line[0] == 0:
        baseX = [line[1], line[2], line[5], line[6], line[9], line[10], b1_b2, b1_b3, b2_b3]

    X_data.append((line[0], baseX[0], baseX[1], baseX[2], baseX[3], baseX[4], baseX[5], baseX[6], baseX[7], baseX[8]))
    
    Y_data.append((line[1], line[2], line[5], line[6], line[9], line[10], b1_b2, b1_b3, b2_b3))
    
X_data = np.array(X_data)
Y_data = np.array(Y_data)

# Combine and shuffle the batches
combined = list(zip(X_data, Y_data))
np.random.shuffle(combined)
X_data, Y_data = zip(*combined)

# Split the data into training and validation sets
X_train, X_val, Y_train, Y_val = train_test_split(X_data, Y_data, test_size=0.5, shuffle=False)

BATCH_SIZE=257*1000

best_model = None
best_model_mse = 99
START_DEGREE = 1
FINAL_DEGREE = 10
model_count = 0
x_train_batches = np.array_split(X_train, math.ceil(len(X_train) / BATCH_SIZE))
y_train_batches = np.array_split(Y_train, math.ceil(len(Y_train) / BATCH_SIZE))

def show_plot(prediction, Y_val, i):
    plt.figure()
    plt.scatter(Y_val[:,0], prediction[:,0], color='blue')
    plt.scatter(Y_val[:,1], prediction[:,1], color='green')
    plt.scatter(Y_val[:,2], prediction[:,2], color='red')
    plt.scatter(Y_val[:,3], prediction[:,3], color='yellow')
    plt.scatter(Y_val[:,4], prediction[:,4], color='purple')
    plt.scatter(Y_val[:,5], prediction[:,5], color='orange')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'Step {i} with Squared Mean Squared Error: {rmse}')
    plt.savefig(f'{FOLDER_NAME}/step_{str(i).zfill(6)}_loss_{rmse}.png')
    #plt.show()
    plt.close()

for i in range(START_DEGREE, FINAL_DEGREE):
    predictions = []
    degree = i
    model = make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=10))
    for x_train, y_train in zip(x_train_batches, y_train_batches):
        model.fit(x_train, y_train)
            
        prediction = model.predict(X_val)
        Y_val = np.array(Y_val)
        prediction = np.array(prediction)
        rmse = math.sqrt(mean_squared_error(Y_val[:,[0, 1, 2, 3, 4, 5]], prediction[:,[0, 1, 2, 3, 4, 5]]))
        
        if rmse < best_model_mse:
            best_model_mse = rmse
            best_model = model
            filename = f'{FOLDER_NAME}/model_{model_count}.plt'
            model_count = model_count + 1
            pickle.dump(model, open(filename, 'wb'))
            print(f"Step {i} with Mean Squared Error: {rmse}")
        else:
            print(f"No improvemnt -> Step {i} with Mean Squared Error: {rmse}")
            
        show_plot(prediction, Y_val, i)
        
        filename = f'{FOLDER_NAME}/model_{rmse}.plt'
        pickle.dump(model, open(filename, 'wb'))