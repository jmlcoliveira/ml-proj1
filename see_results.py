
import pandas as pd
import pickle
import numpy as np
import joblib
import math
import matplotlib.pyplot as plt


NR_TEST_SAMPLES = 257
arr=[]
X_train = pd.read_csv("X_train.csv")
path = "models/_potential_models-2023-10-16_14-56-02"
loaded_model = pickle.load(open(f'{path}/model_1.plt', 'rb'))
USE_SCALLER = False

if USE_SCALLER:
    loaded_x_scaler = joblib.load(f'{path}/x_scaler.pkl')
    loaded_y_scaler = joblib.load(f'{path}/y_scaler.pkl')

for line in X_train.values:
    first_row = True
    Ox = (line[1] + line[5] + line[9]) / 3
    Oy = (line[2] + line[6] + line[10]) / 3
    if first_row:
        baseX = [f for f in line]
        first_row = False
        baseX.append(Ox)
        baseX.append(Oy)
        
    b1_b2 = math.sqrt((line[1] - line[5])**2 + (line[2] - line[6])**2)
    b1_b3 = math.sqrt((line[1] - line[9])**2 + (line[2] - line[10])**2)
    b2_b3 = math.sqrt((line[5] - line[9])**2 + (line[6] - line[10])**2)
    
    distance = math.sqrt((Ox - baseX[13])**2 + (Oy - baseX[14])**2)
    
    # Using the Law of Cosines to calculate the internal angles
    # Angle at b1
    try:
        cos_angle_b1 = (b1_b2**2 + b1_b3**2 - b2_b3**2) / (2 * b1_b2 * b1_b3)
        angle_b1 = math.degrees(math.acos(cos_angle_b1))
    except:
        angle_b1=0

    # Angle at b2
    try:
        cos_angle_b2 = (b1_b2**2 + b2_b3**2 - b1_b3**2) / (2 * b1_b2 * b2_b3)
        angle_b2 = math.degrees(math.acos(cos_angle_b2))
    except:
        angle_b2=0
        
    # Angle at b3
    try:
        cos_angle_b3 = (b1_b3**2 + b2_b3**2 - b1_b2**2) / (2 * b1_b3 * b2_b3)
        angle_b3 = math.degrees(math.acos(cos_angle_b3))
    except:
        angle_b3=0
    
    # arr.append([line[0], line[1], line[2], line[5], line[6], line[9], line[10], b1_b2, b1_b3, b2_b3, Ox, Oy, angle_b1, angle_b2, angle_b3])
    arr.append([line[0], line[1], line[2], line[5], line[6], line[9], line[10], b1_b2, b1_b3, b2_b3])
    # arr.append([line[0],  b1_b2, b1_b3, b2_b3, Ox, Oy])

tests = np.array_split(arr, math.ceil(len(arr) / NR_TEST_SAMPLES))

for test in tests:
    if USE_SCALLER:
        predictions = loaded_model.predict(loaded_x_scaler.transform(test))
            
        result = loaded_y_scaler.inverse_transform(predictions)
    else:
        predictions = loaded_model.predict(test)
        
        result = loaded_model.predict(test)
        
       
    plt.figure()    
    plt.scatter(test[:,1], test[:,2], color='blue')
    plt.scatter(test[:,3], test[:,4], color='red')
    plt.scatter(test[:,5], test[:,6], color='purple')

    plt.scatter(result[:,0], result[:,1], color='green')
    plt.scatter(result[:,2], result[:,3], color='yellow')
    plt.scatter(result[:,4], result[:,5], color='orange')
    plt.show()
    
    # Wait for the user to close the plot window
    while True:
        if plt.get_fignums():  # Check if any figure windows are open
            plt.waitforbuttonpress()
        else:
            break  # Exit the loop when the window is closed