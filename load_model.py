import pandas as pd
import pickle
import numpy as np
import joblib
import math

def flatten_comprehension(matrix):
     return [item for row in matrix for item in row]

# load the model from disk
loaded_model = pickle.load(open('models/best_model.plt', 'rb'))
X_test = pd.read_csv("X_test.csv")
loaded_x_scaler = joblib.load('x_scaler.pkl')
loaded_y_scaler = joblib.load('y_scaler.pkl')
BATCH_SIZE=300000

arr = []
for line in X_test.values:
    b1_b2 = math.sqrt((line[2] - line[4])**2 + (line[3] - line[5])**2)
    b1_b3 = math.sqrt((line[2] - line[6])**2 + (line[3] - line[7])**2)
    b2_b3 = math.sqrt((line[4] - line[6])**2 + (line[5] - line[7])**2)
    Ox = (line[2] + line[4] + line[6]) / 3
    Oy = (line[3] + line[5] + line[7]) / 3
    arr.append([line[1], line[2], line[3], 0.0, 0.0, line[4], line[5], 0.0, 0.0, line[6], line[7], 0.0, 0.0, b1_b2, b1_b3, b2_b3, Ox, Oy])
    
arr = loaded_x_scaler.transform(arr)
x_test_batches = np.array_split(arr, math.ceil(len(arr) / BATCH_SIZE))
result = []
for x_test in x_test_batches:
    prediction = loaded_model.predict(x_test)
    for i in range(0, len(prediction)):
        result.append(prediction[i])
        


#result = loaded_model.predict(arr)
Y_test_inverted = loaded_y_scaler.inverse_transform(result)
columns_to_save = [0, 1, 4, 5, 8, 9]
result = Y_test_inverted[:, columns_to_save]
result = pd.DataFrame(result)

# Define custom headers
custom_headers = ["x_1",
                  "y_1",
                  "x_2",
                  "y_2",
                  "x_3",
                  "y_3"]

# Save the DataFrame to a CSV file with custom headers
result.to_csv("result.csv", index=True, header=custom_headers, index_label="Id")