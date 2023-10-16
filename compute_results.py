import pandas as pd
import pickle
import numpy as np
import math

def flatten_comprehension(matrix):
     return [item for row in matrix for item in row]

# load the model from disk
path = "models/_potential_models-2023-10-16_14-56-02"
loaded_model = pickle.load(open(f'{path}/model_1.plt', 'rb'))

X_test = pd.read_csv("X_test.csv")

BATCH_SIZE=300000

arr = []
for line in X_test.values:
    b1_b2 = math.sqrt((line[2] - line[4])**2 + (line[3] - line[5])**2)
    b1_b3 = math.sqrt((line[2] - line[6])**2 + (line[3] - line[7])**2)
    b2_b3 = math.sqrt((line[4] - line[6])**2 + (line[5] - line[7])**2)

    arr.append([line[1], line[2], line[3], line[4], line[5], line[6], line[7], b1_b2, b1_b3, b2_b3])
    
x_test_batches = np.array_split(arr, math.ceil(len(arr) / BATCH_SIZE))
result_all = []
for x_test in x_test_batches:
    prediction = loaded_model.predict(x_test)
    for i in range(0, len(prediction)):
        result_all.append(prediction[i])

result_all = np.array(result_all)
columns_to_save = slice(0, 6)
result = result_all[:, columns_to_save]
result = pd.DataFrame(result)

# Define custom headers
custom_headers = ["x_1",
                  "y_1",
                  "x_2",
                  "y_2",
                  "x_3",
                  "y_3"]

# Save the DataFrame to a CSV file with custom headers
result.to_csv(f"{path}/result.csv", index=True, header=custom_headers, index_label="Id")