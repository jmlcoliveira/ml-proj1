import pandas as pd
import pickle
import numpy as np

# load the model from disk
loaded_model = pickle.load(open('best_model copy.plt', 'rb'))
X_test = pd.read_csv("X_test.csv")
arr = []
for line in X_test.values:
    arr.append([line[1], line[2], line[3], line[4], line[5], line[6], line[7]])
result = loaded_model.predict(arr)

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