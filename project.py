import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

def load_train_values(file_name):
    rows=[]
    lines = open(file_name).readlines()
    for line in lines[1:]:
        parts = line.split(',')
        rows.append([
                    float(parts[0]), #timestep
                    float(parts[1]), #x1
                    float(parts[2]), #y1
                    float(parts[3]), #vx1
                    float(parts[4]), #vy1
                    float(parts[5]), #x2
                    float(parts[6]), #y2
                    float(parts[7]), #vx2
                    float(parts[8]), #vy2
                    float(parts[9]), #x3
                    float(parts[10]), #y3
                    float(parts[11]), #vx3
                    float(parts[12]), #vy3
                     ])
    return np.array(rows)

def load_test_values(file_name):
    rows=[]
    lines = open(file_name).readlines()
    for line in lines[1:]:
        parts = line.split(',')
        rows.append([float(parts[0]), #timestep
                     float(parts[1]), #x1
                     float(parts[2]), #y1
                     float(parts[3]), #x2
                     float(parts[4]), #y2
                     float(parts[5]), #x3
                     float(parts[6])  #y3
                     ])
    return np.array(rows)

train_data = pd.read_csv("X_train.csv")
# train_data = load_train_values('X_train.csv')
# test_data = pd.read_csv("X_test.csv")
# test_data = load_test_values('X_test.csv')
X_train = []
Y_train = []
baseX = None
for line in train_data.values:
    if line[0] == 0:
        baseX = [line[0], line[1], line[2], line[5], line[6], line[9], line[10]]
    X_train.append([line[0], baseX[1], baseX[2], baseX[3], baseX[4], baseX[5], baseX[6]])
    Y_train.append([line[1], line[2], line[5], line[6], line[9], line[10]])

plt.figure()
scaler = StandardScaler()
train_data=scaler.fit(train_data).transform(train_data)
print(train_data.min(), train_data.max())

# X_train = train_data[:, 1:]  # Features (columns 1 onwards)
# y_train = train_data[:, 0]   # Target variable (timestep, column 0)

# X_test = test_data[:, 1:]    # Features for test data

model = LinearRegression()
model.fit(X_train, Y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")