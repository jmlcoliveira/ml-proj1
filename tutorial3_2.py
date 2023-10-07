import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

def load_bluegills_data(file_name):
    rows=[]
    lines = open(file_name).readlines()
    for line in lines[1:]:
        parts = line.split('\t')
        rows.append([float(parts[0]),float(parts[1])])
    return np.array(rows)

data = load_bluegills_data('bluegills.txt')
scaler = StandardScaler()
data=scaler.fit(data).transform(data)
train, test = train_test_split(data, test_size=0.5)
test, val = train_test_split(test, test_size=0.5)

results = []
plt.figure()
N=5
degree_labels = []
for i in range(0, N):
    degree=i+1 #degree of polynomial
    polyreg = make_pipeline(PolynomialFeatures(degree),LinearRegression())
    polyreg.fit(train[:,0].reshape(-1, 1),train[:,1].reshape(-1,1))
    
    #### validation error
    val_prediction = polyreg.predict(val[:,0].reshape(-1, 1)).reshape(-1,1)
    val_error_sum = 0
    for i in range(0, len(val_prediction)):
         val_error_sum = val_error_sum + (pow(val[i,1] - val_prediction[i], 2))
    val_error = (val_error_sum/len(val_prediction))
    ####################
    
    ##### test error
    test_prediction = polyreg.predict(test[:,0].reshape(-1, 1)).reshape(-1,1)
    test_error_sum = 0
    for i in range(0, len(test_prediction)):
         test_error_sum = test_error_sum + (pow(test[i,1] - test_prediction[i], 2))
    test_error = (test_error_sum/len(test_prediction))
    ####################
    X_seq = np.linspace(data[:,0].min(),data[:,0].max(),100).reshape(-1,1)
    c = np.random.rand(3,)
    plt.plot(X_seq,polyreg.predict(X_seq.reshape(-1,1)).reshape(-1,1),color=c, label=f'Degree {degree}/Val Error {val_error}/Test Error {test_error}')
    results.append([degree, val_error, test_error])
    degree_labels.append(f'Degree {degree}/Val Error {val_error}/Test Error {test_error}')
    
print(results)


plt.scatter(train[:,0],train[:,1],color='blue')
plt.scatter(test[:,0],test[:,1],color='green')
plt.scatter(val[:,0],val[:,1],color='red')
plt.xlabel('Age')
plt.ylabel('Lenght')
plt.legend(degree_labels)
plt.show()