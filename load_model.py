
import pickle

# load the model from disk
loaded_model = pickle.load(open('models/best_model.plt', 'rb'))
result = loaded_model.predict(X_test, Y_test)
print(result)