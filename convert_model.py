import pickle
from skl2onnx import convert
from skl2onnx.common.data_types import FloatTensorType
from onnx import save_model

path = "models/_potential_models-2023-10-16_14-56-02"

with open(f'{path}/model_1.plt', 'rb') as model_file:
    scikit_learn_model = pickle.load(model_file)
    
initial_type = [("x_values", FloatTensorType([None, 10]))]
onnx_model = convert.convert_sklearn(scikit_learn_model, initial_types=initial_type)

save_model(onnx_model, f'{path}/scikit_learn_model.onnx')