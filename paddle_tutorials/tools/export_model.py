import numpy as np
import paddle
from paddle.static import save_inference_model
from paddle_tutorials.modeling.models import MyModel
from paddle_tutorials.utils.save_load import init_model

img_path = 'C:/data/SliderSN/SliderSN_mnist/sn_x_train.npy'
label_path = 'C:/data/SliderSN/SliderSN_mnist/sn_y_train.npy'
checkpoints = "C:/model_outputs/cnn_sn/model_save/9"

model = MyModel()
optim = paddle.optimizer.Adam(learning_rate=1e-3, parameters=model.parameters())
init_model(model, checkpoints=checkpoints, opt=optim)

in_np = np.random.random([1, 1, 40, 24]).astype('float32')
input_var = paddle.to_tensor(in_np)


path = "C:/model_outputs/cnn_sn/static_model/test"
paddle.jit.save(model, path, input_spec=[input_var])