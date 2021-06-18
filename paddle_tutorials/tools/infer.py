import numpy as np
import PIL.Image
import paddle.inference as paddle_infer
import skimage.io as io

image_name = "C:/data/SliderSN/SliderSN/train/0/sn_0_17.bmp"
model_file = "C:/model_outputs/cnn_sn/static_model/test.pdmodel"
params_file = "C:/model_outputs/cnn_sn/static_model/test.pdiparams"
batch_size = 1

config = paddle_infer.Config(model_file, params_file)
predictor = paddle_infer.create_predictor(config)

input_names = predictor.get_input_names()
input_handle = predictor.get_input_handle(input_names[0])

### fake data
#fake_input = np.random.randn(batch_size, 1, 40 , 24).astype('float32')

model_input = io.imread(image_name)
model_input = np.asarray(model_input).astype('float32')
model_input = np.expand_dims(model_input, 0)
model_input = np.expand_dims(model_input, 0)
model_input = model_input / 255.0

input_handle.reshape([batch_size, 1, 40, 24])
input_handle.copy_from_cpu(model_input)

predictor.run()

output_names = predictor.get_output_names()
output_handle = predictor.get_output_handle(output_names[0])
output_data = output_handle.copy_to_cpu()
print(output_data)
print("end")
