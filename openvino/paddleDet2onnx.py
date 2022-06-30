import paddle2onnx
import paddle

from openvino_ppdet import nms_mapper

model_prefix = "C:/GitWorkspace/Experimental_Factory_master/openvino/inference/ttfnet_darknet53_1x_coco/model"
model = paddle.jit.load(model_prefix)
input_shape_dict = {
    "image":[1,3,640,640],

}

onnx_model = paddle2onnx.run_convert(model, input_shape_dict=input_shape_dict, opset_version=11)

with open("./ttfnet.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())