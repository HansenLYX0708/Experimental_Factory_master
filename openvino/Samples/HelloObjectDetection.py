import cv2
import matplotlib.pyplot as plt
import numpy as np
from openvino.runtime import Core

ie = Core()

model = ie.read_model(model="model/horizontal-text-detection-0001.xml")
compiled_model = ie.compile_model(model=model, device_name="CPU")

input_layer_ir = next(iter(compiled_model.inputs))
