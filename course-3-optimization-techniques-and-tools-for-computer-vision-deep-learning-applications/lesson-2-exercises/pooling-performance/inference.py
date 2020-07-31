from openvino.inference_engine import IENetwork, IECore
from openvino.inference_engine import IEPlugin

import numpy as np
import time

# Getting model bin and xml file
model_path='pool_cnn/pool_cnn'
model_weights=model_path+'.bin'
model_structure=model_path+'.xml'

# TODO: Load the model
# Use either IECore or IEPlugin API

# Reading and Preprocessing Image
input_img=np.load('image.npy')
input_img=input_img.reshape(1, 28, 28)


# TODO: Using the input image, run inference on the model for 10 iterations


# TODO: Finish the print statement
print("Time taken to run 10 iterations is: ")