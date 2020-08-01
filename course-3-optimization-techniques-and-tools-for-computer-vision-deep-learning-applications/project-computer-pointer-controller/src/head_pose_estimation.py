"""
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
"""

import cv2
import logging as log
from openvino.inference_engine import IECore
import time
import numpy as np


class ModelHeadPoseEstimation:
    """
    Class for the Head Pose Estimation Model.
    """

    def __init__(self, model_name, device="CPU", threshold=0.5, extensions=None):
        """
        Used to set all instance variables.
        """
        self.name = "Head Pose Estimation"
        self.model_weights = model_name + ".bin"
        self.model_structure = model_name + ".xml"
        self.device = device
        self.threshold = threshold
        self.extensions = extensions

        self.plugin = IECore()
        try:
            self.model = self.plugin.read_network(self.model_structure, self.model_weights)
        except Exception as e:
            raise ValueError("Could not initialize network. Check model path.")

        self.input_name = next(iter(self.model.inputs))
        self.input_shape = self.model.inputs[self.input_name].shape
        self.output_name = next(iter(self.model.outputs))
        self.output_shape = self.model.outputs[self.output_name].shape
        self.network = None
        self.exec_network = None

        self.load_time = 0.0
        self.infer_time = 0.0
        self.process_time = 0.0
        self.total_infer_time = 0.0
        self.total_process_time = 0.0
        self.index = 1

    def load_model(self):
        """
        This method is for loading the model to the device specified by the user.
        If layers are not supported, appropriate plugins are loaded.
        :return: None
        """
        # Check supported layers
        if self.check_model():
            log.info('All layers are supported.')
        else:
            log.warning("Adding extension.")
            self.plugin.add_extension(self.extensions, self.device)

        # Load the IENetwork into the plugin
        self.load_time = time.time()
        self.exec_network = self.plugin.load_network(network=self.model, device_name=self.device, num_requests=2)
        self.load_time = time.time() - self.load_time

        return

    def predict(self, image):
        """
        This method is meant for running predictions on the input image.
        :param image: Image input to process.
        :return: tbd
        """
        process_time = time.time()
        # Preprocess the input image
        p_image = self.preprocess_input(image)

        # Make an asynchronous inference request, given an input image
        infer_time = time.time()
        infer_request = self.exec_network.start_async(request_id=0, inputs={self.input_name: p_image})

        if infer_request.wait() == 0:
            # Get the output of the inference
            result = infer_request.outputs
        self.total_infer_time += time.time() - infer_time

        # Identify the the orientation of the head pose
        coords = self.preprocess_output(result, image)

        self.total_process_time += (time.time() - process_time)
        self.index += 1

        return coords

    def check_model(self):
        """
        This method checks for plugins if there are unsupported layers.
        :return: True: If all layers supported, False if there are some unsupported layers.
        """
        # Check for supported layers
        if "CPU" in self.device:
            supported_layers = self.plugin.query_network(self.model, "CPU")
            unsupported_layers = [l for l in self.model.layers.keys() if l not in supported_layers]

            if unsupported_layers:
                log.warning("Unsupported layers found: {}".format(unsupported_layers))
                if not self.extensions:
                    ext_path = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/\
                                libcpu_extension_sse4.so "
                    self.extensions = ext_path
                return False

        return True

    def preprocess_input(self, image):
        """
        This method pre-processes the input image before feeding the data into the model for inference.
        :param image: Initial image to process.
        :return: p_image: Pre-processed image.
        """
        p_image = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        p_image = p_image.transpose((2, 0, 1))
        p_image = p_image.reshape(1, *p_image.shape)

        return p_image

    def preprocess_output(self, outputs, image):
        """
        This method pre-processes the outputs before feeding the output of this model to the next model.
        :return: vec_coords: Coordinates of head pose direction vector.
        """
        yaw = outputs["angle_y_fc"][0, 0]
        pitch = outputs["angle_p_fc"][0, 0]
        roll = outputs["angle_r_fc"][0, 0]
        vector_coords = [yaw, pitch, roll]

        return vector_coords

    def stats(self, header=True):
        load = self.load_time
        proc = np.average(np.array(self.total_process_time, dtype=np.float))
        infer = np.average(np.array(self.total_infer_time, dtype=np.float))
        log.info(f"{self.name} Model: loading {load:.3}s, processing {proc:.3}s, inference {infer:.3}s")
