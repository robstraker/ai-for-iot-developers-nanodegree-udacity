import cv2
import time
import numpy as np
import logging as log

from openvino.inference_engine import IECore


class ModelFaceDetection:
    """
    Class for the Face Detection Model.
    """

    def __init__(self, model_name, device="CPU", threshold=0.5, extensions=None):
        """
        Used to set all instance variables.
        """
        self.name = 'Face Detection'
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
            log.info("All layers are supported.")
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
        :return: box_bl, box_tr: Bottom left and top right corners of bounding box
        :return: image: Annotated image with intermediate outputs
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

        # Crop the frame to include only the detected bounding box
        coord, image = self.preprocess_output(result, image)

        self.total_process_time += time.time() - process_time
        self.index += 1

        return coord, image

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
                    self.extensions = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/" \
                                      "libcpu_extension_sse4.so"
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
        :param outputs: Initial outputs to process.
        :return: face_points: Coordinates of corners of face bounding box.
        :return: face_image: Cropped image of face to process.
        """
        output = outputs[self.output_name]
        h = image.shape[0]
        w = image.shape[1]

        # Define bounding box and cropped image for face
        for box in output[0][0]:
            if box[2] >= self.threshold:
                # Face detected above threshold
                x_min = int(box[3] * w)
                y_min = int(box[4] * h)
                x_max = int(box[5] * w)
                y_max = int(box[6] * h)
                face_image = image[y_min:y_max, x_min:x_max]
                face_points = [(x_min, y_min), (x_max, y_max)]
                return face_points, face_image
            else:
                # No face detected
                log.warning("No face detected, so read another frame")
                return None, None

    def stats(self):
        load = self.load_time
        proc = np.average(np.array(self.total_process_time, dtype=np.float))
        infer = np.average(np.array(self.total_infer_time, dtype=np.float))
        log.info(f"{self.name} Model: loading {load:.3}s, processing {proc:.3}s, inference {infer:.3}s")

