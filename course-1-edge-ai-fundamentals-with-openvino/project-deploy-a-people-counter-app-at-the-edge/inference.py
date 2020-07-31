#!/usr/bin/env python3
"""
 Copyright (c) 2018 Intel Corporation.

 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
 the following conditions:

 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import os
import sys
import logging as log
from openvino.inference_engine import IENetwork, IECore


class Network:
    """
    Load and configure inference plugins for the specified target devices 
    and performs synchronous and asynchronous modes for the specified infer requests.
    """

    def __init__(self, model, device, prob_threshold, cpu_extension):
        ### TODO: Initialize any class variables desired ###
        self.model_weights=os.path.splitext(model)[0]+'.bin'
        self.model_structure=model
        self.device=device
        self.threshold=prob_threshold
        self.extension=cpu_extension
        
        try:
            self.model=IENetwork(self.model_structure, self.model_weights)
        except Exception as e:
            raise ValueError("Could not Initialise the network. Have you enterred the correct model path?")
        
        self.input_name = next(iter(self.model.inputs))
        self.input_shape = self.model.inputs[self.input_name].shape
        self.output_name = next(iter(self.model.outputs))
        self.output_shape = self.model.outputs[self.output_name].shape
        
        self.exec_network = None
        self.infer_request = None

    def load_model(self):
        ### TODO: Load the model ###
        log.warning("Loading model ...") 
        self.plugin = IECore()
        
        ### TODO: Check for supported layers ###
        # Get the supported layers of the network
        
        if 'CPU' in self.device:
            supported_layers = self.plugin.query_network(self.model, 'CPU')
            
            # Let user know if anything is missing. Exit the program, if so.
            unsupported_layers = [l for l in self.model.layers.keys() if l not in supported_layers]
            
            if len(unsupported_layers) != 0:
                log.warning("Unsupported layers found: {}".format(unsupported_layers))
                log.warning("Check whether extensions are available to add to IECore.")
                #sys.exit(1)
        
        ### TODO: Add any necessary extensions ###
        if self.extension and 'CPU' in self.device:
            self.plugin.add_extension(self.extension, self.device)
        
        ### TODO: Return the loaded inference plugin ###
        ### Note: You may need to update the function parameters. ###
        self.exec_network = self.plugin.load_network(self.model, self.device)
        return

    def get_input_shape(self):
        ### TODO: Return the shape of the input layer ###
        return self.input_shape

    def exec_net(self, image):
        ### TODO: Start an asynchronous request ###
        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        self.exec_network.start_async(request_id=0, inputs={self.input_name: image})
        return
    
    def wait(self):
        ### TODO: Wait for the request to be complete. ###
        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        status = self.exec_network.requests[0].wait(-1)
        return status

    def get_output(self):
        ### TODO: Extract and return the output results
        ### Note: You may need to update the function parameters. ###
        return self.exec_network.requests[0].outputs[self.output_name]
    
    def clean(self):
        ### Deletes all the instances
        del self.model
        del self.plugin
        del self.exec_network