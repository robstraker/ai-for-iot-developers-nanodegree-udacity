"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
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
import time
import socket
import json
import cv2

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser


def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
    return client


def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # Initialise the class  
    infer_network = Network(args.model, args.device, args.prob_threshold, args.cpu_extension)

    ### TODO: Load the model through `infer_network` ###
    start_load = time.time()
    infer_network.load_model()
    log.warning("Time to load model is {}!".format(round(time.time() - start_load, 4)))
    
    net_input_shape = infer_network.get_input_shape()
    
    ### TODO: Handle the input stream ###
    single_image_mode = False
    media_file = args.input
    
    # Check if input is a live feed
    if media_file =='CAM':
        input_stream = 0
    
    # Check if the input is an image
    elif media_file.endswith('.jpg') or media_file.endswith('.bmp'):
        single_image_mode = True
        input_stream = media_file
    
    # Check if the input is a video
    else:
        input_stream = media_file
    try:
        cap = cv2.VideoCapture(input_stream)
    except FileNotFoundError:
        log.exception("Cannot locate video file: {}".format(input_stream))
    except Exception as e:
        log.exception("Something else went wrong with the video file: {}".format(e))
    
    if input_stream:
        cap.open(input_stream)
    
    ### TODO: Loop until stream is over ###
    # Initialize person count and duration variables
    current_count = 0
    last_count = 0
    total_count = 0
    start_time = 0
    duration = 0
    last_duration = 0
    
    # Initialize detection 'smoothing' variables
    xmin, ymin, xlast, ylast = 0, 0, 0, 0
    distance = 0
    start_seen = 0
    last_seen = 0
    
    # Set detection 'smoothing' parameters
    ND_threshold = 2.0 #Non-detection lag time
    MV_threshold = 200 #Re-detection move distance
    
    start_inference = time.time()            
    while cap.isOpened():
        
        ### TODO: Read from the video capture ###
        flag, frame = cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)
        
        ### TODO: Pre-process the image as needed ###
        p_frame = cv2.resize(frame, (net_input_shape[3], net_input_shape[2]))
        p_frame = p_frame.transpose((2,0,1))
        p_frame = p_frame.reshape(1, *p_frame.shape)

        ### TODO: Start asynchronous inference for specified request ###
        infer_network.exec_net(p_frame)
        if infer_network.wait() == 0:

            ### TODO: Get the results of the inference request ###
            result = infer_network.get_output()
            
            ### TODO: Extract any desired stats from the results ### 
            # Draw bounding boxes onto the input, count people
            current_count = 0
            height = frame.shape[0]
            width = frame.shape[1]
            for box in result[0][0]: # Output shape is 1x1xNx7
                if box[2] >= args.prob_threshold:
                    xmin = int(box[3] * width)
                    ymin = int(box[4] * height)
                    xmax = int(box[5] * width)
                    ymax = int(box[6] * height)
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)
                    current_count += 1
                    start_seen = time.time()
            
            ### TODO: Calculate and send relevant information on ###
            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            ### Topic "person/duration": key of "duration" ###
            
            # Persist last_count if non-detection and just see
            last_seen = time.time() - start_seen
            if current_count == 0 and last_seen < ND_threshold:       
                current_count = last_count
            
            # New person entering frame, so increase total count and start timing
            if current_count > last_count:
                # Adjust total and duration if re-detection of same person
                distance = ((xmin-xlast)**2 + (ymin-ylast)**2)**0.5
                if distance < MV_threshold:
                    interval = time.time() - interval_time
                    start_time = round(time.time() - interval - last_duration, 2)
                    log.warning("Person #{} redetection! interval {}, last duration {}".format(total_count, interval, last_duration))
                else:
                    start_time = time.time()
                    total_count += current_count - last_count
                    log.warning("Person #{} new detection!". format(total_count))
                    client.publish("person", json.dumps({"total": total_count}))
            
            # Person leaving frame
            if current_count < last_count:
                duration = round(time.time() - start_time, 2)
                interval_time = time.time()
                start_seen = 0
                xlast, ylast = xmin, ymin
                client.publish("person/duration", json.dumps({"duration": duration}))
                last_duration = duration
                log.warning("Person #{} duration was: {}".format(total_count, duration))
            
            client.publish("person", json.dumps({"count": current_count}))
            last_count = current_count
            
            # Break if escape key pressed
            if key_pressed == 27:
                break

        ### TODO: Send the frame to the FFMPEG server ###
        sys.stdout.buffer.write(frame)
        sys.stdout.flush()

        ### TODO: Write an output image if `single_image_mode` ###
        if single_image_mode:
            cv2.imwrite('output_image.jpg', frame)
    
    # Release the capture and destroy any OpenCV windows
    client.publish("person", json.dumps({"count": current_count}))
    cap.release()
    cv2.destroyAllWindows()
        
    # Disconnect from MQTT
    client.disconnect()
    
    # Reset network
    log.warning("Time to run inference on the model is {}!".format(round(time.time() - start_inference, 2)))
    infer_network.clean()


def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()
