import os
import time
import cv2
import numpy as np
from openvino.inference_engine import IENetwork
from openvino.inference_engine import IEPlugin
import argparse

class Queue:
    '''
    Class for dealing with queues
    '''
    def __init__(self):
        self.queues=[]

    def add_queue(self, points):
        self.queues.append(points)

    def get_queues(self, image):
        for q in self.queues:
            x_min, y_min, x_max, y_max=q
            frame=image[y_min:y_max, x_min:x_max]
            yield frame
    
    def check_coords(self, coords):
        d={k+1:0 for k in range(len(self.queues))}
        for coord in coords:
            for i, q in enumerate(self.queues):
                if coord[0]>q[0] and coord[2]<q[2]:
                    d[i+1]+=1
        return d

class PersonDetect:
    '''
    Class for the Person Detection Model.
    '''
    
    def __init__(self):
        '''
        TODO: This method needs to be completed by you (COMPLETED!!!)
        '''
        self.plugin = None
        self.network = None
        self.input_blob = None
        self.output_blob = None
        self.exec_network = None
        # self.infer_request = None

    def load_model(self):
        '''
        TODO: This method needs to be completed by you (COMPLETED!!!)
        '''
    
        model_weights = model + '.bin'
        model_structure = model + '.xml'
    
        # Read the IR as a IENetwork
        self.network = IENetwork(model_structure, model_weights)
    
        # Load the plugin
        self.plugin = IEPlugin(device=device)
        
        # Check plugin for CPU extension and unsupported layers
        check_plugin(self.plugin)
        
        # Add a CPU extension, if applicable
        if extensions and 'CPU' in device:
            self.plugin.add_extension(extensions, device=device)
    
        # Load the IENetwork into the plugin
        self.exec_network = self.plugin.load(self.network, num_requests=1)
    
        # Get the input layer
        self.input_blob = next(iter(self.network.inputs))
        self.output_blob = next(iter(self.network.outputs))
    
        return

    def check_plugin(self, plugin):
        '''
        TODO: This method needs to be completed by you (COMPLETED!!!)
        '''
        # Add a CPU extension, if applicable
        if extensions and 'CPU' in device:
            plugin.add_extension(extensions, 'CPU')

        # Get the supported layers of the network
        if 'CPU' in device:
            supported_layers = plugin.query_network(self.network, 'CPU')

            # Check for any unsupported layers, and let the user
            # know if anything is missing. Exit the program, if so.
            unsupported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]
            if len(unsupported_layers) != 0:
                print("Unsupported layers found: {}".format(unsupported_layers))
                print("Check whether extensions are available to add to IECore.")
                exit(1)
                
        return
        
    def predict(self, image):
        '''
        TODO: This method needs to be completed by you (COMPLETED!!!)
        '''
    
        # Get the name of the input node
        input_name=next(iter(self.network.inputs))
    
        # Preprocess the input image
        preprocessed_image = self.preprocess_input(image)

        '''
        # Running Inference in a loop on the same image
        input_dict = {input_name:preprocessed_image}
    
        start=time.time()
        for _ in range(10):
            self.exec_network.infer(input_dict)
    
        print(f"Time Taken to run 10 Inference on CPU is: {time.time()-start} seconds")
        '''
    
        # Perform inference on the frame
        self.exec_network.start_async(request_id=0, inputs={self.input_blob: image})
    
        # Get the output of inference
        if self.exec_network.wait() == 0:
            outputs = self.exec_network.extract_output()
            # Update the frame to include detected bounding boxes
            image = preprocess_outputs(outputs)
    
        return outputs, image
    
    def preprocess_outputs(self, outputs):
        '''
        TODO: This method needs to be completed by you (COMPLETED!!!)
        '''
        '''
        Draw bounding boxes onto the frame.
        '''
        for box in outputs[0][0]: # Output shape is 1x1xNx7
            conf = box[2]
            if conf >= threshold:
                xmin = int(box[3] * width)
                ymin = int(box[4] * height)
                xmax = int(box[5] * width)
                ymax = int(box[6] * height)
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)
    
        return image

    def preprocess_input(self, image):
        '''
        TODO: This method needs to be completed by you (COMPLETED!!!)
        Given an input image:
        - Find height and width
        - Resize to height and width
        - Transpose the final "channel" dimension to be first
        - Reshape the image to add a "batch" of 1 at the start 
        '''
        n, c, h, w = self.network.inputs[self.input_blob].shape
        preprocess_image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
        preprocess_image = np.moveaxis(image, -1, 0)
        #preprocess_image = preprocess_image.transpose((2,0,1))
        #preprocess_image = preprocess_image.reshape(1, 3, h, w)
        return preprocess_image

def main(args):
    model=args.model
    device=args.device
    extensions=args.extensions
    visualise=args.visualise
    
    video_file=args.video
    queue_file=args.queue_param
    # max_people=args.max_people
    threshold=args.threshold
    
    start=time.time()
    pd=PersonDetect()    
    pd.load_model()
    print(f"Time taken to load the model is: {time.time()-start} seconds")

    try:
        queue=Queue()
        for item in np.load(queue_file):
            queue.add_queue(item)

        # Get and open video capture
        # video_file=args.video
        cap=cv2.VideoCapture(video_file)
        cap.open(video_file)
        
        # Grab the shape of the input 
        width = int(cap.get(3))
        height = int(cap.get(4))
        
        # Create a video writer for the output video
        # The second argument should be
        #`cv2.VideoWriter_fourcc('M','J','P','G')`
        # on Mac, and `0x00000021` on Linux
        out = cv2.VideoWriter('output_video.mp4', 0x00000021, 30, (width,height))

        # Process frames until the video ends, or process is exited
        while cap.isOpened():
            flag, frame=cap.read()
            if not flag:
                break
            key_pressed = cv2.waitKey(60)

            print(f"Total People in frame = {len(coords)}")
            print(f"Number of people in queue = {num_people}")
        
            if visualise:
                #coords, image=pd.predict(frame)
                image = frame
                coords = None
                # Write out the frame
                output_video.write(image)
                num_people=queue.check_coords(coords)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                coords=pd.predict(frame)
                print(coords)

        output_video.release()
        cap.release()
        cv2.destroyAllWindows()
    except Exception as e:
        print("Could not run Inference", e)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--device', default='CPU')
    parser.add_argument('--extensions', default=None)
    
    parser.add_argument('--visualise', action='store_true')
    parser.add_argument('--video', default=None)
    parser.add_argument('--queue_param', default=None)
    parser.add_argument('--max_people', default=1)
    parser.add_argument('--threshold', default=0.5)
    
    args=parser.parse_args()

    main(args)

