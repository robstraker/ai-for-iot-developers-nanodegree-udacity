import numpy as np
from openvino.inference_engine import IECore
from openvino.inference_engine import IENetwork
from openvino.inference_engine import IEPlugin
import os
import cv2
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
        self.infer_request = None

    def load_model(self):
        '''
        TODO: This method needs to be completed by you (COMPLETED!!!)
        '''
        if not check_plugin(device):
            print("OpenVino does not support {device} as a plugin.")
            exit(1)
    
        model_weights = model + '.bin'
        model_structure = model + '.xml'
    
        # Initialize the plugin
        self.plugin = IECore()
    
        # Add a CPU extension, if applicable
        if extensions and 'CPU' in device:
            self.plugin.add_extension(extensions, device)
    
        # Read the IR as a IENetwork
        self.network = IENetwork(model=model_structure, weights=model_weights)
    
        # Load the IENetwork into the plugin
        self.exec_network = self.plugin.load_network(self.network, device)
    
        # Get the input layer
        self.input_blob = next(iter(self.network.inputs))
        self.output_blob = next(iter(self.network.outputs))
    
        return

    def check_plugin(self, plugin):
        '''
        TODO: This method needs to be completed by you (COMPLETED!!!)
        '''
        if plugin in ('CPU', 'GPU', 'FPGA', 'MYRIAD'):
            return TRUE
        else:
            return FALSE
        
    def predict(self, image):
        '''
        TODO: This method needs to be completed by you (COMPLETED!!!)
        '''
    
        '''
        #Perform inference on the frame
        self.exec_network.async_inference(preprocessed_image)
        '''
    
        # Get the name of the input node
        input_name=next(iter(self.network.inputs))
    
        # Preprocess the input image
        preprocessed_image = self.preprocess_input(image)

        # Running Inference in a loop on the same image
        input_dict = {input_name:preprocessed_image}
    
        start=time.time()
        for _ in range(10):
            self.exec_network.infer(input_dict)
    
        print(f"Time Taken to run 10 Inference on CPU is: {time.time()-start} seconds")
    
        # Get the output of inference
        if self.exec_network.wait() == 0:
            result = self.exec_network.extract_output()
            # Update the frame to include detected bounding boxes
            image = preprocess_outputs(image)
    
        return image
    
    def preprocess_outputs(self, outputs):
        '''
        TODO: This method needs to be completed by you (INCOMPLETE)
        '''
        '''
        Draw bounding boxes onto the frame.
        '''
        for box in result[0][0]: # Output shape is 1x1xNx7
            conf = box[2]
            if conf >= 0.5:
                xmin = int(box[3] * width)
                ymin = int(box[4] * height)
                xmax = int(box[5] * width)
                ymax = int(box[6] * height)
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)
    
        '''
        # Show image and bounding boxes if requested
        if visualise:
            # Get only pose detections above 0.5 confidence, set to 255
            for c in range(len(outputs)):
                outputs[c] = np.where(outputs[c]>0.5, 255, 0)
            # Sum along the "class" axis
            outputs = np.sum(outputs, axis=0)
            # Get semantic mask
            # Create an empty array for other color channels of mask
            empty = np.zeros(outputs.shape)
            # Stack to make a Green mask where text detected
            mask = np.dstack((empty, outputs, empty))
            # Combine with original image
            image = image + mask
            return num_people, image
        '''
    
        return TBD

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
        preprocess_image = cv2.resize(image, (w, h))
        preprocess_image = preprocess_image.transpose((2,0,1))
        preprocess_image = preprocess_image.reshape(1, *preprocess_image.shape)
        return preprocess_image

def main(args):
    extensions=args.extensions
    model=args.model
    device=args.device
    visualise=args.visualise

    start=time.time()
    pd=PersonDetect()    
    pd.load_model()
    print(f"Time taken to load the model is: {time.time()-start} seconds")

    try:
        queue=Queue()
        type = os.path.splitext(args.video)[0]
        if type == 'retail':
            queue.add_queue([620, 1, 915, 562])
            queue.add_queue([1000, 1, 1264, 461])
        elif type == 'manufacturing':
            queue.add_queue([15, 180, 730, 780])
            queue.add_queue([921, 144, 1424, 704])
        elif type == 'transport': 
            queue.add_queue([50, 90, 838, 794])
            queue.add_queue([852, 74, 1430, 841])

        # Get and open video capture
        cap=cv2.VideoCapture(args.video)
        cap.open(args.video)
        
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
            HERE HERE HERE HERE

            print(f"Total People in frame = {len(coords)}")
            print(f"Number of people in queue = {num_people}")
        
            if visualise:
                coords, image=pd.predict(frame)
                # Write out the frame
                output_video.write(image)
                num_people=queue.check_coords(coords)
                cv2.imshow("frame", image)
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

