import cv2
import argparse
import logging as log
import numpy as np

from input_feeder import InputFeeder
from face_detection import ModelFaceDetection
from facial_landmarks_detection import ModelFacialLandmarksDetection
from head_pose_estimation import ModelHeadPoseEstimation
from gaze_estimation import ModelGazeEstimation
from mouse_controller import MouseController

global model_face, model_gaze, model_landmarks, model_pose, mouse_controller


def build_arg():
    """
    Parse command line arguments.
    :return: Command line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--media", required=True, type=str, default="video",
                        help="Type of media: image, video or cam is acceptable (default = video).")
    parser.add_argument("-f", '--file', required=True, type=str, default="./bin/demo.mp4",
                        help="Path to image or video file (default = ./bin/demo.mp4).")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Device to run inference: CPU, GPU, VPU or FPGA are acceptable (default = CPU).")
    parser.add_argument("-t", "--threshold", type=float, default=0.5,
                        help="Probability threshold for face detections (default = 0.5).")
    parser.add_argument("-o", "--output", type=str, default=True,
                        help="Display output: gaze bounding box and vector projection (default = True).")
    parser.add_argument("-p", "--precision", type=str, default="FP16",
                        help="Precision of Gaze, Head Post and Landmarks estimation models (optional).")
    parser.add_argument("-mp", "--mouse_precision", type=str, default="medium",
                        help="Precision of mouse (optional).")
    parser.add_argument("-ms", "--mouse_speed", type=str, default="fast",
                        help="Speed of mouse (optional).")
    return parser


def infer_on_stream(args):
    """
    Initialize the inference network, stream video to network, and output stats and video.
    :param args: Command line arguments parsed by 'build_arg()'
    :return: None
    """
    global mouse_controller

    # Handle the input stream
    if args.media == "cam" or args.media == "video":
        if args.media == "cam":
            feed = InputFeeder(input_type="cam")
            output_file = None
        elif args.media == "video":
            feed = InputFeeder(input_type="video", input_file=args.file)
            output_file = "./bin/output_{}.mp4".format(args.precision)

        feed.load_data()

        initial_w = int(feed.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        initial_h = int(feed.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fps = int(feed.cap.get(cv2.CAP_PROP_FPS))
        out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'avc1'), fps, (initial_w, initial_h), True)

        frame_count = 0

        for batch in feed.next_batch():
            frame_count += 1

            if batch is None:
                break

            result, frame = process_frame(batch, args.output)
            out.write(frame)

            log.debug('Frame #{counter} result: {result}')
            if type(result) == str and result == 'No face detected':
                log.warning('Frame {}: No face detected', frame_count)

            # Mouse control not used because PyAutoGUI fail-safe triggered using demo.mp4
            # mouse_controller.move(result[0], result[1])

        out.release()
        feed.close()

    elif args.media == 'image':
        feed = InputFeeder(input_type='image', input_file=args.file)
        feed.load_data()

        for batch in feed.next_batch():
            result, image = process_frame(batch, args.output)
            cv2.imwrite('./bin/output_{}.png'.format(args.precision), image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        feed.close()


def process_frame(image, output):
    """
    Process each frame of input video, cam, or image.
    :param image: Input image frame'
    :param output: Flag indicating whether to display intermediate output.
    :return: result: Intermediate output results.
    :return: image: Intermediate image output.
    """
    global model_face, model_gaze, model_landmarks, model_pose

    face = model_face.predict(image)

    if face[1] is None:
        return 'No face detected', image

    landmarks = model_landmarks.predict(face[1])
    pose = model_pose.predict(face[1])
    gaze = model_gaze.predict(((landmarks[1][0], landmarks[1][1]), pose))

    # Find gaze projection endpoint
    r_vec, t_vec = np.array([0, 0, 0], np.float), np.array([0, 0, 0], np.float)
    cam_mtx = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float)
    img_pts, joc = cv2.projectPoints(gaze, r_vec, t_vec, cam_mtx, None)

    if output:
        # Draw face bounding box
        image = cv2.rectangle(image, face[0][0], face[0][1], (0, 0, 255), 2)

        # Draw gaze vectors
        res = (int(img_pts[0][0][0] * 50), int(img_pts[0][0][1] * 50))
        for i in range(2):
            x = face[0][0][0] + landmarks[0][i][0]
            y = face[0][0][1] + landmarks[0][i][1]
            start_point = (x, y)
            end_point = (x - res[0], y + res[1])
            cv2.arrowedLine(image, start_point, end_point, (255, 0, 0), 2)

    return img_pts[0][0], image


def main():
    """
    Load the network and parse the output
    :return: None
    """
    # Grab command line arguments
    args = build_arg().parse_args()

    # Initialize logger
    log.basicConfig(filename="./bin/output_{}.log".format(args.precision),
                    format="[ %(levelname)s ] %(message)s", level=log.INFO)
    logger = log.getLogger()

    # Initialize models
    global model_face, model_gaze, model_landmarks, model_pose, mouse_controller

    model_face_file = "./models/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001"
    model_gaze_file = "./models/intel/gaze-estimation-adas-0002/" + args.precision + \
                      "/gaze-estimation-adas-0002"
    model_landmarks_file = "./models/intel/landmarks-regression-retail-0009/" + args.precision + \
                           "/landmarks-regression-retail-0009"
    model_pose_file = "./models/intel/head-pose-estimation-adas-0001/" + args.precision + \
                      "/head-pose-estimation-adas-0001"

    model_face = ModelFaceDetection(model_face_file, args.device, args.threshold)
    model_gaze = ModelGazeEstimation(model_gaze_file, args.device, args.threshold)
    model_landmarks = ModelFacialLandmarksDetection(model_landmarks_file, args.device, args.threshold)
    model_pose = ModelHeadPoseEstimation(model_pose_file, args.device, args.threshold)

    model_face.load_model()
    model_gaze.load_model()
    model_landmarks.load_model()
    model_pose.load_model()

    if args.mouse_precision in ["high", "low", "medium"] and args.mouse_speed in ["fast", "slow", "medium"]:
        mouse_controller = MouseController(args.mouse_precision, args.mouse_speed)
    else:
        log.warning("Mouse precision: {} and speed: {} not specified!", args.mouse_precision, args.mouse_speed)
        mouse_controller = None

    # Perform inference on the input stream
    infer_on_stream(args)

    # Print stats
    model_face.stats()
    model_landmarks.stats()
    model_pose.stats()
    model_gaze.stats()


if __name__ == "__main__":
    main()

