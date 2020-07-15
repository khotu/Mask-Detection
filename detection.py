import cv2
from utils import detector_utils as detector_utils
import pandas as pd
import argparse
import numpy as np
import datetime
from datetime import date

ap = argparse.ArgumentParser()
ap.add_argument('-d', '--display', dest='display', type=int,
                        default=1, help='Display the detected images using OpenCV. This reduces FPS')
args = vars(ap.parse_args())
detection_graph, sess = detector_utils.load_inference_graph()

if __name__ == '__main__':
    # Detection confidence threshold to draw bounding box
    score_thresh = 0.80
    num_hands_detect = 2

    # define a video capture object
    vid = cv2.VideoCapture(0)


    # Used to calculate fps
    start_time = datetime.datetime.now()
    num_frames = 0

    while True:

        # Capture the video frame
        # by frame
        ret, frame = vid.read()

        im_height, im_width = (None, None)
        frame = np.array(frame)
        if im_height == None:
            im_height, im_width = frame.shape[:2]

        # Convert image to rgb since opencv loads images in bgr, if not accuracy will decrease
        try:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except:
            print("Error converting to RGB")

        boxes, scores, classes = detector_utils.detect_objects(frame, detection_graph, sess)

        a, b = detector_utils.draw_box_on_image(num_hands_detect, score_thresh,
                                                scores, boxes, classes, im_width,
                                                im_height, frame)

        # Calculate Frames per second (FPS)
        num_frames += 1
        elapsed_time = (datetime.datetime.now() -
                        start_time).total_seconds()
        fps = num_frames / elapsed_time

        if args['display']:

            # Display FPS on frame
            detector_utils.draw_text_on_image("FPS : " + str("{0:.2f}".format(fps)), frame)
            cv2.imshow('Detection', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                vid.stop()
                break
        # # Display the resulting frame
        # cv2.imshow('frame', frame)
        #
        # # the 'q' button is set as the
        # # quitting button you may use any
        # # desired button of your choice
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()
