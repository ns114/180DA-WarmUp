'''
Functions and code from opencv.org

Creating Bounding boxes and circles for contours
https://docs.opencv.org/4.x/da/d0c/tutorial_bounding_rects_circles.html

Getting Started with Videos
https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_video_display/py_video_display.html

'''
import numpy as np
import cv2
import random as rng
import argparse

# Create argument parser to accept input from user
argparser = argparse.ArgumentParser()
argparser.add_argument('input')     
argparser.add_argument('output')
args = argparser.parse_args()

input_path = args.input
output_path = args.output

# Calculates contour verticies and plots bounding rectangle onto frame
# Accepts threshold value and frame as paramaters
def thresh_callback(val, frame):
    threshold = val
 
    canny_output = cv2.Canny(frame, threshold, threshold * 2)

    contours, _ = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_poly = [None]*len(contours)
    boundRect = [None]*len(contours)

    for i, c in enumerate(contours):
        contours_poly[i] = cv2.approxPolyDP(c, 3, True)
        boundRect[i] = cv2.boundingRect(contours_poly[i])
    
 
    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
    drawing = frame
 
    for i in range(len(contours)):
        color = (255, 0, 0)
        cv2.rectangle(drawing, (int(boundRect[i][0]), int(boundRect[i][1])), \
        (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), color, 2)
    
    return drawing

# Obtain video capture and setup video writer
cap = cv2.VideoCapture(input_path)

source_window = 'Source'
cv2.namedWindow(source_window)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_video = cv2.VideoWriter(output_path,fourcc, 20.0, (640,480))

# Parse video capture
while(cap.isOpened()):
    ret, frame = cap.read()
    
    # Leave loop if video has ended
    if frame is None:
        break
    
    # Convert to HSV and introduce blur
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    frame_blurred = cv2.blur(frame_hsv, (3,3))

    max_thresh = 255
    thresh = 100
    cv2.createTrackbar('Canny thresh:', source_window, thresh, max_thresh, thresh_callback)
    box = thresh_callback(thresh, frame_blurred)

    cv2.imshow('Bounding Box', box)
    output_video.write(box)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
output_video.release()
cv2.destroyAllWindows()