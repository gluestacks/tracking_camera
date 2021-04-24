from mtcnn.mtcnn import MTCNN
import cv2
import sys
import numpy as np
import pigpio
import time
import sys
from time import sleep
from math import sin, cos, radians
detector = MTCNN()

video_feed = 50
while video_feed != 1 and video_feed != 0:
    video_feed = int(input("Video Input Choice (0 or 1): "))

cap = cv2.VideoCapture(video_feed)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

clrs_6 = (0, 255, 255)

pi = pigpio.pi()
x_deg_face = 1500
y_deg_face = 1700
pi.set_servo_pulsewidth(13, x_deg_face)
pi.set_servo_pulsewidth(26, y_deg_face)

detection_factor = 0
factor = 0
while float(detection_factor) < 0.2 or float(detection_factor) > 1:
    detection_factor = input("Enter detecting factor: ")

while float(factor) < 0.1 or float(factor) > 1:
    factor = input("Enter tracking factor: ")
    if factor == "back":
        factor = 0
        detection_factor = input("Enter detecting factor: ")
    if factor == "q":
        sys.exit()

detection_factor = float(detection_factor)
factor = float(factor)


x_ = int(220*factor)
y_ = int(170*factor)
w_ = int(640*factor-2*(x_))
h_ = int(480*factor-2*(y_))

#def servo_move(x, y, w, h, position_x, position_y):
##    if (x_ + w_) <= (x + 2*w/3):
##        x_diff_face = ((x/factor) + 2*(w/factor)/3) - 320
##        deg_change_face_x = 0.3*x_diff_face
##        position_x -= deg_change_face_x
##        pi.set_servo_pulsewidth(13, x_deg_face)
##    elif x + w/3 <= x_:
##        x_diff_face = 320 - ((x/factor) + (w/factor)/3) 
##        deg_change_face_x = 0.3*x_diff_face
##        position_x += deg_change_face_x
##        pi.set_servo_pulsewidth(13, x_deg_face)
##    if (y_ + h_) <= (y + 2*h/3):
##        y_diff_face = ((y/factor) + 2*(h/factor)/3) - 240
##        deg_change_face_y = 0.4*y_diff_face
##        position_y -= deg_change_face_y
##        pi.set_servo_pulsewidth(26, y_deg_face)
##    elif y + h/3 <= y_:
##        y_diff_face = 240 - ((y/factor) + (h/factor)/3) 
##        deg_change_face_y = 0.4*y_diff_face
##        position_y += deg_change_face_y
##        pi.set_servo_pulsewidth(26, y_deg_face)


OPENCV_OBJECT_TRACKERS = {
        "csrt": cv2.TrackerCSRT_create, 
        "kcf": cv2.TrackerKCF_create, 
        "boosting": cv2.TrackerBoosting_create, 
        "mil": cv2.TrackerMIL_create,
        "tld": cv2.TrackerTLD_create,
        "medianflow": cv2.TrackerMedianFlow_create, 
        "mosse": cv2.TrackerMOSSE_create 
    }

tracker = OPENCV_OBJECT_TRACKERS["csrt"]()
counter = 0
nuttime = 0
t1 = 0
t2 = 0
default = 0

while True: 
    __, max_frame = cap.read()
    t3 = time.perf_counter()
    bigger_frame = cv2.resize(max_frame, (0,0), fx= detection_factor, fy= detection_factor)
    frame = cv2.resize(max_frame, (0,0), fx= factor, fy= factor)
    cv2.rectangle(frame, (x_, y_), (x_+w_, y_+h_), clrs_6, 2)
    t2 = time.perf_counter()
    looptime = t2 - t1
    if default == 0 or looptime >= 0.5:
        result = detector.detect_faces(bigger_frame)
        
        if result != []:
            for face in result:
                bbox = face['box']
                x = int(bbox[0]*(factor/detection_factor))
                y = int(bbox[1]*(factor/detection_factor))
                w = int(bbox[2]*(factor/detection_factor))
                h = int(bbox[3]*(factor/detection_factor))
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0,155,255), 2)
                #servo_move(x, y, w, h, x_deg_face, y_deg_face)
##                if (x_ + w_) <= (x + 2*w/3):
##                    x_diff_face = ((x/factor) + 2*(w/factor)/3) - 320
##                    deg_change_face_x = 0.3*x_diff_face
##                    x_deg_face -= deg_change_face_x
##                    pi.set_servo_pulsewidth(13, x_deg_face)
##                elif x + w/3 <= x_:
##                    x_diff_face = 320 - ((x/factor) + (w/factor)/3) 
##                    deg_change_face_x = 0.3*x_diff_face
##                    x_deg_face += deg_change_face_x
##                    pi.set_servo_pulsewidth(13, x_deg_face)
##                if (y_ + h_) <= (y + 2*h/3):
##                    y_diff_face = ((y/factor) + 2*(h/factor)/3) - 240
##                    deg_change_face_y = 0.4*y_diff_face
##                    y_deg_face -= deg_change_face_y
##                    pi.set_servo_pulsewidth(26, y_deg_face)
##                elif y + h/3 <= y_:
##                    y_diff_face = 240 - ((y/factor) + (h/factor)/3) 
##                    deg_change_face_y = 0.4*y_diff_face
##                    y_deg_face += deg_change_face_y
##                    pi.set_servo_pulsewidth(26, y_deg_face)
                #if x != None and default == 1:
                #    tracker = None
                tracker = OPENCV_OBJECT_TRACKERS["csrt"]()
                tracker.init(frame, (x,y,w,h))
                t1 = time.perf_counter()
                default = 1
    
    else:
        (success, box) = tracker.update(frame)
        (x, y, w, h) = [int(v) for v in box]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        #servo_move(x, y, w, h, x_deg_face, y_deg_face)
        if (x_ + w_) <= (x + 2*w/3):
            x_diff_face = ((x/factor) + 2*(w/factor)/3) - 320
            deg_change_face_x = 0.2*x_diff_face
            x_deg_face -= deg_change_face_x
            pi.set_servo_pulsewidth(13, x_deg_face)
        elif x + w/3 <= x_:
            x_diff_face = 320 - ((x/factor) + (w/factor)/3) 
            deg_change_face_x = 0.2*x_diff_face
            x_deg_face += deg_change_face_x
            pi.set_servo_pulsewidth(13, x_deg_face)
        if (y_ + h_) <= (y + 2*h/3):
            y_diff_face = ((y/factor) + 2*(h/factor)/3) - 240
            deg_change_face_y = 0.3*y_diff_face
            y_deg_face -= deg_change_face_y
            pi.set_servo_pulsewidth(26, y_deg_face)
        elif y + h/3 <= y_:
            y_diff_face = 240 - ((y/factor) + (h/factor)/3) 
            deg_change_face_y = 0.3*y_diff_face
            y_deg_face += deg_change_face_y
            pi.set_servo_pulsewidth(26, y_deg_face)
    cv2.imshow('frame', frame)
    t4 = time.perf_counter()
    print(f'{counter}: {1/(t4-t3):.2f} Hz')
    
##    if 1/(t4-t3) >= 13:
##        x_deg_face = 1500
##        y_deg_face = 1700
##        pi.set_servo_pulsewidth(13, x_deg_face)
##        pi.set_servo_pulsewidth(26, y_deg_face)
    
    if cv2.waitKey(1) &0xFF == ord('q'):
        break
        
        
        
cap.release()
cv2.destroyAllWindows()
