##---Import Libraries---##
import cv2
from mtcnn.mtcnn import MTCNN
import pigpio
import RPi.GPIO as GPIO
import sys
import numpy as np
import time

##---Define Basic Variables---##
#LED
GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)
ledPin_green = 7
ledPin_red = 11
ledPin_white = 40
GPIO.setup(ledPin_green, GPIO.OUT)
GPIO.setup(ledPin_red, GPIO.OUT)
GPIO.setup(ledPin_white, GPIO.OUT)
GPIO.output(ledPin_white, GPIO.HIGH)

#Video
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
clr_main = (0, 255, 255)

#Detection
detection_factor = 0.7
detector = MTCNN()

#Tracking
factor = 0.5
tracker = cv2.TrackerCSRT_create()

#Servo Movement
servo_step_x = 18
servo_step_y = 18
distance_stepup = 2
pi = pigpio.pi()
x_deg_face = 1500
y_deg_face = 1700
pi.set_servo_pulsewidth(13, x_deg_face)
pi.set_servo_pulsewidth(26, y_deg_face)
x_ = int(220*factor)
y_ = int(170*factor)
w_ = int(640*factor-2*(x_))
h_ = int(480*factor-2*(y_))
x_old = 0
y_old = 0
w_old = 0
h_old = 0
undetected = 0
x_diff_face = 0
deg_change_face_x = 0
y_diff_face = 0
deg_change_face_y = 0

#Time 
counter = 0
looptime = 0
t1 = 0
t2 = 0
default = 0

##---Define Functions---##
def servo_movement(box_x, box_y, box_width, box_height):
    global x_deg_face
    global y_deg_face
    if box_y < y_ and box_y + box_height > y_+h_:
        pass
    else:
        if (x_ + w_) <= (box_x + 2*box_width/3):
            if box_width*box_height > 3500:
                x_deg_face -= (servo_step_x + 2*distance_stepup)
            elif 1000 >= box_width*box_height <= 3500:
                x_deg_face -= (servo_step_x + distance_stepup)
            else:
                x_deg_face -= servo_step_x
            pi.set_servo_pulsewidth(13, x_deg_face)
        elif box_x + box_width/3 <= x_:
            if box_width*box_height > 3500:
                x_deg_face += (servo_step_x + 2*distance_stepup)
            elif 1000 >= box_width*box_height <= 3500:
                x_deg_face += (servo_step_x + distance_stepup)
            else:
                x_deg_face += servo_step_x
            pi.set_servo_pulsewidth(13, x_deg_face)
    if box_x < x_ and box_x + box_width > x_+w_:
        pass
    else:
        if (y_ + h_) <= (box_y + 2*box_height/3):
            if box_width*box_height > 3500:
                y_deg_face -= (servo_step_y + 2*distance_stepup)
            elif 1000 >= box_width*box_height <= 3500:
                y_deg_face -= (servo_step_y + distance_stepup)
            else:
                y_deg_face -= servo_step_y
            pi.set_servo_pulsewidth(26, y_deg_face)
        elif box_y + box_height/3 <= y_:
            if box_width*box_height > 3500:
                y_deg_face += (servo_step_y + 2*distance_stepup)
            elif 1000 >= box_width*box_height <= 3500:
                y_deg_face += (servo_step_y + distance_stepup)
            else:
                y_deg_face += servo_step_y
            pi.set_servo_pulsewidth(26, y_deg_face)

def undetected_goal(last_x, last_y, last_w, last_h, dilation):
    global x_diff_face
    global deg_change_face_x
    global y_diff_face
    global deg_change_face_y
    x_diff_face = abs(((last_x/dilation) + 2*(last_w/dilation/3) - 320))
    if x_diff_face < 70:
        deg_change_face_x = servo_step_x*round((0.25*x_diff_face)/servo_step_x)
    else:
        deg_change_face_x = servo_step_x*round((0.25*70)/servo_step_x)
    #deg_change_face_x = servo_step_x*round((0.25*x_diff_face)/servo_step_x)
    y_diff_face = abs(((last_y/dilation) + 2*(last_h/dilation)/3) - 240)
    deg_change_face_y = servo_step_y*round((0.35*y_diff_face)/servo_step_y)

def old_coordinate_save(recent_x, recent_y, recent_w, recent_h):
    global x_old
    global y_old
    global w_old
    global h_old
    x_old = recent_x
    y_old = recent_y
    w_old = recent_w
    h_old = recent_h

##---Detection, Tracking, and Movement---##
while True:
    GPIO.output(ledPin_white, GPIO.LOW)
    __, max_frame = cap.read()
    t3 = time.perf_counter()
    bigger_frame = cv2.resize(max_frame, (0,0), fx= detection_factor, fy= detection_factor)
    frame = cv2.resize(max_frame, (0,0), fx= factor, fy= factor)
    cv2.rectangle(frame, (x_, y_), (x_+w_, y_+h_), clr_main, 2)
    t2 = time.perf_counter()
    looptime = t2 - t1
    if default == 0 or looptime >= 0.5:
        result = detector.detect_faces(bigger_frame)
        
        if result != []:
            GPIO.output(ledPin_red, GPIO.LOW)
            GPIO.output(ledPin_green, GPIO.HIGH)
            undetected = 0
            for face in result[:1]:
                bbox = face['box']
                x = int(bbox[0]*(factor/detection_factor))
                y = int(bbox[1]*(factor/detection_factor))
                w = int(bbox[2]*(factor/detection_factor))
                h = int(bbox[3]*(factor/detection_factor))
                old_coordinate_save(x, y, w, h)
                servo_movement(x, y, w, h)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0,155,255), 2)
                tracker = cv2.TrackerCSRT_create()
                tracker.init(frame, (x,y,w,h))
                t1 = time.perf_counter()
                default = 1

        else:
            if undetected == 0:
                undetected = 1
                undetected_goal(x_old, y_old, w_old, h_old, factor)
            GPIO.output(ledPin_red, GPIO.HIGH)
            GPIO.output(ledPin_green, GPIO.LOW)
            if (x_ + w_) <= (x_old + 2*w_old/3) and deg_change_face_x >= 0:
                deg_change_face_x -= servo_step_x
                x_deg_face -= servo_step_x
                pi.set_servo_pulsewidth(13, x_deg_face)
            elif x_old + w_old/3 <= x_ and deg_change_face_x >= 0:
                deg_change_face_x -= servo_step_x
                x_deg_face += servo_step_x
                pi.set_servo_pulsewidth(13, x_deg_face)
            if (y_ + h_) <= (y_old + 2*h_old/3) and deg_change_face_y >= 0:
                deg_change_face_y -= servo_step_y
                y_deg_face -= servo_step_y
                pi.set_servo_pulsewidth(26, y_deg_face)
            elif y_old + h_old/3 <= y_ and deg_change_face_y >= 0:
                deg_change_face_y -= servo_step_y
                y_deg_face += servo_step_y
                pi.set_servo_pulsewidth(26, y_deg_face)
                
    else:
        (success, box) = tracker.update(frame)
        GPIO.output(ledPin_red, GPIO.LOW)
        GPIO.output(ledPin_green, GPIO.HIGH)
        if success:
            (x, y, w, h) = [int(v) for v in box]
            old_coordinate_save(x, y, w, h)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        servo_movement(x, y, w, h)

    cv2.imshow('frame', frame)
    t4 = time.perf_counter()
    print(f'{counter}: {1/(t4-t3):.2f} Hz')
    
    if cv2.waitKey(1) &0xFF == ord('0'):
        GPIO.output(ledPin_green, GPIO.LOW)
        GPIO.output(ledPin_red, GPIO.LOW)
        GPIO.output(ledPin_white, GPIO.LOW)
        break
        
        
        
cap.release()
cv2.destroyAllWindows()

