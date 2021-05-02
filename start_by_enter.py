import os
import RPi.GPIO as GPIO
ledPin = 7
while True:
    input("Press Enter")
    os.system('python3 /home/pi/tracking_camera/prototype_v1_final.py')
