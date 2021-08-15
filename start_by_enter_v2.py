import os
import RPi.GPIO as GPIO
import sys
GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)
ledPin_green = 11
ledPin_white = 29
ledPin_yellow = 40
GPIO.setup(ledPin_green, GPIO.OUT)
GPIO.setup(ledPin_white, GPIO.OUT)
GPIO.setup(ledPin_yellow, GPIO.OUT)
while True:
    GPIO.output(ledPin_yellow, GPIO.HIGH)
    input("Press Enter")
    GPIO.output(ledPin_yellow, GPIO.LOW)
    os.system('python3 /home/pi/tracking_camera/prototype_v2.py')
    GPIO.output(ledPin_yellow, GPIO.HIGH)
    leave = input("Press 5")
    if leave == "5":
        GPIO.output(ledPin_white, GPIO.LOW)
        GPIO.output(ledPin_green, GPIO.LOW)
        GPIO.output(ledPin_yellow, GPIO.LOW)
        sys.exit()
