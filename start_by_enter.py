import os
import RPi.GPIO as GPIO
import sys
ledPin = 7
GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)
ledPin_green = 11
ledPin_red = 7
ledPin_blue = 40
GPIO.setup(ledPin_green, GPIO.OUT)
GPIO.setup(ledPin_red, GPIO.OUT)
GPIO.setup(ledPin_blue, GPIO.OUT)
while True:
    GPIO.output(ledPin_red, GPIO.HIGH)
    GPIO.output(ledPin_green, GPIO.HIGH)
    input("Press Enter")
    GPIO.output(ledPin_red, GPIO.LOW)
    GPIO.output(ledPin_green, GPIO.LOW)
    os.system('python3 /home/pi/tracking_camera/prototype_v1_final.py')
    GPIO.output(ledPin_red, GPIO.HIGH)
    GPIO.output(ledPin_green, GPIO.HIGH)
    leave = input("Press 00")
    if leave == "00":
        sys.exit()
