import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)
ledPin = 7
GPIO.setup(ledPin, GPIO.OUT)

for i in range(5):
    print("LED is on")
    GPIO.output(ledPin, GPIO.HIGH)
    time.sleep(0.5)
    GPIO.output(ledPin, GPIO.LOW)
    time.sleep(0.5)
