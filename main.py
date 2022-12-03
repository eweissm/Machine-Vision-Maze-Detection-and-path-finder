
import cv2
from matplotlib import pyplot as plt
import numpy as np

video_0 = cv2.VideoCapture(0)

lower_bound = np.array([90,50, 70])
upper_bound = np.array([128, 255, 255])

while(True):
    retu, frame = video_0.read()
    i=0
    j=0
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    # define kernel size
    kernel = np.ones((7, 7), np.uint8)
    # Remove unnecessary noise from mask
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


    segmented_img = cv2.bitwise_and(frame, frame, mask=mask)
    contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Draw contour on original image
    output = cv2.drawContours(frame, contours, -1, (0, 0, 255), 3)
    cv2.imshow('frame',frame)


    if cv2.waitKey(1) & 0xFF==ord('a'):
        break
video_0.release()
cv2.destroyAllWindows()

