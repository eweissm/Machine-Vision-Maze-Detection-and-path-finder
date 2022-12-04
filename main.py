
import cv2
from matplotlib import pyplot as plt
import numpy as np
color_dict_HSV = {'black': [[180, 255, 30], [0, 0, 0]],
              'white': [[180, 18, 255], [0, 0, 231]],
              'red1': [[180, 255, 255], [159, 50, 70]],
              'red2': [[9, 255, 255], [0, 50, 70]],
              'green': [[89, 255, 255], [36, 50, 70]],
              'blue': [[128, 255, 255], [90, 50, 70]],
              'yellow': [[35, 255, 255], [25, 50, 70]],
              'purple': [[158, 255, 255], [129, 50, 70]],
              'orange': [[24, 255, 255], [10, 50, 70]],
              'gray': [[180, 18, 230], [0, 0, 40]]}

video_0 = cv2.VideoCapture(0)

lower_bound = np.array([36, 50, 70])
upper_bound = np.array([89, 255, 255])

lower_bound2 = np.array([90, 50, 70])
upper_bound2 = np.array([128, 255, 255])



while(True):
    retu, frame = video_0.read()
    i=0
    j=0
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    mask2 = cv2.inRange(hsv, lower_bound2, upper_bound2)
    # define kernel size
    kernel = np.ones((7, 7), np.uint8)
    # Remove unnecessary noise from mask
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel)
    mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel)

    color1 = cv2.bitwise_and(frame, frame, mask=mask)

    color2 = cv2.bitwise_and(frame, frame, mask=mask2)

    contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours2, hierarchy2 = cv2.findContours(mask2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    C = np.empty([len(contours2), 2], 'i')
    #output = cv2.bitwise_and(frame,segmented_img, mask =mq)
    # Draw contour on original image
    output = cv2.drawContours(frame, contours, -1, (0, 0, 255), 3)
    output = cv2.drawContours(frame, contours2,-1, (0, 255, 0), 3)
    #print(np.shape(contours))

    print(len(contours2))
    if len(contours2)>0:
       for i in range(len(contours2)):
            M = cv2.moments(contours2[i])
            C[i,0]= int(M['m10'] / M['m00']) #cx
            C[i,1] = int(M['m01'] / M['m00'])#cy
            output[C[i,1]-2:C[i,1]+2,C[i,0]-2:C[i,0]+2] = [255, 255, 255]

    cv2.imshow('frame',output)


    if cv2.waitKey(1) & 0xFF==ord('a'):
        break
video_0.release()
cv2.destroyAllWindows()

