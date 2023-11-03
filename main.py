
import cv2
#from matplotlib import pyplot as plt
import numpy as np

from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder

#this function performs a pathfinding algorithm (A*) given a maze, a starting location and an end location
#the maze is a matrix where zeros are the wall of the maze and numbers are locations where travel is allowed. The higher the number the higher the cost
def FindPath(maze, startPoint, endPoint):

    grid = Grid(np.size(maze,1),np.size(maze,0) ,matrix=maze, inverse=True)
    start = grid.node(startPoint[0], startPoint[1])
    end = grid.node(endPoint[0], endPoint[1])
    finder = AStarFinder(diagonal_movement=DiagonalMovement.always)
    path, runs = finder.find_path(start, end, grid)

    return path

#connect camera
video_0 = cv2.VideoCapture(0)

#specify color HSV bounds
lower_bound = np.array([130, 50,50])
upper_bound = np.array([170, 255, 255])

lower_bound2 = np.array([36, 50, 70]) # define green (color of maze start and end)
upper_bound2 = np.array([89, 255, 255])

lower_bound3 = np.array([25, 50, 70]) # define yellow (color of maze corners)
upper_bound3 = np.array([35, 255, 255])

meshSize =10

cameraXDim = 640
cameraYDim = 480

while(True):

#get video frames
    retu, frame = video_0.read()

#create empty meshed to detect where maze walls are
    meshedMazeWalls= np.zeros([int(cameraYDim/meshSize), int(cameraXDim/meshSize)])
    mazeFrame = np.ones([int(cameraYDim/meshSize), int(cameraXDim/meshSize),3])

#get hsv colors
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

#find color masks
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    mask2 = cv2.inRange(hsv, lower_bound2, upper_bound2)
    mask3 = cv2.inRange(hsv, lower_bound3, upper_bound3)

    # define kernel size
    kernel = np.ones((7, 7), np.uint8)

    # Remove unnecessary noise from mask
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel)
    mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel)
    mask3 = cv2.morphologyEx(mask3, cv2.MORPH_CLOSE, kernel)
    mask3 = cv2.morphologyEx(mask3, cv2.MORPH_OPEN, kernel)

    color1 = cv2.bitwise_and(frame, frame, mask=mask)
    color2 = cv2.bitwise_and(frame, frame, mask=mask2)
    color3 = cv2.bitwise_and(frame, frame, mask=mask3)

#finds contours from colors
    contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours2, hierarchy2 = cv2.findContours(mask2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours3, hierarchy3 = cv2.findContours(mask3.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#array of center points of contours
    C = np.empty([len(contours), 2], 'i')
    C2 = np.empty([len(contours2), 2], 'i')
    C3 = np.empty([len(contours3), 2], 'i')

    #fixed = np.empty([len(contours), 2], 'i')

    # Draw contour on original image
    output = cv2.drawContours(frame, contours, -1, (0, 0, 255), 5)
    output = cv2.drawContours(frame, contours2,-1, (0, 255, 0), 5)
    output = cv2.drawContours(frame, contours3,-1, (255, 0, 0), 5)

#finds centerpoint of colored dots... adds to array and adds dot to image
    if len(contours2) > 0:
        for i in range(len(contours2)):
            M = cv2.moments(contours2[i])
            C2[i, 0] = int(M['m10'] / M['m00'])  # cx
            C2[i, 1] = int(M['m01'] / M['m00'])  # cy
            output[C2[i, 1] - 2:C2[i, 1] + 2, C2[i, 0] - 2:C2[i, 0] + 2] = [255, 255, 255]
            output = cv2.putText(output, str(i), (C2[i, 0], C2[i, 1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (245, 244, 66),2, cv2.LINE_AA)

    if len(contours2) > 1:
        cv2.line(output, (C2[0,0],C2[0,1]),(C2[1,0],C2[1,1]), (255, 0, 0), 2 )

    if len(contours3) > 0:
        for i in range(len(contours3)):
            M = cv2.moments(contours3[i])
            C3[i, 0] = int(M['m10'] / M['m00'])  # cx
            C3[i, 1] = int(M['m01'] / M['m00'])  # cy
            output[C3[i, 1] - 2:C3[i, 1] + 2, C3[i, 0] - 2:C3[i, 0] + 2] = [255, 255, 255]
            output = cv2.putText(output, str(i), (C3[i, 0], C3[i, 1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (245, 244, 66),2, cv2.LINE_AA)

# transform the image to rectangularize it---------------------------------------------------------------------------
    if len(contours3) == 4:
        destpts = np.float32([[0, cameraYDim], [cameraXDim,cameraYDim], [0, 0], [cameraXDim, 0]])
        resmatrix = cv2.getPerspectiveTransform(np.float32(C3), destpts)
        frame = cv2.warpPerspective(frame, resmatrix, (cameraXDim, cameraYDim))
        #color1=cv2.warpPerspective(color1, resmatrix, (cameraXDim, cameraYDim))

#finds centerpoint of colored dots... adds to array and adds dot to image
    if len(contours)>0:
#converts maze wall into a pixelated version with a given mesh size
#This will reduce the computer power needed to do pathfinding and it will make it so glitchy contours dont get missed
       for a in range(np.size(meshedMazeWalls,0)):
            for b in range(np.size(meshedMazeWalls, 1)):

                if np.any(color1[a*meshSize:a*meshSize+meshSize, b*meshSize:b*meshSize+meshSize]):
                    meshedMazeWalls[a][b]=1
                    mazeFrame[a][b] = [0, 0, 0]
## Path finding stuff------------------------------------------------------------------------
    if len(contours2)>=2:
        start = np.array([int(C2[0,0]/meshSize), int(C2[0,1]/meshSize)], 'i')
        end = np.array([int(C2[1,0]/meshSize), int(C2[1,1]/meshSize)], 'i')
    else:
        start = np.array([1, 1], 'i')
        end = np.array([62,46], 'i')

    path = FindPath(meshedMazeWalls, start, end)

    for i in range(len(path)):
        mazeFrame[path[i][1]][path[i][0]] = [255, 0, 0]

    mazeFrame[start[1],start[0]] = [0, 0, 255]
    mazeFrame[end[1],end[0]] = [0, 255, 0]

#TODO: make it so that the index numbers are correct regardless of which dot is higher
       #for j in [1, 2, 3]:
           #if C[0,0] > C[j,0] & C[0,1] < C[j,1]

#output images----------------------------------------------------------------------------------------
    cv2.imshow('frame',frame)
    cv2.imshow('frame2', output)
    cv2.imshow('frame3', cv2.resize(mazeFrame,[cameraXDim,cameraYDim]))

    if cv2.waitKey(1) & 0xFF==ord('a'):
        break
video_0.release()
cv2.destroyAllWindows()


