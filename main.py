
import cv2
#from matplotlib import pyplot as plt
import numpy as np

from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder

#this function performs a pathfinding algorithm (A*) given a maze, a starting location and an end location
#the maze is a matrix where zeros are the wall of the maze and numbers are locations where travel is allowed. The higher the number the higher the cost
def FindPath(maze, startPoint, endPoint):
    #print(np.shape(endPoint))
    grid = Grid(matrix=maze, inverse=True)
    start = grid.node(startPoint[0], startPoint[1])
    end = grid.node(endPoint[0], endPoint[1])
    finder = AStarFinder(diagonal_movement=DiagonalMovement.always)
    path, runs = finder.find_path(start, end, grid)
    #print('operations:', runs, 'path length:', len(path))
    #print(grid.grid_str(path=path, start=start, end=end))
    return path

#connect camera
video_0 = cv2.VideoCapture(0)

#specify color HSV bounds
lower_bound = np.array([150, 150,120])
upper_bound = np.array([170, 255, 255])

lower_bound2 = np.array([36, 50, 70])
upper_bound2 = np.array([89, 255, 255])

meshSize =20

cameraXDim = 640
cameraYDim = 480

while(True):

#get video frames
    retu, frame = video_0.read()

#create empty meshed to detect where maze walls are
    mazeWalls = np.zeros([cameraYDim, cameraXDim])
    meshedMazeWalls= np.zeros([int(np.size(mazeWalls,0)/meshSize), int(np.size(mazeWalls,1)/meshSize)])

#get hsv colors
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

#find color masks
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

#finds contours from colors
    contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours2, hierarchy2 = cv2.findContours(mask2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#array of center points of contours
    C = np.empty([len(contours), 2], 'i')
    C2 = np.empty([len(contours2), 2], 'i')
    #fixed = np.empty([len(contours), 2], 'i')

    # Draw contour on original image
    output = cv2.drawContours(frame, contours, -1, (0, 0, 255), 5)
    output = cv2.drawContours(frame, contours2,-1, (0, 255, 0), 5)

#finds centerpoint of colored dots... adds to array and adds dot to image
    if len(contours)>0:
       for i in range(len(contours)):
            M = cv2.moments(contours[i])
            C[i,0]= int(M['m10'] / M['m00']) #cx
            C[i,1] = int(M['m01'] / M['m00'])#cy
            output[C[i,1]-2:C[i,1]+2,C[i,0]-2:C[i,0]+2] = [255, 255, 255]
            output = cv2.putText(output, str(i),(C[i,0],C[i,1]),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 0),2,cv2.LINE_AA)

#put contours into mazeWall array
            for j in range(len(contours[i])):
                mazeWalls[contours[i][j][0,1]][contours[i][j][0,0]]=1

#converts maze wall into a pixelated version with a given mesh size
#This will reduce the computer power needed to do pathfinding and it will make it so glitchy contours dont get missed
       for a in range(np.size(meshedMazeWalls,0)):
            for b in range(np.size(meshedMazeWalls, 1)):

                if np.any(mazeWalls[a*meshSize:a*meshSize+meshSize, b*meshSize:b*meshSize+meshSize]):
                    meshedMazeWalls[a][b]=1
    print(np.shape(meshedMazeWalls))
    start = np.array([0, 0], 'i')
    end = np.array([23,24], 'i')
    path = FindPath(meshedMazeWalls, start, end)

    #print(path[0][0])

    for i in range(len(path)):
        meshedMazeWalls[path[i][1]][path[i][0]] = .5

    meshedMazeWalls[start[0],start[1]]=.7
    meshedMazeWalls[end[0],end[1]]=.7
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

#TODO: make it so that the index numbers are correct regardless of which dot is higher
       #for j in [1, 2, 3]:
           #if C[0,0] > C[j,0] & C[0,1] < C[j,1]

# transform the image to rectangularize it
    if len(contours) == 4:
        destpts = np.float32([[0, cameraYDim], [cameraXDim,cameraYDim], [0, 0], [cameraXDim, 0]])
        resmatrix = cv2.getPerspectiveTransform(np.float32(C), destpts)
        frame = cv2.warpPerspective(frame, resmatrix, (cameraXDim, cameraYDim))

#output images
    cv2.imshow('frame',frame)
    cv2.imshow('frame2', output)
    #cv2.imshow('frame3', mazeWalls)
    cv2.imshow('frame4', cv2.resize(meshedMazeWalls,[cameraXDim,cameraYDim]))
    if cv2.waitKey(1) & 0xFF==ord('a'):
        break
video_0.release()
cv2.destroyAllWindows()


