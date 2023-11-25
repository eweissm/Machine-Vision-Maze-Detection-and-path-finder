# Machine-Vision-Maze-Detection-and-path-finder
This project involved writing a Python script which would analyze the live feed from a web cam 
to (1) find the start and end of a maze, (2) determine the walls of the maze, (3) determine where the 
corners of the maze were, (4) use the found corners of the maze to apply a parallelogram correction to 
rectangularize the image, and (5) solve the maze. This was achieved by detecting volumes in the image 
of certain colors, allowing for the corners (yellow), walls (pink), the start, and the finish (both green) to 
be detected in the image. Then it was a simple matter of applying the inverse parallelogram 
transformations and using an A-star pathfinding algorithm to find the shortest path through the maze. In 
the images below you will see (top left) the original image with the various elements detected, (top 
right) the rectangularized image, and (figure 2) the shortest path denoted by the blue line.
This project was inspired by the IEEE Micromouse competition which requires a robot to solve a 
maze as quickly as possible. Given enough development time, an algorithm like this could be written in 
which a robot could quickly evaluate the maze, use the rectangularization to convert the maze into a 
usable bitmap, and then quickly complete the maze.


![image](https://github.com/eweissm/Machine-Vision-Maze-Detection-and-path-finder/assets/73143081/f18b13a7-e95c-4e51-92b1-5429f127a3ef)
Fig 1. Screenshot showing the original image (left) and the rectangularized image (right). Note the outlined volumes denoting different elements.

![image](https://github.com/eweissm/Machine-Vision-Maze-Detection-and-path-finder/assets/73143081/7efeacc2-b12d-493c-91a2-4da06ad7673f)


Fig 2. Bitmap interpretation of the "maze". Note that the pink paper became an obstacle for the maze to find a path around.
