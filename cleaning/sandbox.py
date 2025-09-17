# C:\Users\geodz\OneDrive\Documents\School\Robotics Club\data\blue1\blue1_image105_21.png

import cv2
import os

image = cv2.imread(os.path.dirname(__file__) + "\\..\\..\\data\\blue1\\blue1_image105_21.png")


for x in range(image.shape[0]):
    for y in range(image.shape[1]):
        pixel = image[x, y]
        # if pixel is green
        
        
        lower = (0, 144, 0) 
        upper = (10, 164, 10) 

        if pixel[0] >= lower[0] and pixel[0] <= upper[0] and pixel[1] >= lower[1] and pixel[1] <= upper[1] and pixel[2] >= lower[2] and pixel[2] <= upper[2]:
            print(pixel[0], pixel[1], pixel[2])