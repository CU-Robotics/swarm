#!/usr/bin/env python3
import cv2
import numpy as np
import pipeline
import logging
import pathlib
import os
import json
import sys

folder = '' 
data = ''
target_x_pixels = 100
target_y_pixels = 100

def main():
    
    labels = json.load(open(data))
    pathlib.Path(folder + '/../cropped').mkdir(parents=True, exist_ok=True)
    # iterate over data rows
    for entry in labels["examples"]:
        
        # print(entry["name"])
        image = cv2.imread(folder + '/' + entry["name"])
        
        for i, plate in enumerate(entry["labels"].get("plates", [])):
            # get bounding box
            x, y, w, h = plate

            # crop image to bounding box
            
            cropped = image[(y + h//2 - w//2):(y + h//2 + w//2), x:(x + w)]

            # cropped = image[y:(y + h), x:(x + w)]

            # resize image to target size
            cropped = cv2.resize(cropped, (target_x_pixels, target_y_pixels))

            # save cropped image
            cv2.imwrite(f"{folder}/../cropped/{i}-{entry["name"]}", cropped)
        
        

if __name__ == "__main__":
    # get folder from command line argument

    if len(sys.argv) == 2:
        print("Arguments passed:" + sys.argv[1])
        folder = os.getcwd() + sys.argv[1]
        print("Reading from directory: " + folder + "\n")
        parent_dir = os.path.dirname(folder)
        parent_folder_name = os.path.basename(parent_dir)
        data = parent_dir + '/metadata.json'
        print("Reading from metadata file: " + data + "\n")
    else:
        print("Incorrect number of arguments passed. Please ONLY provide the folder path to be processed.")
        exit()

    main()