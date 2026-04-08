#!/usr/bin/env python3
import cv2
import numpy as np
import pipeline
import logging
import pathlib

target_x_pixels = 100
target_y_pixels = 100

def main():
    ctx = pipeline.get_stage_context()
    ctx.make_output_dir("cropped")
    
    # iterate over data rows
    for data, src, dst in ctx.rows():
        image = cv2.imread(str(src))

        for i, plate in enumerate(data["labels"].get("plates", [])):
            # get bounding box
            x, y, w, h = plate

            # find out largest side, if the width is greater, we want to make the height match
            # If the height is greater, we want to keep as usual
            l_side = max(w, h)

            # crop image to bounding box
            cropped = image[(y + h//2 - l_side//2):(y + h //2 + l_side//2), x:(x + w)]

            # resize image to target size
            cropped = cv2.resize(cropped, (target_x_pixels, target_y_pixels))

            bw = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
            
            # If there are multiple plates, give them unique names so we don't overwrite the cropped images
            filename = str(dst.stem)

            if len(data["labels"].get("plates", [])) > 1:
                filename = filename + f"-{i}"
            
            filename = dst.parent / (filename + dst.suffix)
            cv2.imwrite(str(filename), bw)
        
        
    # update metadata
    ctx.update()

if __name__ == "__main__":
    main()