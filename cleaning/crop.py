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

        for plate in data["labels"].get("plates", []):
            # get bounding box
            x, y, w, h = plate

            # crop image to bounding box
            cropped = image[(y + h//2 - w//2):(y + h //2 + w //2), x:(x + w)]

            # resize image to target size
            cropped = cv2.resize(cropped, (target_x_pixels, target_y_pixels))

            # save cropped image
            cv2.imwrite(str(dst), cropped)
        
        
    # update metadata
    ctx.update()

if __name__ == "__main__":
    main()