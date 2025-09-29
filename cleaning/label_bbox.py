#!/usr/bin/env python3
import cv2
import sys
import numpy as np
import pipeline
import logging
from util import TextBox, silence_stderr, KeyCode
from tqdm import tqdm

BASE_BORDER = 1

# globals for drawing
drawing = False
curr_image = None
ox, oy = -1, -1

def mouse(event, x, y, flags, param):
    global curr_image, drawing, ox, oy
    if curr_image is None:
        return
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ox, oy = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            image = curr_image.copy()
            cv2.rectangle(image, (ox, oy), (x, y), (0, 255, 0), 2)
            with silence_stderr():
                cv2.imshow("review image", image)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        rect = (ox, oy, x, y)
        logging.error(rect)
        cv2.rectangle(curr_image, (ox, oy), (x, y), (0, 255, 0), 2)
        with silence_stderr():
            cv2.imshow("review image", curr_image)

def main():
    global curr_image

    ctx = pipeline.get_stage_context()
    rows = list(ctx.rows(filter_by={False})) # query all invalid rows

    if len(rows) == 0:
        logging.info("no invalid images, skipping stage")
        sys.exit(0)

    edits = []

    cv2.namedWindow("review image")
    cv2.setMouseCallback("review image", mouse)

    with tqdm(total=len(rows), unit="img", desc=f"Labeling missing plates") as bar:
        image_index = 0
        done = False

        while not done:
            data, src, _ = rows[image_index]
            undist = cv2.imread(src)

            first_view = True
            while True:
                image = undist.copy()
                rsz = cv2.resize(image, None, fx=0.5, fy=0.5)

                with silence_stderr():
                    cv2.imshow("review image", rsz)
                
                curr_image = rsz

                key = cv2.waitKey(0) & 0xff
                if key == ord(" "):
                    # queue next image
                    image_index = (image_index + 1) % len(rows)
                    bar.n = max(bar.n, image_index + 1)
                    bar.refresh()
                    break
                elif key == KeyCode.BACKSPACE:
                    # queue next image
                    image_index = (image_index - 1) % len(rows)
                    bar.n = max(bar.n, image_index + 1)
                    bar.refresh()
                    break
                elif key == KeyCode.ESCAPE:
                    bar.n = len(rows)
                    bar.refresh()
                    done = True
                    break
                
                first_view = False
    
    cv2.destroyAllWindows()
    ctx.update()

if __name__ == "__main__":
    main()