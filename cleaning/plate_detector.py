#!/usr/bin/env python3
import cv2
import numpy as np
import yaml
import pathlib
import os
import logging
import pipeline

# Path to data to detect plates on
parent_dir = pathlib.Path(__file__).resolve().parent
config_path = parent_dir / "config.yaml"

# Config info for color mask bounds
config = yaml.load(config_path.read_text(), Loader=yaml.FullLoader)

blue_lower = np.array(config['blue_lower'])
blue_upper = np.array(config['blue_upper'])

red_lower = np.array(config['red_lower'])
red_upper = np.array(config['red_upper'])

# Magic Numbers copied over from the C++ version, idk how these were found
min_pixel_area = 5
min_h_to_w_ratio = 1.0  
min_light_overlap = 0.8
min_light_size_ratio = 0.1
min_width_to_height_ratio = .8 # 1
max_width_to_height_ratio = 6
min_inner_width_respect_to_light_width = 1000 #1.5

# Arguments
#   - image: the image to detect on
#   - color: what team color are you looking for from plates (default blue team)
#   - debug: unsure rn
# Outputs
#   - armor_plates: rectangles of said plates
#   - image:        image with rectangles drawn
def detect_armor_plates(image, color="blue", debug=True):
    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Team color mask
    if color == "blue":
        mask_color = cv2.inRange(hsv, blue_lower, blue_upper)
    elif color == "red":
        red_lower1 = np.array([red_lower[0] + 180, red_lower[1], red_lower[2]])
        red_upper1 = np.array([180, red_upper[1], red_upper[2]])
        red_lower2 = np.array([0, red_lower[1], red_lower[2]])
        red_upper2 = np.array(red_upper)

        mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
        mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
        mask_color = cv2.bitwise_or(mask1, mask2)
    else:
        raise ValueError(f"Invalid color: {color}")

    # Morphological cleanup
    mask_color = cv2.dilate(mask_color, None, iterations=3)
    mask_color = cv2.erode(mask_color, None, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(mask_color, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    filtered_lights = []
    for cnt in contours:
        rect = cv2.boundingRect(cnt)
        x, y, w, h = rect
        area = w * h
        if area < min_pixel_area:  # min area
            continue
        if h / max(w, 1) < min_h_to_w_ratio:  # height/width ratio
            continue
        filtered_lights.append(rect)
        if debug:
            cv2.rectangle(image, (x, y), (x+w, y+h), (200, 104, 52), 2)

    overlap_multiplier = (min_light_overlap / 2) + 5
    # Pair lights into possible armor plates
    armor_plates = []
    for i in range(len(filtered_lights)):
        x1, y1, w1, h1 = filtered_lights[i]
        light1_overlap = y1 + h1 * overlap_multiplier

        for j in range(i+1, len(filtered_lights)):
            x2, y2, w2, h2 = filtered_lights[j]
            light2_overlap = y2 + h2 * overlap_multiplier

            if (light1_overlap < y2 or light2_overlap < y1): 
                continue

            # Check height ratio
            vertical_size_ratio = h1 / (h2)
            if vertical_size_ratio < min_light_size_ratio or vertical_size_ratio > (1 / min_light_size_ratio):  # rough filter
                continue

            # Build target rect
            target_x = min(x1, x2)
            target_y = min(y1, y2)
            target_width = max(x1+w1, x2+w2) - target_x
            target_height = max(y1+h1, y2+h2) - target_y

            target_width_height_ratio = target_width / (target_height)

            if target_width_height_ratio < min_width_to_height_ratio or target_width_height_ratio > max_width_to_height_ratio:
                continue

            armor_plates.append((target_x, target_y, target_width, target_height))
            if debug:
                cv2.rectangle(image, (target_x, target_y), (target_x+target_width, target_y+target_height), (0, 255, 0), 2)

    return armor_plates, image

def main():
    ctx = pipeline.get_stage_context()

    debug = False
    for data, src, _ in ctx.rows():
        image = cv2.imread(src)
        
        plates, debug_img = detect_armor_plates(image, color="blue", debug = debug)
        
        if debug:
            # resize debug image for better visibility
            debug_img = cv2.resize(debug_img, (0,0), fx=0.5, fy=0.5)
            cv2.imshow("Detections", debug_img)
            cv2.waitKey(0)
        
        data["labels"]["plates"] = plates
        if len(plates) == 0:
            data["valid"] = False
    
    if debug:
        cv2.destroyAllWindows()

    # update metadata
    ctx.update()

if __name__ == "__main__":
    main()