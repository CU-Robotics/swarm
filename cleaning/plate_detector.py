import cv2
import numpy as np
import yaml
import os

parent_dir = os.path.dirname(os.path.abspath(__file__)) + "\\"
config = yaml.load(open(parent_dir  + "config.yaml", 'r', encoding='utf-8'), Loader=yaml.FullLoader)


blue_lower = np.array(config['blue_lower'])
blue_upper = np.array(config['blue_upper'])

red_lower = np.array(config['red_lower'])
red_upper = np.array(config['red_upper'])

min_pixel_area = 5
min_h_to_w_ratio = 1.0
min_light_overlap = 0.8
min_light_size_ratio = 0.1
min_width_to_height_ratio = .8 # 1
max_width_to_height_ratio = 6
min_inner_width_respect_to_light_width = 1000 #1.5

#Min heght to search for symbol
# int inner_symbol_search_height = 3
#How big the symbol can be in relation to target
min_symbol_to_target_height_ratio = 0.6
# Max offset x symbol can be from center of target
max_symbol_x_offset_from_center = 0.15
# Max offset y symbol can be from center of target
max_symbol_y_offset_from_center = 3 # 0.3


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


if __name__ == "__main__":
    # img = cv2.imread(parent_dir + "..\\testing_data\\blue1_image5_1.png")
    # cv2.imshow("Original", img)
    # cv2.waitKey(0)

    folder_path = parent_dir + "..\\..\\data\\blue1"
    for image_name in os.listdir(folder_path):
        img = cv2.imread(os.path.join(folder_path, image_name))

        plates, debug_img = detect_armor_plates(img, color="blue", debug=True)
        print("file:", image_name)
        print("Detected armor plates:", plates)

        # resize debug image for better visibility
        debug_img = cv2.resize(debug_img, (0,0), fx=0.5, fy=0.5)
        cv2.imshow("Detections", debug_img)
        cv2.waitKey(0)
    
    cv2.destroyAllWindows()
    
