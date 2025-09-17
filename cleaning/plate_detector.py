import cv2
import numpy as np
import yaml
import os

parent_dir = os.path.dirname(os.path.abspath(__file__)) + "\\"
config = yaml.load(open(parent_dir  + "config.yaml", 'r', encoding='utf-8'), Loader=yaml.FullLoader)

# HSV thresholds (you can tune these based on your YAML values)
white_lower = np.array(config['white_lower'])
white_upper = np.array(config['white_upper'])

blue_lower = np.array(config['blue_lower'])
blue_upper = np.array(config['blue_upper'])

red_lower = np.array(config['red_lower'])
red_upper = np.array(config['red_upper'])

red_lower1 = np.array([red_lower[0] + 180, red_lower[1], red_lower[2]])
red_upper1 = np.array([180, red_upper[1], red_upper[2]])
red_lower2 = np.array([0, red_lower[1], red_lower[2]])
red_upper2 = np.array(red_upper)



def detect_armor_plates(image, color="blue", debug=True):
    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # White mask
    mask_white = cv2.inRange(hsv, white_lower, white_upper)

    # Team color mask
    if color == "blue":
        mask_color = cv2.inRange(hsv, blue_lower, blue_upper)
    elif color == "red":
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
        if area < 5:  # min area
            continue
        if h / max(w, 1) < 1.0:  # height/width ratio
            continue
        filtered_lights.append(rect)
        if debug:
            cv2.rectangle(image, (x, y), (x+w, y+h), (200, 104, 52), 2)

    # Pair lights into possible armor plates
    armor_plates = []
    for i in range(len(filtered_lights)):
        for j in range(i+1, len(filtered_lights)):
            x1, y1, w1, h1 = filtered_lights[i]
            x2, y2, w2, h2 = filtered_lights[j]

            # Check height ratio
            ratio = h1 / float(h2)
            if ratio < 0.1 or ratio > 10:  # rough filter
                continue

            # Build target rect
            tx = min(x1, x2)
            ty = min(y1, y2)
            tw = max(x1+w1, x2+w2) - tx
            th = max(y1+h1, y2+h2) - ty

            whr = tw / float(th)
            if whr < 0.8 or whr > 6:
                continue

            armor_plates.append((tx, ty, tw, th))
            if debug:
                cv2.rectangle(image, (tx, ty), (tx+tw, ty+th), (0, 255, 0), 2)

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
    
