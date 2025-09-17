import cv2
import os
import json

datadir = os.path.dirname(__file__) + "\\..\\testing_data"
folder_name = datadir.split("\\")[-1]

bad_images_count = 0
total_count = 0

labels = {}
labels[folder_name] = []

for image in os.listdir(datadir):
    img = cv2.imread(os.path.join(datadir, image))

    lower = (0, 154, 0) 
    upper = (0, 255, 0) 

    mask = cv2.inRange(img, lower, upper)
    result = cv2.bitwise_and(img, img, mask=mask)

    # count how many non-zero pixels are in the mask
    count = cv2.countNonZero(mask)

    data = {
        "file_name": image,
        "is_green": count > 500,
        "label": ""
    }
    
    if count > 500:
        bad_images_count += 1
        print(f"{image}: {count} pixels")
        
    
    labels[folder_name].append(data)
    total_count += 1


print(f"{bad_images_count} bad, out of {total_count} images: {bad_images_count/total_count*100}%")
print(total_count)

with open("labels.json", "w", encoding="utf-8") as f:
    json.dump(labels, f, indent=4)

# json, file_name, (green or not green), (label) 

# big folder
#   - blue 1
#      - image1
#   - blue 2
#      - image1
#   - red 1
#      - image1
#   - red 2
#      - image1