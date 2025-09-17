import cv2
import os
import json
import sys

def main():
    if len(sys.argv) == 2:
        print("Arguments passed:" + sys.argv[1])
    else:
        print("Incorrect number of arguments passed. Please provide the folder path to be processed.")
        exit()
    

    datadir = os.path.dirname(os.path.abspath(__file__)) + sys.argv[1]
    print("Reading from directory: " + datadir + "\n")

    folder_name = datadir.split("\\")[-1]

    bad_images_count = 0
    total_count = 0

    labels = {}
    labels[folder_name] = []

    print("=====PROCESSING=====")
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
        
        print(f"{image}: {count} green pixels")
            
        
        labels[folder_name].append(data)
        total_count += 1
    print("=====FINISHED=====\n")

    print(f"{bad_images_count} out of {total_count} images had green error pixels: {round(bad_images_count/total_count*100, 2)}% error rate")
    print(f"Labels saved to {folder_name}.json\n")
    
    with open(f"{folder_name}.json", "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=4)
    
    

if __name__ == "__main__":
    main()


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