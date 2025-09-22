import cv2
import os
import json
import sys

# accepts folder name "raw", for each valid image, process it & edit metadata.
def main():
    if len(sys.argv) == 2:
        print("Arguments passed:" + sys.argv[1])
    else:
        print("Incorrect number of arguments passed. Please ONLY provide the folder path to be processed.")
        exit()
    
    datadir = os.getcwd() + sys.argv[1]
    parent_dir = os.path.dirname(datadir)
    parent_folder_name = os.path.basename(parent_dir)

    print("Reading from directory: " + datadir + "\n")

    with open(f"{parent_dir}/metadata.json", 'r') as f:
        metadata = json.load(f)
        f.close()
      
    
    

    bad_images_count = 0
    total_count = 0

    print("=====PROCESSING=====")
    for entry in metadata[parent_folder_name]:
        img = cv2.imread(os.path.join(datadir, entry["file_name"]))

        lower = (0, 154, 0) 
        upper = (0, 255, 0) 

        mask = cv2.inRange(img, lower, upper)
        result = cv2.bitwise_and(img, img, mask=mask)

        # count how many non-zero pixels are in the mask
        count = cv2.countNonZero(mask)

        # If there's a significant amount of pure green pixels, we can safley assume it was corrupted
        # Set label to be false
        if count > 500:
            bad_images_count += 1
            entry["is_valid"] = False
            entry["labels"]["is_green"] =  True
            
        
        print(f"{entry["file_name"]}: {count} green pixels")
            
        
        total_count += 1
    print("=====FINISHED=====\n")

    print(f"{bad_images_count} out of {total_count} images had green error pixels: {round(bad_images_count/total_count*100, 2)}% error rate")

    with open(f"{parent_dir}/metadata.json", 'w') as f:
        json.dump(metadata, f, indent=4)
        f.close()
    


    
    

if __name__ == "__main__":
    main()
    # add return to new folder name

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