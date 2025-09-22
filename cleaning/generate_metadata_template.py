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

    metadata_template = {
        parent_folder_name: []
    }

    for image_path in os.listdir(datadir):
        metadata_template[parent_folder_name].append({"file_name": image_path, "is_valid": True, "label": {}})

    print(json.dumps(metadata_template))

    with open(f"{parent_dir}/metadata.json", "w") as f:
        f.write(json.dumps(metadata_template, indent=4))
    
    

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