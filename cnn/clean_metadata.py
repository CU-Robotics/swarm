import os
import json
import sys

# iterates through the metadata json and ensure all entries exist in the cropped folder
# if not, delete the entry from the json
def main():
    if len(sys.argv) == 2:
        print("Arguments passed:" + sys.argv[1])
    else:
        print("Incorrect number of arguments passed. Please ONLY provide the folder path to be processed.")
        exit()
        
        
    datadir = os.getcwd() + sys.argv[1]
    # print("Datadir: " + datadir + "\n")
    parent_dir = os.path.dirname(datadir)

    metadata_path = os.path.join(datadir, "combined_metadata.json")
    if not os.path.exists(metadata_path):
        print(f"Error: {metadata_path} does not exist")
        exit()
    else:
        print(f"Found metadata file: {metadata_path}")
    
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
        
    cleaned_metadata = []
    for entry in metadata:
        folder = entry["folder"]
        name = entry["name"]
        cropped_path = os.path.join(datadir, folder, "cropped", name)
        if os.path.exists(cropped_path):
            cleaned_metadata.append(entry)
        else:
            print(f"Warning: {cropped_path} does not exist, removing entry from metadata")
        
    print(f"Cleaned metadata has {len(cleaned_metadata)} entries (out of {len(metadata)})")
    cleaned_metadata_path = os.path.join(datadir, "cleaned_metadata.json")
    with open(cleaned_metadata_path, "w") as f:
        json.dump(cleaned_metadata, f, indent=4)

if __name__ == "__main__":
    main()