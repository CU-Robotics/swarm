import sys 
import os
import json


def main():
    if len(sys.argv) == 2:
            print("Arguments passed:" + sys.argv[1])
    else:
        print("Incorrect number of arguments passed. Please ONLY provide the folder path to be processed.")
        exit()
        
    datadir = os.getcwd() + "/" +  sys.argv[1]

    # for each folder in datadir
    combined_metadata = []

    for folder in os.listdir(datadir):
        folder_path = os.path.join(datadir, folder)
        if not os.path.isdir(folder_path):
            continue 
        

        metadata_path = ""

        # look for json file starting with pipeline-metadata and ending with .json
        for file in os.listdir(os.path.join(datadir, folder)):
            # print("file:", file)
            if file.startswith("pipeline-meta") and file.endswith(".json"):
                metadata_path = os.path.join(folder_path, file)
                break

        if not os.path.exists(metadata_path):
            print(f"Warning: {metadata_path} does not exist, skipping")
            continue
        else:
            print(f"Found metadata file: {metadata_path}")
        
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

            for data in metadata[folder]:

                if data["valid"] == False:
                    continue

                for plate in data["labels"]["plates"]:
                    combined = {}
                    combined["folder"] = folder
                    combined["name"] = data["name"]
                    combined["plate"] = plate
                    combined["labels"] = data["labels"]

                    combined["labels"].pop("plates", None)
                    

                    combined_metadata.append(combined)
    
    # write combined metadata to json file
    with open(os.path.join(datadir, "combined_metadata.json"), "w") as f:
        json.dump(combined_metadata, f, indent=4)
    print(f"Combined metadata written to {os.path.join(datadir, 'combined_metadata.json')}")

                


if __name__ == "__main__":
    main()