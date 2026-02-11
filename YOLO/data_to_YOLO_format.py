import os
import shutil
from urllib.parse import unquote

# --- CONFIGURATION ---
# The folder you unzipped from Label Studio (contains a 'labels' subfolder)
ls_export_path = './YOLO/data' 
# Where you want the final training-ready data to go
output_dir = './YOLO/clean_yolo_dataset'
# The keyword in your path to start the relative structure from
project_root = 'sandbox' 
# ---------------------

def fix_yolo_structure():
    ls_labels_dir = os.path.join(ls_export_path, 'labels')
    
    if not os.path.exists(ls_labels_dir):
        print(f"Error: Could not find 'labels' folder in {ls_export_path}")
        return

    for label_file in os.listdir(ls_labels_dir):
        if not label_file.endswith('.txt'):
            continue

        # 1. Decode URL encoding and split the hash
        # Input: 0af73ceb__Users%5Cgeodz%5C...%5Csandbox%5Cframes_openspace%5Cframe_00031.txt
        decoded_name = unquote(label_file)
        
        # Get the absolute path part
        try:
            abs_path = decoded_name.split('__')[-1]
        except IndexError:
            print(f"Skipping {label_file}: No hash separator found.")
            continue

        # 2. Extract the relative path (everything after 'sandbox')
        # Result: frames_openspace\frame_00031.txt
        if project_root in abs_path:
            rel_path = abs_path.split(project_root)[-1].lstrip('\\/')
        else:
            print(f"Skipping {label_file}: '{project_root}' not found in path.")
            continue

        # 3. Setup destination directories
        sub_folder = os.path.dirname(rel_path)
        dest_label_dir = os.path.join(output_dir, 'labels', sub_folder)
        dest_image_dir = os.path.join(output_dir, 'images', sub_folder)
        
        os.makedirs(dest_label_dir, exist_ok=True)
        os.makedirs(dest_image_dir, exist_ok=True)

        # 4. Define source and destination filenames
        clean_filename = os.path.basename(rel_path)
        image_filename = clean_filename.replace('.txt', '.jpg') # Change to .png if needed
        source_image_path = abs_path.replace('.txt', '.jpg')   # The image's current home

        # 5. Move/Copy files
        # Copy the label from the Label Studio export
        shutil.copy(os.path.join(ls_labels_dir, label_file), 
                    os.path.join(dest_label_dir, clean_filename))
        
        # Copy the original image from your computer
        if os.path.exists(source_image_path):
            shutil.copy(source_image_path, os.path.join(dest_image_dir, image_filename))
        else:
            print(f"Warning: Original image not found at {source_image_path}")

    # Copy notes/classes
    for extra in ['classes.txt', 'notes.json']:
        extra_path = os.path.join(ls_export_path, extra)
        if os.path.exists(extra_path):
            shutil.copy(extra_path, output_dir)

    print(f"Done! Clean YOLO dataset created at: {output_dir}")

if __name__ == "__main__":
    fix_yolo_structure()