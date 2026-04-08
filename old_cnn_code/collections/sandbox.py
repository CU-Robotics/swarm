
# for each image in images folder, check if there is a corresponding label in the labels folder
import os
import numpy as np

def find_unlabeled_images(images_folder, labels_folder):
    unlabeled_images = []
    for image_name in os.listdir(images_folder):
        label_name = os.path.splitext(image_name)[0] + ".png"
        if not os.path.exists(os.path.join(labels_folder, label_name)):
            unlabeled_images.append(image_name)
    return unlabeled_images

if __name__ == "__main__":
    file_folder_dir = os.path.dirname(os.path.abspath(__file__))
    images_folder = os.path.join(file_folder_dir, "images")
    labels_folder = os.path.join(file_folder_dir, "labels")

    unlabeled_images = find_unlabeled_images(labels_folder, images_folder)
    print(f"Unlabeled images: {unlabeled_images}")
