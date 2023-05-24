import cv2
import os

def apply_filters_to_images(folder_path, new_folder_path):
    """
    Applies some filters to all the images in a given folder
    """
    files = os.listdir(folder_path)
    image_files = [file for file in files if file.endswith(('.jpg'))]

    for i, file_name in enumerate(image_files):
        image_path = os.path.join(folder_path, file_name)
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2GRAY)
        image = cv2.GaussianBlur(image, (17, 17), 0)
        image = cv2.Canny(image, threshold1=30, threshold2=100)

        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # image = cv2.equalizeHist(image)
        # image = cv2.GaussianBlur(image, (5, 5), 1)

        cv2.imwrite(new_folder_path+file_name, image)
        print(f"Processed: {file_name}")

def rename_images(folder_path):
    """
    Renames all the images in a given folder
    """
    files = os.listdir(folder_path)
    image_files = [file for file in files if file.endswith(('.jpg'))]

    for i, file_name in enumerate(image_files):
        image_path = os.path.join(folder_path, file_name)

        new_file_name = f"img__{i}.jpg"
        new_path = os.path.join(folder_path, new_file_name)

        os.rename(image_path, new_path)
        print(f"Renamed: {file_name} -> {new_file_name}")

var = 'neg'
folder_path = var + "_unprocessed/"
new_folder_path = var + '/'
rename_images(folder_path)