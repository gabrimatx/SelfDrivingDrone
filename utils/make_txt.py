import os

def write_image_paths_to_file(folder_path, output_file):
    """
    Creates a txt file with the file paths of all the images in a given folder
    """
    files = os.listdir(folder_path)
    image_files = [file for file in files if file.endswith(('.jpg'))]

    with open(output_file, 'w') as file:
        for image_file in image_files:
            image_path = os.path.join(folder_path, image_file)
            file.write(image_path + '\n')

    print(f"File paths written to: {output_file}")

folder_path = "neg/"
output_file = "neg.txt"
write_image_paths_to_file(folder_path, output_file)