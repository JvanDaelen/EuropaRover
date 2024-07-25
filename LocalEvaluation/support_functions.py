"""A set of functions to support operations for reinforcement learning.
These functions are not related to the machine learning process itself."""
import os


def save_file_path(folder: str, file_name : str, extension : str):
    # Ensure the folder exists in the EuropaRover directory
    current_file_path = os.path.abspath(__file__)
    europarover_folder_path = os.path.dirname(current_file_path)
    target_directory = os.path.join(europarover_folder_path, folder)
    if os.path.isdir(target_directory) is False:
        os.makedirs(target_directory)
    
    file_path = os.path.join(target_directory, f"{file_name}.{extension}")
    # Check if such file already exists to warn the user
    version = 1
    while os.path.isfile(file_path) is True:
        if version == 1:
            file_path = file_path.split('.')[0] + f'-{version}.{extension}'
        else:
            file_path = file_path.replace(f'-{version-1}.{extension}', f'-{version}.{extension}') 
        version += 1
    return file_path