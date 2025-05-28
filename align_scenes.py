import cv2
import numpy as np
import rawpy
import os
import csv
from utils import apply_homography_transformation

# Load the scene image 
def load_scene(scene):
    scene_path = os.path.join("scenes", scene + ".NEF")
    with rawpy.imread(scene_path) as raw:
        rgb = raw.postprocess()
    scene_img = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    height, width, _ = scene_img.shape
    if width < height:
        scene_img = np.rot90(scene_img)
    return scene_img

# Load the scene shot image
def load_scene_shot(scene_id, shot_id):
    scene_shot_path = os.path.join("scenes_shots", scene_id, shot_id)
    with rawpy.imread(scene_shot_path) as raw:
        rgb = raw.postprocess()
    scene_shot_img = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    height, width = scene_shot_img.shape
    if width < height:
        scene_shot_img = np.rot90(scene_shot_img)
    return scene_shot_img

# Function to read the homography matrix from the CSV file
def read_homography_from_csv(file_path, scene, scene_id):
    with open(file_path, mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            # Check if the row matches the desired scene and scene_id
            if row[0] == scene and row[1] == scene_id:
                # Extract the homography values (skip the first two columns: Scene, Scene ID)
                H_values = list(map(float, row[2:]))
                # Reshape the flattened values into a 3x3 matrix
                H = np.array(H_values).reshape(3, 3)
                return H
    return None  # Return None if no matching row is found


def align_shots_with_scenes():
    folders = ['A&W1', 'A&W2', 'A&W3', 'Harbour1', 'Harbour2', 'Harbour3',
            'Harbour4', 'SFU_art', 'blue_ceiling', 'dining_area',
            'downtown_smith', 'edu_area', 'foodcourt_mcnz', 'hallway',
            'image_theater', 'owl_statue', 'playground', 'rugs', 'seat_rows',
            'study_area', 'stump', 'subway1', 'subway2', 'theater', 'tree_tunel',
            'uncle_fatih1', 'uncle_fatih2', 'under_tree2', 'wall_art',
            'wall_hallway', 'wall_lab']
    
    # Path to the CSV file containing homography matrices
    csv_file_path = 'homography_transformation.csv'

    for scene_id in folders:

        scene_img = load_scene(scene_id)

        # Create a directory to save resulting images
        folder_save = os.path.join('scenes_aligned', scene_id)
        os.makedirs(folder_save, exist_ok=True)

        for shot_id in os.listdir(os.path.join('scenes_shots', scene_id)):
            
            # Read the homography matrix for the given scene and scene shot
            H = read_homography_from_csv(csv_file_path, scene_id, shot_id)

            if H is not None:
                # Apply the homography to transform the scene shot image
                transformed_img = apply_homography_transformation(scene_img, H)

                # Save the transformed image
                save_name = os.path.splitext(shot_id)[0] + '.jpg'
                save_path = os.path.join(folder_save, save_name)

                cv2.imwrite(save_path, transformed_img)

            else:
                print(f"No homography matrix found for {scene_id}/{shot_id}")


if __name__ == '__main__':
    align_shots_with_scenes()