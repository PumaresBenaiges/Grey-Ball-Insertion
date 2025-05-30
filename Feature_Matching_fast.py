"""
Run this code to obtain 'homography_transformation.csv' which contains the homographic matrices between scene and scene shot.
An example usage can be found in transformation.py
"""

import cv2
import numpy as np
import csv
import os
import rawpy
from concurrent.futures import ThreadPoolExecutor


def load_scene(scene):
    """
    load the scene image (NEF, not TIFF!) from the scene path
    input: name of the scene that shall be loaded
    output: scene image (greyscaled, rotated to have same orientation)
    """
    scene_path = os.path.join("scenes", scene + ".NEF")
    with rawpy.imread(scene_path) as raw:
        rgb = raw.postprocess()
    scene_img = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    height, width = scene_img.shape
    if width < height:
        scene_img = np.rot90(scene_img)
    return scene_img


def load_scene_shot(scene, scene_id):
    """
    load the scene shot image from the scene shot path
    input: name of the scene and ID of the corresponding shot that shall be loaded
    output: scene shot image (greyscaled, rotated to have same orientation)
    """
    scene_shot_path = os.path.join("scenes_shots", scene, scene_id)
    with rawpy.imread(scene_shot_path) as raw:
        rgb = raw.postprocess()
    scene_shot_img = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    height, width= scene_shot_img.shape
    if width < height:
        scene_shot_img = np.rot90(scene_shot_img)
    return scene_shot_img


def feature_matching(scene_img, scene_shot_img):

    """
    apply feature matching between scene and scene shot image
    input: image of scene and scene shot
    output: homographic matrix, transformation between scene and scene shot
    """

    # Initialize ORB feature detector, number of keypoints limited for 500 for faster matching
    orb = cv2.ORB_create(nfeatures=500) 
  
    # Detect keypoints and compute descriptors for both images
    kp1, des1 = orb.detectAndCompute(scene_img, None)
    kp2, des2 = orb.detectAndCompute(scene_shot_img, None)
 
    # Check if keypoints were found in both images
    if len(kp1) == 0 or len(kp2) == 0:
        raise ValueError("No keypoints found in one of the images.")

    # Initialize BF matcher with Hamming distance, enable cross-checking for better match quality
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors for both images
    matches = bf.match(des1, des2)

    # Sort matches based on descriptor distance (lower distance --> better match)
    matches = sorted(matches, key=lambda x: x.distance)

    # Get coordinates of matched keypoints for each image
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])


    # Algorithm needs at least 4 points to compute a homography
    if len(pts1) < 4:
        raise ValueError("Not enough points to compute homography.")

    # compute homography matrix, use RANSAC for handling outliers
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC)

    return H  


def apply_transformation(scene_img, H):
    """
    apply homographic transformation on scene image
    input: scene that needs to be transformed, homographic matrix
    output: transformed image
    """

    # Get size of scene image
    rows, cols = scene_img.shape

    # Reconstruct Homographic (translation) matrix, neglect rotational part as only translation between images
    translation_matrix = np.array([[1, 0, H[0, 2]], [0, 1, H[1, 2]]], dtype=np.float32)  

    # Apply transformation on input image
    transformed_img = cv2.warpAffine(scene_img, translation_matrix, (cols, rows))
    return transformed_img


def process_scene(folder, scene_img, scene_id):

    """
    apply homographic transformation on a specific scene shot within scene
    input: folder name that contains the scene, scene image to match against, ID of the scene shot
    output: tuple: (folder, scene_id, flattened_H), where flattened_H is the 3x3 homography matrix flattened to a list
    """

    # Load the scene shot image using the given folder and scene_id
    scenes_shot_img = load_scene_shot(folder, scene_id)

    # Perform feature matching between scene and scene shot to get homography matrix
    H = feature_matching(scene_img, scenes_shot_img)

    # Flatten 3x3 matrix to list
    flattened_H = H.flatten().tolist()

    return folder, scene_id, flattened_H


def main():

    # List of scenes to process
    folders = ['A&W1', 'A&W2', 'A&W3', 'Harbour1', 'Harbour2', 'Harbour3', 'Harbour4',
               'SFU_art', 'blue_ceiling', 'dining_area', 'downtown_smith', 'edu_area',
               'foodcourt_mcnz', 'hallway', 'image_theater', 'owl_statue', 'playground',
               'rugs', 'seat_rows', 'study_area', 'stump', 'subway1', 'subway2', 'theater',
               'tree_tunel', 'uncle_faith1', 'uncle_faith2', 'under_tree2', 'wall_art',
               'wall_hallway', 'wall_lab']


    transformation_data = []

    # Create ThreadPoolExecutor for parallel processing to speed up homography computations
    with ThreadPoolExecutor(max_workers=64) as executor:
        futures = []

        # Loop through each folder and each scene shot in that folder and submit a processing task for each scene shot
        for folder in folders:
            scenes_img = load_scene(folder)
            for scene_id in os.listdir("scenes_shots/" + folder):
                print(f"Processing: {folder}/{scene_id}")

                futures.append(executor.submit(process_scene, folder, scenes_img, scene_id))

        # Collect results from futures
        for future in futures:
            folder, scene_id, flattened_H = future.result()
            transformation_data.append([folder, scene_id] + flattened_H)
            print(f"Added data for {folder}/{scene_id}")  # Debugging print statement

    # Save the transformation matrix as CSV (including full homography matrix)
    with open('homography_transformation.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        header = ['Scene', 'Scene ID'] + [f'H_{i}{j}' for i in range(3) for j in range(3)]  # Column names for H matrix
        writer.writerow(header)
        writer.writerows(transformation_data)

    print("Homography matrix saved as 'homography_transformation.csv'.")


# Execute whole program
if __name__ == '__main__':
    main()
