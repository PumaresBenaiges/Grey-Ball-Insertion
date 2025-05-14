import os
import rawpy
import numpy as np
import pandas as pd
import torch
import cv2
from matplotlib.image import interpolations_names
from torch.utils.data import Dataset, DataLoader
import torchvision.utils as vutils
from PIL import Image
from torchvision.utils import save_image


import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

IM_SIZE=(2844,4284,3)
# NEW_SIZE=(2848,4288)
NEW_SIZE=(256,256)
import cv2
import numpy as np


def read_homography_from_csv(reader, scene, scene_id):
    for row in reader:
        # Check if the row matches the desired scene and scene_id
        if row[0] == scene and row[1] == scene_id:
            # Extract the homography values (skip the first two columns: Scene, Scene ID)
            H_values = list(map(float, row[2:]))
            # Reshape the flattened values into a 3x3 matrix
            H = np.array(H_values).reshape(3, 3)
            return H
    return None  # Return None if no matching row is found

def crop_center(image, ball_data, crop_size=(256, 256)):
    """
    Crops the image with the provided crop_size centered at the ball's center.
    Applies a circular mask around the ball, everything else is set to black.
    """
    crop_h, crop_w = crop_size
    cx = int(ball_data['circle_x'].iloc[0])
    cy = int(ball_data['circle_y'].iloc[0])
    radius = int(ball_data['circle_radiuos'].iloc[0]) + 50
    
    # Compute top-left corner
    start_x = cx - crop_w // 2
    start_y = cy - crop_h // 2

    # Get full image dimensions
    img_h, img_w = image.shape[:2]

    # Compute end coordinates
    end_x = start_x + crop_w
    end_y = start_y + crop_h

    # Initialize a black image (the same size as crop)
    cropped = np.zeros((crop_h, crop_w, 3), dtype=np.uint8)

    # Compute valid crop region in the source image
    src_x1 = max(start_x, 0)
    src_y1 = max(start_y, 0)
    src_x2 = min(end_x, img_w)
    src_y2 = min(end_y, img_h)

    # Compute corresponding destination coordinates
    dst_x1 = src_x1 - start_x
    dst_y1 = src_y1 - start_y
    dst_x2 = dst_x1 + (src_x2 - src_x1)
    dst_y2 = dst_y1 + (src_y2 - src_y1)

    # Copy the valid region from the source image
    cropped[dst_y1:dst_y2, dst_x1:dst_x2] = image[src_y1:src_y2, src_x1:src_x2]

    # Create circular mask centered relative to the crop
    mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
    circle_center = (cx - start_x, cy - start_y)
    cv2.circle(mask, circle_center, radius, 255, -1)

    # Apply mask to cropped image
    cropped = cv2.bitwise_and(cropped, cropped, mask=mask)

    return cropped


def crop_center2(image, ball_data, crop_size=(256, 256)):
    """
    Crops the image with the provided crop_size centered at 'center'.
    If the crop goes out of bounds, the region is padded (or resized afterward).
    """
    crop_h, crop_w = crop_size
    cx = ball_data['circle_x']
    cy = ball_data['circle_y']
    # Compute top-left corner
    start_x = int((cx - crop_w // 2).iloc[0])
    start_y = int((cy - crop_h // 2).iloc[0])
    # Get full image dimensions
    img_h, img_w = image.shape[:2]
    # Clip crop coordinates to image boundaries
    start_x = max(start_x, 0)
    start_y = max(start_y, 0)
    end_x = min(start_x + crop_w, img_w)
    end_y = min(start_y + crop_h, img_h)
    cropped = image[start_y:end_y, start_x:end_x]
    # In case the crop is smaller than desired, use resizing (or you might pad instead)
    if (cropped.shape[0] != crop_h) or (cropped.shape[1] != crop_w):
        cropped = cv2.resize(cropped, (crop_w, crop_h), interpolation=cv2.INTER_AREA)
    return cropped


def load_image_jpg(path):
    """
    Args:
        path: path of one JPG image to load.

    Reads the JPG image from the path, preprocesses it and resizes if needed.

    Returns:
        Preprocessed and possibly rotated image as a NumPy array.
    """
    image = Image.open(path).convert('RGB')  # Ensure it's in RGB mode
    image_np = np.array(image)
    
    height, width, _ = image_np.shape
    if width < height:
        image_np = np.rot90(image_np).copy()

    return image_np

def load_image(path):
    """
    Args:
        path: path of one image to load.

    Reads the image from the path, preprocesses it and resizes.

    Returns:
        Preprocessed and resized image.
    """
    # Load and process input image (scene)
    with rawpy.imread(path) as raw:
        image = raw.postprocess()
        height, width, channels = image.shape
        if width < height:
            image = np.rot90(image).copy()
    return image

def load_input_scenes(input_paths):
    """
    Args:
        List of paths of the input scenes images.
    Returns:
        Dictionary with scene_id and loaded image.
    """
    # Load and process input image
    # Changed. How does it saves path?
    input_images = {}
    for scene_id, input_path in input_paths.items():
        input_image = load_image(input_path)
        input_images[scene_id] = input_image
    return input_images

def load_ball_transformation_data(path):
    """
    Returns:
        Dataframe with the ball transformation data for one shot.
    """
    df = pd.read_csv(path)
    return df

def load_ball_position_data(path):
    """
    Returns:
        Dataframe with the ball position data for one shot.
    """
    df = pd.read_csv(path)
    df['image_name'] = df['image_name'].apply(lambda x: os.path.splitext(x)[0])
    df = df[['image_name', 'circle_x', 'circle_y', 'circle_radiuos']]
    return df

def create_probability_map(df, binary=True):
    """
    Given the ball position df creates the mask for the ball.
    """
    x = df['circle_x'].values[0]
    y = df['circle_y'].values[0]
    radius = df['circle_radiuos'].values[0]
    xx, yy = np.meshgrid(np.arange(IM_SIZE[1]), np.arange(IM_SIZE[0]))
    dist = np.sqrt((xx-x) ** 2 + (yy-y) ** 2)

    if binary: # Binary mask: 1 inside circle, o outside
        mask = (dist <= radius).astype(np.float32)

    else: # Gaussian probability map
        mask = np.exp(-dist ** 2 / (2 * (radius / 2) ** 2))
        mask = mask.astype(np.float32)
    mask = cv2.resize(mask, NEW_SIZE, interpolation=cv2.INTER_AREA)
    return mask


def create_full_ball_dataframe():
    """
    Only needs to be run once to create the ball data for all the shots.

    Returns:
        - Dataframe with ball data for all the shots
    """

    ball_data = pd.DataFrame()
    # Go over each scene
    for scene in os.listdir('scenes'):
           
        scene_id = os.path.splitext(scene)[0]
        df = load_ball_position_data(os.path.join('illumination_gt', scene_id + '.csv'))
        ball_data = pd.concat([ball_data, df], axis=0)

    pd.save_csv(ball_data, 'ball_data.csv', index=False)
    return ball_data


def get_image_paths(ball_data):
    """
    Returns:
        - Dictionary of the paths for input images (scenes)
        - List of tuples (scene_id, shot_id, path) for output images (shots)
    """
    input_paths = {}
    output_paths = []
    shots = ball_data['image_name'].values
    # Go over each scene
    for scene in os.listdir('scenes'):
        input_path = os.path.join('scenes', scene)
        scene_id = os.path.basename(input_path)[:-4]
        input_paths[scene_id] = input_path
        scene_id = os.path.splitext(scene)[0]
        shots_dir = os.path.join('scenes_shots', scene_id)

        # Go over each shot
        for shot in os.listdir(shots_dir):
            output_path = os.path.join(shots_dir, shot)
            shot_id = os.path.splitext(shot)[0]
            if shot_id in shots:
                output_paths.append((scene_id, shot_id, output_path))

    print(f"Found {len(input_paths)} scenes.")
    print(f"Found {len(output_paths)} scene shots.")

    return input_paths, output_paths

def extract_features_from_scene(image_tensor, resnet_feature_extractor, feature_proj):
    """
    Extracts features from the scene using the ResNet18 model.
    """
    if image_tensor.ndim == 3:
        image_tensor = image_tensor.unsqueeze(0) 

    with torch.no_grad():
        features = resnet_feature_extractor(image_tensor).squeeze(-1).squeeze(-1)  # [B, 512]
        projected = feature_proj(features)
        projected = projected.view(-1, 256, 8, 8)  # [B, 256, 8, 8]
        upsampled = F.interpolate(projected, size=(32, 32), mode='bilinear', align_corners=False) # [B, 256, 32, 32]
    
    return upsampled.squeeze(0)  # [1, 256, 32, 32]

def load_resnet18():
    # Load ResNet18 for global features
    resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    resnet.eval()
    for param in resnet.parameters():
        param.requires_grad = False

    # Remove the last layer
    resnet_feature_extractor = nn.Sequential(*list(resnet.children())[:-1])  # Global AvgPool output

    return resnet_feature_extractor


