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

from align_scenes import apply_homography_transformation

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
    for index, (scene_id, input_path) in enumerate(input_paths.items()):
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

def get_image_paths():
    """
    Returns:
        - Dictionary of the paths for input images (scenes)
        - List of the scene_id, shot_id and path for output images (shots)
        - Dataframe with ball data for all the shots
    """
    input_paths = {}
    output_paths = []
    ball_data = pd.DataFrame()

    # Go over each scene
    for scene in os.listdir('scenes'):
        input_path = os.path.join('scenes', scene)
        scene_id = os.path.basename(input_path)[:-4]
        input_paths[scene_id] = input_path
        scene_id = os.path.splitext(scene)[0]
        shots_dir = os.path.join('scenes_shots', scene_id)

        df = load_ball_position_data(os.path.join('illumination_gt', scene_id + '.csv'))
        ball_data = pd.concat([ball_data, df], axis=0)

        # Go over each shot
        for shot in os.listdir(shots_dir):
            output_path = os.path.join(shots_dir, shot)
            shot_id = os.path.splitext(shot)[0]
            if shot_id in df['image_name'].values:
                output_paths.append((scene_id, shot_id, output_path))

    print(f"Found {len(input_paths)} scenes.")
    print(f"Found {len(output_paths)} scene shots.")

    return input_paths, ball_data, output_paths

# PyTorch Dataset Class
class SceneDataset(Dataset):
    """
    - input_images: 30 images of scenes
    - df: dataframe with the ball data for all shots
    - output_images_paths: contains list of scene_id, shot_id and path of the image (shot)
    """
    def __init__(self, input_images, df, output_images_paths, ball_transformation_data):
        self.input_images = input_images
        self.df = df
        self.output_images_paths = output_images_paths
        self.ball_transformation_data = ball_transformation_data

    def __len__(self):
        return len(self.output_images_paths)

    def __getitem__(self, idx):
        """
        Each item of dataset is composed by:
        - Input image (scene)
        - Input image cropped (scene)
        - Output image cropped (shot)
        """
        scene_id, shot_id, path = self.output_images_paths[idx]
        ball_data = self.df[self.df['image_name'] == shot_id]
        shot_id_transformation = shot_id + '.NEF'
        ball_scene_t = self.ball_transformation_data[self.ball_transformation_data['Scene'] == scene_id]
        ball_transformation = ball_scene_t[ball_scene_t['Scene ID'] == shot_id_transformation]
        H_values = ball_transformation.iloc[0, 2:].values.tolist()
        H = np.array(H_values).reshape(3, 3)

        input_image = self.input_images[scene_id]
        input_image = apply_homography_transformation(input_image, H)
        input_image_cropped = crop_center(input_image, ball_data)
        input_image = cv2.resize(input_image, (224,224), interpolation=cv2.INTER_AREA)

        output_image = load_image(path)
        output_image = crop_center(output_image, ball_data)
        #input_image_c = load_image_jpg('scenes_aligned/' + scene_id + '/' + shot_id+'.jpg')
        #input_image_cropped = crop_center(input_image_c, ball_data)
        

        # Convert to tensor and normalize
        # Cropped image: To [0, 1]
        # Whole image: to mean and std of resnet
        input_image = torch.from_numpy(input_image).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        input_image = input_image.sub(mean).div(std)
        input_image_cropped = torch.from_numpy(input_image_cropped).permute(2, 0, 1).float() / 255.0
        output_image = torch.from_numpy(output_image).permute(2, 0, 1).float() / 255.0
       

        return input_image, input_image_cropped, output_image

if __name__ == '__main__':
    csv_file_path = 'homography_transformation.csv'
    input_paths, ball_data, output_paths = get_image_paths()
    ball_transformation_data = load_ball_transformation_data(csv_file_path)
    input_images = load_input_scenes(input_paths)
    print('Input images loaded.')
    # ball_data.to_csv('ball_data.csv')

    # Create dataset and dataloader
    dataset = SceneDataset(input_images, ball_data, output_paths, ball_transformation_data)
    #dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0, pin_memory=True)
    #print(dataset[0])
    # Get one batch as example and check dataset is created correctly
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0, pin_memory=True)

    for batch_idx, (input_images, input_cropped, output_cropped) in enumerate(dataloader):
        print(f"Batch {batch_idx}")
        print(f"Input images shape: {input_images.shape}")  # (B, 3, H, W)
        print(f"Output cropped shape: {output_cropped.shape}")  # (B, 3, H, W)
        print(f"Input cropped shape: {input_cropped.shape}")  # (B, 1, H, W)
        print(f"Input image range: {input_images.min()} - {input_images.max()}")
        print(f"Output cropped range: {output_cropped.min()} - {output_cropped.max()}")
        print(f"Input cropped range: {input_cropped.min()} - {input_cropped.max()}")

        # Save grid of images
        vutils.save_image(input_images, 'input_samples.png', nrow=4, normalize=True)
        vutils.save_image(input_cropped, 'input_crop_samples.png', nrow=4, normalize=True)
        vutils.save_image(output_cropped, 'output_samples.png', nrow=4, normalize=True)

        if batch_idx == 0:  # check only first batch
            break



