import os
import rawpy
import numpy as np
import pandas as pd
import torch
import cv2
from matplotlib.image import interpolations_names
from torch.utils.data import Dataset, DataLoader
import torchvision.utils as vutils

IM_SIZE=(2844,4284,3)
# NEW_SIZE=(2848,4288)
NEW_SIZE=(256,256)

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
            image = np.rot90(image)
        image = cv2.resize(image, NEW_SIZE, interpolation=cv2.INTER_AREA)
    return image

def load_input_scenes(input_paths):
    """
    Args:
        List of paths of the input scenes images.
    Returns:
        Dictionary with scene_id and loaded image.
    """
    # TODO preprocessing, normalization (done afterwards)?
    # Load and process input image
    input_images = {}
    for input_path in input_paths:
        scene_id = os.path.basename(input_path)[:-4]
        input_image = load_image(input_path)
        input_images[scene_id] = input_image
    return input_images

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
        - List of the paths for input images (scenes)
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
    def __init__(self, input_images, df, output_images_paths):
        self.input_images = input_images
        self.df = df
        self.output_images_paths = output_images_paths

    def __len__(self):
        return len(self.output_images_paths)

    def __getitem__(self, idx):
        """
        Each item of dataset is composed by:
        - Input image (scene)
        - Mask (ball position)
        - Output image (shot)
        """
        scene_id, shot_id, path = self.output_images_paths[idx]
        # input_image = self.input_images[scene_id].copy()
        input_image =load_image(self.input_images[scene_id])
        output_image = load_image(path)
        ball_data = self.df[self.df['image_name'] == shot_id]

        if ball_data.empty:
            print('No data for image'+ scene_id+shot_id)

        # Generate mask for ball position
        mask = create_probability_map(ball_data)

        # Convert to tensor and normalize to [0, 1]
        input_image = torch.from_numpy(input_image).permute(2, 0, 1).float() / 255.0
        output_image = torch.from_numpy(output_image).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(mask).unsqueeze(0).float()  # (1, H, W)

        return input_image, mask, output_image

if __name__ == '__main__':
    input_paths, ball_data, output_paths = get_image_paths()
    input_images = load_input_scenes(input_paths)
    print('Input images loaded.')
    # ball_data.to_csv('ball_data.csv')

    # Create dataset and dataloader
    dataset = SceneDataset(input_images, ball_data, output_paths)
    #dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0, pin_memory=True)
    print(dataset[0])
    # Get one batch as example and check dataset is created correctly
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)

    for batch_idx, (input_images, masks, output_images) in enumerate(dataloader):
        print(f"Batch {batch_idx}")
        print(f"Input images shape: {input_images.shape}")  # (B, 3, H, W)
        print(f"Output images shape: {output_images.shape}")  # (B, 3, H, W)
        print(f"Masks shape: {masks.shape}")  # (B, 1, H, W)
        print(f"Input image range: {input_images.min()} - {input_images.max()}")
        print(f"Output image range: {output_images.min()} - {output_images.max()}")
        print(f"Mask range: {masks.min()} - {masks.max()}")

        # Save grid of images
        # vutils.save_image(input_images, 'input_samples.png', nrow=4, normalize=True)
        # vutils.save_image(masks, 'mask_samples.png', nrow=4, normalize=True)
        # vutils.save_image(output_images, 'output_samples.png', nrow=4, normalize=True)

        if batch_idx == 0:  # check only first batch
            break




