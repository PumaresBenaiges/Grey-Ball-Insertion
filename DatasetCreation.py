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


from align_scenes import apply_homography_transformation
import utils

IM_SIZE=(2844,4284,3)
NEW_SIZE=(256,256)

# PyTorch Dataset Class
class SceneDataset(Dataset):
    """
    - input_images: 30 images of scenes
    - df: dataframe with the ball data for all shots
    - output_images_paths: contains list of scene_id, shot_id and path of the image (shot)
    """
    def __init__(self, input_images, ball_df, output_images_paths, transformations_df, features=None, model_name='unet_mobilenet'):
        self.features = features
        self.input_images = input_images
        self.df = ball_df
        self.output_images_paths = output_images_paths
        self.transformation_data = transformations_df
        self.model_name = model_name

    def __len__(self):
        return len(self.output_images_paths)

    def __getitem__(self, idx):
        """
        Each item of dataset is composed by:
        - Input image (scene) -> now features extracted from image
        - Input image cropped (scene)
        - Output image cropped (shot)
        """
        scene_id, shot_id, path = self.output_images_paths[idx]
        ball_data = self.df[self.df['image_name'] == shot_id]
        shot_id_transformation = shot_id + '.NEF'
        ball_scene_t = self.transformation_data[self.transformation_data['Scene'] == scene_id]
        ball_transformation = ball_scene_t[ball_scene_t['Scene ID'] == shot_id_transformation]
        H_values = ball_transformation.iloc[0, 2:].values.tolist()
        H = np.array(H_values).reshape(3, 3)

        # Load already transformed scenes
        #input_image_c = load_image_jpg('scenes_aligned/' + scene_id + '/' + shot_id+'.jpg')
        #input_image_cropped = crop_center(input_image_c, ball_data)

        # Load, transform and crop scene (input_image)
        input_image = self.input_images[scene_id]
        input_image = apply_homography_transformation(input_image, H)
        input_image_cropped, mask = utils.crop_center(input_image, ball_data)

        # Load and crop shot (output_image)
        output_image = utils.load_image(path)
        output_image, _ = utils.crop_center(output_image, ball_data)

        # Normalize to [0,1]
        input_image_cropped = torch.from_numpy(input_image_cropped).permute(2, 0, 1).float() / 255.0
        output_image = torch.from_numpy(output_image).permute(2, 0, 1).float() / 255.0

        if self.model_name == 'unet':
            # we use the extracted features
            input = self.features[scene_id]
        else:
            # Resize, convert to tensor and normalize to mean and std of resnet
            input_image = cv2.resize(input_image, (224,224), interpolation=cv2.INTER_AREA)
            input_image = torch.from_numpy(input_image).permute(2, 0, 1).float() / 255.0
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            input = input_image.sub(mean).div(std)

        return input, input_image_cropped, output_image, mask

if __name__ == '__main__':
    # Load dataframes
    ball_data = pd.read_csv('ball_data.csv')
    transformations_data = pd.read_csv('homography_transformation.csv')

    input_paths, output_paths = utils.get_image_paths(ball_data)
    input_images = utils.load_input_scenes(input_paths)
    print('Input images loaded.')

    # For each scene, extract the features using the ResNet18 or MobileNetV3 model
    resnet, feature_proj = utils.load_resnet18() # or load_mobilenetv3()
    features = {}
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    for scene_id, image in input_images.items():
        image = cv2.resize(image, (224,224), interpolation=cv2.INTER_AREA)
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        image = image.sub(mean).div(std)
        feat = utils.extract_features_from_scene(image, resnet, feature_proj)
        features[scene_id]= feat
    print('Input images features extracted.')

    # Create dataset and dataloader
    dataset = SceneDataset(input_images, ball_data, output_paths, transformations_data, features, 'unet')
    #dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0, pin_memory=True)
    #print(dataset[0])
    # Get one batch as example and check dataset is created correctly
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0, pin_memory=True)

    for batch_idx, (feat, input_images, input_cropped, output_cropped) in enumerate(dataloader):
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



