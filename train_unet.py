import DatasetCreation as DC
import utils
import time
import unet
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
from torchvision.utils import save_image
from tqdm import tqdm
import pandas as pd
import cv2

def train_model(epochs=100):
    # Load dataframes and images
    ball_data = pd.read_csv('ball_data.csv')
    transformations_data = pd.read_csv('homography_transformation.csv')
    input_paths, output_paths = utils.get_image_paths(ball_data)
    input_images = utils.load_input_scenes(input_paths)
    print('Data and images loaded.')

    # For each scene, extract the features using the ResNet18 model
    resnet, feature_proj = utils.load_resnet18()
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
    

    dataset = DC.SceneDataset(input_images, ball_data, output_paths, transformations_data, features, 'unet')
    total_size = len(dataset)
    val_size = int(0.2 * total_size)
    train_size = total_size - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Dataloaders
    train_loader = DC.DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DC.DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)
    print(f'Dataloader Created: {train_size} train samples, {val_size} validation samples')

    # Create and train model
    model = unet.UNet()
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.MSELoss()  # or nn.L1Loss() for pixel-wise error
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    start_time = time.time()
    print("Starting training...")
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        train_loss = 0.0
        val_loss = 0.0
        for idx, (input_image, input_image_cropped, target_image) in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch+1}")):
            input_image, input_image_cropped, target_image = input_image.to(device), input_image_cropped.to(device), target_image.to(device)
            # input_tensor = torch.cat([input_image, mask], dim=1)
            mida = input_image.size()
            print(mida)
            # Forward pass
            output = model(input_image_cropped, input_image)
            loss = criterion(output, target_image)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            # Validation loop
            """model.eval()
            with torch.no_grad():
                for input_image, mask, target_image in val_loader:
                    input_image, mask, target_image = input_image.to(device), mask.to(device), target_image.to(device)
                    input_tensor = torch.cat([input_image, mask], dim=1)
                    output_val = model(input_tensor)
                    loss = criterion(output_val, target_image)
                    val_loss += loss.item()"""

        # Example inside training loop
        if epoch in (0, 1, 2, 3, 25, 50, 75, epochs-1):
            save_image(torch.cat((output[:4], target_image[:4]), dim=0), f"comparison{epoch}.jpg", nrow=4, normalize=True)
        print(f"Train Loss: {train_loss / len(train_loader)}, Val Loss: {val_loss / len(val_loader)}, Time: {time.time()-start_time}s")

    total_time = time.time() - start_time
    print(f"Training complete in {total_time / 60:.2f} minutes ({total_time:.2f} seconds)")


if __name__ == '__main__':

    train_model(5)
