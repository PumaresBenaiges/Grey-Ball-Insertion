import DatasetCreation as DC
import utils
import time
import unet
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
from torchvision.utils import save_image
from tqdm import tqdm
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter

RUN_NAME = "experiment1"

def train_model(epochs=100):
    # Set parameters
    batch_size = 16
    num_workers = 0

    # Load dataframes and images
    ball_data = pd.read_csv('ball_data.csv')
    transformations_data = pd.read_csv('homography_transformation.csv')
    input_paths, output_paths = utils.get_image_paths(ball_data)
    input_images = utils.load_input_scenes(input_paths)
    print('Data and images loaded.')
    
    # Create dataset, train/test split and dataloaders
    dataset = DC.SceneDataset(input_images, ball_data, output_paths, transformations_data)
    total_size = len(dataset)
    val_size = int(0.2 * total_size)
    train_size = total_size - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    writer = SummaryWriter(log_dir=f'runs/{RUN_NAME}') # Log data writer

    # Dataloaders
    train_loader = DC.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DC.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    print(f'Dataloader Created: {train_size} train samples, {val_size} validation samples')

    # Create and train model
    model = unet.UNetMobileNetV3()
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.L1Loss()  # or nn.L1Loss() for pixel-wise error
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    start_time = time.time()
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        epoch_start_time = time.time()

        train_loss = 0.0
        val_loss = 0.0
        for idx, (input_image, input_image_cropped, target_image, mask) in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch+1}")):
            input_image, input_image_cropped, target_image, mask = input_image.to(device), input_image_cropped.to(device), target_image.to(device), mask.to(device)
            # input_tensor = torch.cat([input_image, mask], dim=1)
            mask = mask.unsqueeze(1)             # [B, 1, H, W]
            mask = mask.expand(-1, 3, -1, -1)
            # Forward pass
            output = model(input_image_cropped, input_image)
            output = output * mask  # Apply mask to output
            #save_image(torch.cat((output[:4], mask[:4]), dim=0), f"test{epoch}.jpg", nrow=4, normalize=True)
            loss = criterion(output, target_image)
            loss = loss 

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

        epoch_time = time.time() - epoch_start_time

        # Example inside training loop
        if epoch in (0,1,2,3,4,5,6,7,8,25, 50, 75, epochs-1):
            save_image(torch.cat((output[:4], target_image[:4]), dim=0), f"comparison{epoch}.jpg", nrow=4, normalize=True)
        print(f"Train Loss: {train_loss / len(train_loader)}, Val Loss: {val_loss / len(val_loader)}, Time: {time.time()-start_time}s")

        # SAVE CHECKPOINT AND LOG DATA
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss
        }
        torch.save(checkpoint, 'checkpoint_epoch_{epoch+1}.pt')

        writer.add_scalar('Loss/train', train_loss / len(train_loader), epoch)
        writer.add_scalar('Loss/val', val_loss / len(val_loader), epoch)
        writer.add_scalar('Time/epoch_sec', epoch_time, epoch)

    # AFTER TRAINING OF THE EPOCH IS DONE CLEAR CUDA MEMORY
    del loss, output  # explicitly release tensors
    torch.cuda.empty_cache()

    total_time = time.time() - start_time
    print(f"Training complete in {total_time / 60:.2f} minutes ({total_time:.2f} seconds)")


if __name__ == '__main__':
    train_model(10)