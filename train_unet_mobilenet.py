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

RUN_NAME = "cut_ball"

def masked_l1_loss(output, target, mask):
    output_masked = output * mask
    target_masked = target * mask
    return torch.abs(output_masked - target_masked).sum() / mask.sum().clamp(min=1)

def train_model(epochs=100):
    # Set parameters
    batch_size = 16
    num_workers = 0

    # Load dataframes and images
    ball_data_train = pd.read_csv('ball_data_new.csv')
    ball_data_val = pd.read_csv('ball_data_val.csv')
    transformations_data = pd.read_csv('homography_transformation.csv')

    input_paths_train, output_paths_train = utils.get_image_paths(ball_data_train)
    input_paths_val, output_paths_val = utils.get_image_paths(ball_data_val)
    input_paths_val = {
        'seat_rows': 'scenes\\seat_rows.NEF'
    }

    input_images_train = utils.load_input_scenes(input_paths_train)
    input_images_val = utils.load_input_scenes(input_paths_val)
    print('Data and images loaded.')

    # Create dataset, train/test split and dataloaders
    train_dataset = DC.SceneDataset(input_images_train, ball_data_train, output_paths_train, transformations_data)
    val_dataset = DC.SceneDataset(input_images_val, ball_data_val, output_paths_val, transformations_data)
    train_size = len(train_dataset)
    val_size = len(val_dataset)

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
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    start_time = time.time()

    # Get fixed samples for visualization
    fixed_train_batch = next(iter(train_loader))
    fixed_val_batch = next(iter(val_loader))

    fixed_input_image, fixed_input_cropped, fixed_target_image, fixed_mask = [
        t[:6].to(device) for t in fixed_train_batch
    ]
    fixed_val_input_image, fixed_val_input_cropped, fixed_val_target_image, fixed_val_mask = [
        t[:6].to(device) for t in fixed_val_batch
    ]

    # Clone fixed masks to prevent modification across epochs  <-- CHANGED
    fixed_mask = fixed_mask.clone().detach()  # <-- CHANGED
    fixed_val_mask = fixed_val_mask.clone().detach()  # <-- CHANGED

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        epoch_start_time = time.time()

        train_loss = 0.0
        val_loss = 0.0
        for idx, (input_image, input_image_cropped, target_image, mask) in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch+1}")):
            input_image, input_image_cropped, target_image, mask = input_image.to(device), input_image_cropped.to(device), target_image.to(device), mask.to(device)
            mask = mask.unsqueeze(1)             # [B, 1, H, W]
            mask = mask.expand(-1, 3, -1, -1)

            # Forward pass
            output = model(input_image_cropped, input_image)
            output = output * mask
            loss = masked_l1_loss(output, target_image, mask)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation loop
        model.eval()
        with torch.no_grad():
            for input_image, input_image_cropped, target_image, mask in val_loader:
                input_image, input_image_cropped, target_image, mask = input_image.to(device), input_image_cropped.to(device), target_image.to(device), mask.to(device)
                mask = mask.unsqueeze(1)
                mask = mask.expand(-1, 3, -1, -1)
                output_val = model(input_image_cropped, input_image)
                output_val = output_val * mask
                loss = masked_l1_loss(output_val, target_image, mask)
                val_loss += loss.item()

        epoch_time = time.time() - epoch_start_time
        print(f"Train Loss: {train_loss / len(train_loader)}, Val Loss: {val_loss / len(val_loader)}, Time: {time.time()-start_time}s")

        if epoch in (0, 1, 2, 3, 4, 5, 6, 7, 15, 20, 25, 50, 75, epochs-1):
            model.eval()
            with torch.no_grad():
                output_train = model(fixed_input_cropped, fixed_input_image)
                fixed_mask_view = fixed_mask.view(fixed_mask.size(0), 1, fixed_mask.size(-2), fixed_mask.size(-1))
                output_train = output_train * fixed_mask_view.expand(-1, 3, -1, -1)
                save_image(output_train, f"train_out_{epoch}.jpg", nrow=3, normalize=True)
                save_image(fixed_target_image, f"train_gt_{epoch}.jpg", nrow=3, normalize=True)

                output_val = model(fixed_val_input_cropped, fixed_val_input_image)
                fixed_val_mask_view = fixed_val_mask.view(fixed_val_mask.size(0), 1, fixed_val_mask.size(-2), fixed_val_mask.size(-1))
                output_val = output_val * fixed_val_mask_view.expand(-1, 3, -1, -1)
                save_image(output_val, f"val_out_{epoch}.jpg", nrow=3, normalize=True)
                save_image(fixed_val_target_image, f"val_gt_{epoch}.jpg", nrow=3, normalize=True)

        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss
        }
        torch.save(checkpoint, f'checkpoint_epoch_{epoch+1}.pt')

        writer.add_scalar('Loss/train', train_loss / len(train_loader), epoch)
        writer.add_scalar('Loss/val', val_loss / len(val_loader), epoch)
        writer.add_scalar('Time/epoch_sec', epoch_time, epoch)

    del loss, output
    torch.cuda.empty_cache()

    total_time = time.time() - start_time
    print(f"Training complete in {total_time / 60:.2f} minutes ({total_time:.2f} seconds)")


if __name__ == '__main__':
    train_model(100)
