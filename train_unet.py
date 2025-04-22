import DatasetCreation as DC
import time
import unet
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
from torchvision.utils import save_image
from tqdm import tqdm


def train_model(epochs=100):
    # Create dataset, train/test split and dataloaders
    input_paths, ball_data, output_paths = DC.get_image_paths()
    input_images = DC.load_input_scenes(input_paths)
    dataset = DC.SceneDataset(input_images, ball_data, output_paths)
    total_size = len(dataset)
    val_size = int(0.2 * total_size)
    train_size = total_size - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Dataloaders
    train_loader = DC.DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DC.DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)
    print(f'Dataloader Created: {train_size} train samples, {val_size} validation samples')

    # Create and train model
    model = unet.UNet2()
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.MSELoss()  # or nn.L1Loss() for pixel-wise error
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    start_time = time.time()
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        train_loss = 0.0
        val_loss = 0.0
        for idx, (input_image, input_image_cropped, target_image) in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch+1}")):
            input_image, input_image_cropped, target_image = input_image.to(device), input_image_cropped.to(device), target_image.to(device)
            # input_tensor = torch.cat([input_image, mask], dim=1)

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
        if epoch in (1, 25, 50, 75, epochs-1):
            save_image(torch.cat((output[:4], target_image[:4]), dim=0), f"comparison{epoch}.jpg", nrow=4, normalize=True)
        print(f"Train Loss: {train_loss / len(train_loader)}, Val Loss: {val_loss / len(val_loader)}, Time: {time.time()-start_time}s")

    total_time = time.time() - start_time
    print(f"Training complete in {total_time / 60:.2f} minutes ({total_time:.2f} seconds)")


if __name__ == '__main__':

    train_model(5)
