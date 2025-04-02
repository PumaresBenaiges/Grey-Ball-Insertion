import DatasetCreation as DC
import unet
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image

def train_model(model, dataloader, epochs=100):
    print('Start training')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = model.cuda()
    model.to(device)
    criterion = nn.MSELoss()  # or nn.L1Loss() for pixel-wise error
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        epoch_loss = 0.0  # Track total loss for averaging
        for idx, (input_image, mask, target_image) in enumerate(dataloader):
            print(f"Batch {idx}")
            input_image, mask, target_image = input_image.to(device), mask.to(device), target_image.to(device)
            input_tensor = torch.cat([input_image, mask], dim=1)

            # Forward pass, CUDA OOM error
            output = model(input_tensor)
            loss = criterion(output, target_image)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Example inside training loop
        if epoch in (1, 25, 50, 75, 99):
            save_image(torch.cat((output[:4], target_image[:4]), dim=0), f"comparison{epoch}.jpg", nrow=4, normalize=True)
        print(f"Epoch {epoch}, Loss: {loss.item()}")

if __name__ == '__main__':

    # Create  dataset and dataloader
    input_paths, ball_data, output_paths = DC.get_image_paths()
    input_images = DC.load_input_scenes(input_paths)
    new_output = output_paths[:64]
    dataset = DC.SceneDataset(input_images, ball_data, output_paths)
    dataloader = DC.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)
    print('Dataloader Created')

    # Create and train model
    # model = unet.SmallUNet(input_channels=4, output_channels=3)
    model = unet.UNet2()
    print('Model Created')
    torch.cuda.empty_cache()
    train_model(model, dataloader, 5)