import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import DatasetCreation as DC
from torchvision.utils import save_image


class UNet(nn.Module):
    def __init__(self, input_channels=4, output_channels=3):
        super(UNet, self).__init__()

        features = [64, 128, 256, 512] # Features: num of filters in each block

        # Encoder (Downsampling path)
        self.encoder = nn.ModuleList()
        for feature in features:
            self.encoder.append(self._block(input_channels, feature))
            input_channels = feature

        # Bottleneck (bridge between encoder and decoder)
        self.bottleneck = self._block(features[-1], features[-1]*2)

        # Decoder (Upsampling path)
        self.decoder = nn.ModuleList()
        for feature in reversed(features):
            self.decoder.append(
                nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2)
            )
            self.decoder.append(self._block(feature*2, feature))  # After concatenation

        # Final output layer
        self.final_conv = nn.Conv2d(features[0], output_channels, kernel_size=1)

    def _block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        skip_connections = []

        # Downsampling
        for down in self.encoder:
            x = down(x)
            skip_connections.append(x)
            x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = self.bottleneck(x)

        # Reverse skip connections for upsampling
        skip_connections = skip_connections[::-1]

        # Upsampling
        for idx in range(0, len(self.decoder), 2):
            x = self.decoder[idx](x)  # Transposed conv (upsample)
            skip_connection = skip_connections[idx // 2]

            # Handle size mismatch (due to rounding during downsampling/upsampling)
            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:])

            # Concatenate skip connection
            x = torch.cat((skip_connection, x), dim=1)

            # Apply conv block
            x = self.decoder[idx + 1](x)

        return torch.sigmoid(self.final_conv(x))  # Sigmoid for normalized output (0-1)

def train_model(model, dataloader, epochs=100):
    model = model.cuda()
    criterion = nn.MSELoss()  # or nn.L1Loss() for pixel-wise error
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(epochs):
        for input_image, mask, target_image in dataloader:
            input_image, mask, target_image = input_image.cuda(), mask.cuda(), target_image.cuda()
            input = torch.cat([input_image, mask], dim=1)

            # Forward pass
            output = model(input)
            loss = criterion(output, target_image)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # Example inside training loop
        if epoch in (1, 25, 50, 75, 99):
            save_image(torch.cat((output[:4], target_image[:4]), dim=0), f"comparison{epoch}.jpg", nrow=4, normalize=True)
        print(f"Epoch {epoch}, Loss: {loss.item()}")

if __name__ == '__main__':

    # Create  dataset and dataloader
    input_paths, ball_data, output_paths = DC.get_image_paths()
    input_images = DC.load_input_scenes(input_paths)
    dataset = DC.SceneDataset(input_images, ball_data, output_paths)
    dataloader = DC.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=16, pin_memory=True)
    print('Dataloader Created')
    # Create and train model
    model = UNet(input_channels=4, output_channels=3)
    print('Model Created')
    train_model(model, dataloader, 100)