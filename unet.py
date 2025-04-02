import torch
import torch.nn as nn
import torch.nn.functional as F


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

class SmallUNet(nn.Module):
    def __init__(self, input_channels=4, output_channels=3):
        super(SmallUNet, self).__init__()

        features = [32, 64, 128, 256]  # Smaller number of filters

        # Encoder (Downsampling)
        self.encoder = nn.ModuleList()
        for feature in features:
            self.encoder.append(self._block(input_channels, feature))
            input_channels = feature

        # Bottleneck (Fixed: Doesn't increase channels)
        self.bottleneck = self._block(features[-1], features[-1])

        # Decoder (Upsampling - Fixed to match correct number of channels)
        self.decoder = nn.ModuleList()
        for feature in reversed(features[:-1]):  # Avoid doubling channels
            self.decoder.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.decoder.append(self._block(feature * 2, feature))

        # Final output layer
        self.final_conv = nn.Conv2d(features[0], output_channels, kernel_size=1)

    def _block(self, in_channels, out_channels):
        """Fixed convolutional block."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        skip_connections = []

        # Downsampling
        for down in self.encoder:
            x = down(x)
            skip_connections.append(x)
            x = F.avg_pool2d(x, kernel_size=2, stride=2)  # More memory-efficient than max pooling

        x = self.bottleneck(x)

        # Upsampling (Fix: Skip last layer in reversed encoder)
        skip_connections = skip_connections[::-1]
        for idx in range(0, len(self.decoder), 2):
            x = self.decoder[idx](x)  # Transposed Conv
            skip_connection = skip_connections[idx // 2 + 1]  # Fix indexing to match shapes

            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:])  # Ensure proper shape

            x = torch.cat((skip_connection, x), dim=1)  # Concatenate
            x = self.decoder[idx + 1](x)

        return torch.sigmoid(self.final_conv(x))  # Normalize output




class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Up, self).__init__()
        self.up_scale = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)

    def forward(self, x1, x2):
        x2 = self.up_scale(x2)

        diffY = x1.size()[2] - x2.size()[2]
        diffX = x1.size()[3] - x2.size()[3]

        x2 = F.pad(x2, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return x


class DownLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DownLayer, self).__init__()
        self.pool = nn.MaxPool2d(2, stride=2, padding=0)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(self.pool(x))
        return x


class UpLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UpLayer, self).__init__()
        self.up = Up(in_ch, out_ch)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        a = self.up(x1, x2)
        x = self.conv(a)
        return x


class UNet2(nn.Module):
    def __init__(self, inc=4, outc=3):
        super(UNet2, self).__init__()
        self.conv1 = DoubleConv(inc, 64)
        self.down1 = DownLayer(64, 128)
        self.down2 = DownLayer(128, 256)
        self.down3 = DownLayer(256, 512)
        # self.down4 = DownLayer(512, 1024)
        # self.up1 = UpLayer(1024, 512)
        self.up2 = UpLayer(512, 256)
        self.up3 = UpLayer(256, 128)
        self.up4 = UpLayer(128, 64)
        self.last_conv = nn.Conv2d(64, outc, 1)
        self.activation = nn.Sigmoid() # to get output normalized images

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        # x5 = self.down4(x4)
        # x1_up = self.up1(x4, x5)
        x2_up = self.up2(x3, x4)
        x3_up = self.up3(x2, x2_up)
        x4_up = self.up4(x1, x3_up)
        output = self.last_conv(x4_up)
        return output


