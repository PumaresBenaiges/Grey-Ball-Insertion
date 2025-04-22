import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


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


# Model that we are currently using
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

"""class VGGFeatures(nn.Module):
    def __init__(self):
        super(VGGFeatures, self).__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        self.features = vgg.features
        self.avgpool = vgg.avgpool
        self.fc = nn.Sequential(*list(vgg.classifier.children())[:-1])  # Up to fc7 (4096) remove last layer

        for param in self.parameters():
            param.requires_grad = False  # Freeze VGG

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x  # [B, 4096]"""


class UNet2(nn.Module):
    def __init__(self, inc=3, outc=3):
        super(UNet2, self).__init__()
        # self.vgg = VGGFeatures()

        # ResNet18 for global features (used optionally)
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        for param in resnet.parameters():
            param.requires_grad = False
        self.resnet_feature_extractor = nn.Sequential(*list(resnet.children())[:-1])  # Global AvgPool output
        self.feature_proj = nn.Linear(512, 32 * 32)  # Map global features to spatial map (reshape to 32x32)


        self.conv1 = DoubleConv(inc, 32)
        self.down1 = DownLayer(32, 64)
        self.down2 = DownLayer(64, 128)
        self.down3 = DownLayer(128, 256)
        # self.down4 = DownLayer(512, 1024)  # bottleneck#

        # New FC to map 4096 -> 1024 (to match UNet bottleneck channels)
        self.vgg_to_bottleneck = nn.Linear(4096, 1024 * 8 * 8)

        # self.up1 = UpLayer(1024, 512)
        self.up2 = UpLayer(256, 128)
        self.up3 = UpLayer(128, 64)
        self.up4 = UpLayer(64, 32)

        self.last_conv = nn.Conv2d(32, outc, 1)
        self.activation = nn.Sigmoid()

    def forward(self, cropped_input, full_input):
        # VGG features
        # vgg_feat = self.vgg(vgg_input)  # [B, 4096]
        # vgg_mapped = self.vgg_to_bottleneck(vgg_feat).view(-1, 1024, 8, 8)

        # Extract global features
        batch_size = full_input.size(0)
        global_feat = self.resnet_feature_extractor(full_input).squeeze(-1).squeeze(-1)  # [B, 512]
        global_feat = self.feature_proj(global_feat).view(batch_size, 1, 32, 32)  # Reshape to [B, 1, 32, 32]
        global_feat = F.interpolate(global_feat, size=(256, 256), mode='bilinear', align_corners=False)

        # UNet encoding
        x1 = self.conv1(cropped_input)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        # x5 = self.down4(x4)

        # Combine VGG features into bottleneck
        # x_bottleneck = x4 + vgg_mapped  # or torch.cat([...], dim=1) and adjust channels

        x1_up = self.up1(x3, x4)
        x2_up = self.up2(x2, x1_up)
        x3_up = self.up3(x1, x2_up)
        # x4_up = self.up4(x1, x3_up)

        output = self.last_conv(x3_up + global_feat)  # Fuse global features here (simple addition)
        return self.activation(output)



