import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


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

class DownLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DownLayer, self).__init__()
        self.pool = nn.MaxPool2d(2, stride=2, padding=0)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(self.pool(x))
        return x

class UpLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, inc=3, outc=3):
        super(UNet, self).__init__()

        # # ResNet18 for global features
        # resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # for param in resnet.parameters():
        #     param.requires_grad = False
        # self.resnet_feature_extractor = nn.Sequential(*list(resnet.children())[:-1])  # Global AvgPool output
        # #self.feature_proj = nn.Linear(512, 32 * 32)  # Map global features to spatial map (reshape to 32x32)
        # self.feature_proj = nn.Linear(512, 256 * 8 * 8)


        self.conv1 = DoubleConv(inc, 32)
        self.down1 = DownLayer(32, 64)
        self.down2 = DownLayer(64, 128)
        self.down3 = DownLayer(128, 256)

        self.bottleneck = DoubleConv(256+256, 256) # 256 of encoder + 256 resnet output

        self.up1 = UpLayer(256, 128)
        self.up2 = UpLayer(128, 64)
        self.up3 = UpLayer(64, 32)

        self.last_conv = nn.Conv2d(32, outc, 1)
        self.activation = nn.Sigmoid()

    def forward(self, cropped_input, full_input):

        batch_size = cropped_input.size(0)

        # === Encoder ===
        x1 = self.conv1(cropped_input)  # [B, 32, 256, 256]
        x2 = self.down1(x1)  # [B, 64, 128, 128]
        x3 = self.down2(x2)  # [B, 128, 64, 64]
        x4 = self.down3(x3)  # [B, 256, 32, 32]

        # # === Global Features ===
        # global_feat = self.resnet_feature_extractor(full_input).squeeze(-1).squeeze(-1)  # [B, 512]
        # global_proj = self.feature_proj(global_feat).view(batch_size, 256, 8, 8)  # [B, 256, 8, 8]
        # global_proj_upsampled = F.interpolate(global_proj, size=(32, 32), mode='bilinear', align_corners=False) # [B, 256, 32, 32]

        # === Bottleneck Fusion ===
        bottleneck_input = torch.cat([x4, full_input], dim=1)  # [B, 512, 32, 32]
        x_bottleneck = self.bottleneck(bottleneck_input)  # [B, 256, 32, 32]

        # === Decoder ===
        x = self.up1(x_bottleneck, x3)  # [B, 128, 64, 64]
        x = self.up2(x, x2)  # [B, 64, 128, 128]
        x = self.up3(x, x1)  # [B, 32, 256, 256]

        output = self.last_conv(x)
        return self.activation(output)



