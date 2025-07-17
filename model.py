import torch
import torch.nn as nn
import torch.nn.functional as F

#Encoder
class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_p=0.3):
        super(EncoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding="same")
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding="same")
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(dropout_p) # Dropout layer

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x) # Apply dropout
        skip = x
        x_pooled = self.pool(x)
        return skip, x_pooled


#Decoder
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, dropout_p=0.3):
        super(DecoderBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout_p) # Dropout layer

    def forward(self, x, skip_features):
        x = self.up(x)
        if skip_features.shape[2:] != x.shape[2:]:
            skip_features = F.interpolate(skip_features, size=x.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip_features], dim=1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x) # Apply dropout
        return x
    

# Unet Model
class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1, dropout_p=0.3):
        super(UNet, self).__init__()
        # Encoder
        self.enc1 = EncoderBlock(in_channels, 64, dropout_p)
        self.enc2 = EncoderBlock(64, 128, dropout_p)
        self.enc3 = EncoderBlock(128, 256, dropout_p)
        self.enc4 = EncoderBlock(256, 512, dropout_p)

        # Bottleneck
        self.bottleneck_conv1 = nn.Conv2d(512, 1024, kernel_size=3, padding="same")
        self.bottleneck_bn1 = nn.BatchNorm2d(1024)
        self.bottleneck_relu1 = nn.ReLU(inplace=True)
        self.bottleneck_conv2 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)
        self.bottleneck_bn2 = nn.BatchNorm2d(1024)
        self.bottleneck_relu2 = nn.ReLU(inplace=True)
        self.bottleneck_dropout = nn.Dropout(dropout_p)

        # Decoder
        self.dec1 = DecoderBlock(1024, 512, 512, dropout_p)
        self.dec2 = DecoderBlock(512, 256, 256, dropout_p)
        self.dec3 = DecoderBlock(256, 128, 128, dropout_p)
        self.dec4 = DecoderBlock(128, 64, 64, dropout_p)

        # Output layer
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        s1, x = self.enc1(x)
        s2, x = self.enc2(x)
        s3, x = self.enc3(x)
        s4, x = self.enc4(x)

        # Bottleneck
        x = self.bottleneck_relu1(self.bottleneck_bn1(self.bottleneck_conv1(x)))
        x = self.bottleneck_relu2(self.bottleneck_bn2(self.bottleneck_conv2(x)))
        x = self.bottleneck_dropout(x)

        # Decoder
        x = self.dec1(x, s4)
        x = self.dec2(x, s3)
        x = self.dec3(x, s2)
        x = self.dec4(x, s1)

        # Final output (logits)
        out = self.final_conv(x)
        return out


if __name__ == "__main__":
    model = UNet(in_channels=3, num_classes=1)
    x = torch.randn(1, 3, 256, 256)  # Batch of 1 RGB image, 256x256
    output = model(x)
    print(output.shape)  # Should be [1, 1, 256, 256]
