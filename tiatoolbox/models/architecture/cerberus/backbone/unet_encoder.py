from torch import nn


class UnetDownModule(nn.Module):
    """U-Net downsampling block."""

    def __init__(self, in_channels, out_channels, downsample=True):
        super().__init__()

        # layers: optional downsampling, 2 x (conv + bn + relu)
        self.maxpool = nn.MaxPool2d((2, 2)) if downsample else None
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.maxpool is not None:
            x = self.maxpool(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        return x


class UnetEncoder(nn.Module):
    """U-Net encoder. https://arxiv.org/pdf/1505.04597.pdf"""

    def __init__(self, pretrained=False):
        super().__init__()
        if pretrained == True:
            print("WARNING: No pre-trained model available for U-Net encoder!")
        self.module1 = UnetDownModule(3, 64, downsample=False)
        self.module2 = UnetDownModule(64, 128)
        self.module3 = UnetDownModule(128, 256)
        self.module4 = UnetDownModule(256, 512)
        self.module5 = UnetDownModule(512, 1024)

    def forward(self, x):
        x1 = self.module1(x)
        x2 = self.module2(x1)
        x3 = self.module3(x2)
        x4 = self.module4(x3)
        x5 = self.module5(x4)

        feats = [x1, x2, x3, x4, x5]

        return feats
