import torch
import torch.nn as nn

D53_conf = [
    (3, 32, 3, 1, 1),
    (32, 64, 3, 2, 1),
    [64],
    (64, 128, 3, 2, 1),
    [128],
    [128],
    (128, 256, 3, 2, 1),
    [256],
    [256],
    [256],
    [256],
    [256],
    [256],
    [256],
    [256],
    (256, 512, 3, 2, 1),
    [512],
    [512],
    [512],
    [512],
    [512],
    [512],
    [512],
    [512],
    (512, 1024, 3, 2, 1),
    [1024],
    [1024],
    [1024],
    [1024],
    (1024, 1024, 1, , 0),
]

class ConvolutionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvolutionBlock, self).__init__()
        self.conv = n.conv2d(in_channels, out_channels, kernel_size,stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.l_relu = nn.leaky_relu(0.1)

    def forward(self, x):
        return self.l_relu(self.bn(self.conv(x)))

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            ConvolutionBlock(channels, channels//2, kernel_size=1, stride=1, padding=0),
            ConvolutionBlock(channels//2, channels, kernel_size=3, stride=1, padding=1),
            )
        )

    def forward(self, x):
        return x + self.block(x)


class DarkNet53(nn.Module):
    def __init__(self):
        super(DarkNet53,self).__init__()
        self.layers = nn.ModuleList()
        
        for layer in D53_conf:
            if isinstance(layer, tuple):
                in_channels, out_channels, kernel_size, stride, padding = layer
                layers.append(ConvolutionBlock(in_channels, out_channels, kernel_size, stride, padding))

            if isinstance(layer, list):
                channels = layer[0]
                layers.append(ResidualBlock(channels))


    def foward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class YOLOv2_5(nn.Module):
    def __init__(self, num_classes=20, num_anchors=3):
        super(DetectionBlock, self).__init__()
        self.d53 = DarkNet53()
        self.num_cls = num_classes
        self.num_anchs = num_anchors
        self.final_conv = nn.Sequential(
            ConvolutionBlock(1024, 512, 1, 1, 0),
            ConvolutionBlock(512, 1024, 3, 1, 1),
            ConvolutionBlock(1024, 512, 1, 1, 0),
            ConvolutionBlock(512, 1024, 3, 1, 1),
            ConvolutionBlock(1024, 512, 1, 1, 0),
        )

        self.head = nn.Conv2d(512, self.num_anchs * (5 + self.num_cls), kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.d53(x)
        x = self.final_conv(x)
        x = self.head(x)

        batch, _, grid, _ = x.shape
        x = x.view(batch,self.num_anchs, 5 + num_cls, grid, grid)
        x = x.permute(0, 3, 4, 1, 2).contiguous()

        x[..., 0:2] = torch.sigmoid(x[..., 0:2])
        x[..., 4]  = torch.sigmoid(x[..., 4])
        x[..., 5:] = torch.sigmoid(x[..., 5:])

        return x
        
