import sys
sys.path.append('/home/ccy/cellcycle')

import torch 
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models       #import some pretrained models like EfficientNet
from torchsummary import summary
import config

'''E-UNet's EfficientNet architecture would be used in encoder'''

'''My version'''
class Convolution_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Convolution_Block,self).__init__()
        self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size=3, padding=1)
        self.bn1= nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x

class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Encoder,self).__init__()
        self.convblock = Convolution_Block(in_channels, out_channels)
        self.maxpooling = nn.MaxPool2d(kernel_size=2)
        
    def forward(self, x):
        x = self.convblock(x)             # for skip connection
        p = self.maxpooling(x)            # for next layer 
        return x, p 
    
class EfficientNet_Encoder(nn.Module):
    def __init__(self):
        super(EfficientNet_Encoder,self).__init__()  
        # Load a pretrained EfficientNet
        efficientnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        # Remove the classification head
        self.efficientnet = nn.Sequential(*list(efficientnet.children())[:-2])
    
    def forward(self,x):
        skips = []  # store the intermediate layer's output for skip connection
        for layer in self.features:
            x = layer(x)
            skips.append(x)
        return skips
        
    
class Decoder(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(Decoder).__init__()
        self.up = nn.ConvTranspose2d(mid_channels,out_channels, kernel_size=2, stride=2)       # Key!!! stride=2: kernal move 2 pixel => image size*2 
        self.convblock = Convolution_Block(mid_channels+out_channels, out_channels)
        
    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)  # skip connection
        x = self.convblock(x)
        return x
        
class EUNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EUNet).__init__()
        self.encoder = EfficientNet_Encoder()
        
        # Decoder blocks
        self.dec1 = Decoder(1280, 512, 320)  # 1280是EfficientNet-B0最后一个卷积层的通道数
        self.dec2 = Decoder(320, 256, 192)   # 320是EfficientNet-B0 stage 8的通道数
        self.dec3 = Decoder(192, 128, 112)   # 192是EfficientNet-B0 stage 7的通道数
        self.dec4 = Decoder(112, 64, 80)     # 112是EfficientNet-B0 stage 6的通道数
        
        # Classifier
        # 最后的解码器输出通道数应与最终分类器的输入通道数匹配
        self.final = nn.Conv2d(80, out_channels, kernel_size=1)  
        
    def forward(self, x):
        skips = self.encoder(x)
        
        for i, skip in enumerate(skips):
            print(f"Shape of skip {i}: ", skip.shape)
        
        # 使用EfficientNet的输出作为skip connection
        x = self.dec1(skips[-1], skips[-4])  
        print("x.shape after layer1:", x.shape)
        
        if x.shape != skips[-5]:
            raise ValueError(f"Shape mismatch at {skips[-5].shape}")
        x = self.dec2(x, skips[-5])          # stage 8
        
        if x.shape != skips[-6]:
            raise ValueError(f"Shape mismatch at {skips[-6].shape}")
        x = self.dec3(x, skips[-6])          # stage 7
        
        if x.shape != skips[-7]:
            raise ValueError(f"Shape mismatch at {skips[-7].shape}")
        x = self.dec4(x, skips[-7])          # stage 6
        
        x = self.final(x)
        return x
    
'''Github version'''
class DecoderBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.double_conv(x)


class MBConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, expansion_rate=6, se=True):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.expansion_rate = expansion_rate
        self.se = se

        expansion_channels = in_channels * expansion_rate
        se_channels = max(1, int(in_channels * 0.25))

        if kernel_size == 3:
            padding = 1
        elif kernel_size == 5:
            padding = 2
        else:
            print("Error: unsupported kernel size")

        # Expansion
        if expansion_rate != 1:
            self.expand_conv = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=expansion_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(expansion_channels),
                nn.ReLU()
            )

        # Depthwise convolution
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(in_channels=expansion_channels, out_channels=expansion_channels, kernel_size=kernel_size,
                      stride=stride, padding=padding, groups=expansion_channels, bias=False),
            nn.BatchNorm2d(expansion_channels),
            nn.ReLU()
        )

        # Squeeze and excitation block
        if se:
            self.se_block = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Conv2d(in_channels=expansion_channels, out_channels=se_channels, kernel_size=1, bias=False),
                nn.ReLU(),
                nn.Conv2d(in_channels=se_channels, out_channels=expansion_channels, kernel_size=1, bias=False),
                nn.Sigmoid()
            )

        # Pointwise convolution
        self.pointwise_conv = nn.Sequential(
            nn.Conv2d(in_channels=expansion_channels, out_channels=out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, inputs):

        x = inputs

        if self.expansion_rate != 1:
            x = self.expand_conv(x)

        x = self.depthwise_conv(x)

        if self.se:
            x = self.se_block(x) * x

        x = self.pointwise_conv(x)

        if self.in_channels == self.out_channels and self.stride == 1:
            x = x + inputs

        return x


class EffUNet(nn.Module):
    """ U-Net with EfficientNet-B0 encoder """

    def __init__(self, in_channels, classes):
        super().__init__()

        self.start_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.down_block_2 = nn.Sequential(
            MBConvBlock(32, 16, kernel_size=3, stride=1, expansion_rate=1),
            MBConvBlock(16, 24, kernel_size=3, stride=2, expansion_rate=6),
            MBConvBlock(24, 24, kernel_size=3, stride=1, expansion_rate=6)
        )

        self.down_block_3 = nn.Sequential(
            MBConvBlock(24, 40, kernel_size=5, stride=2, expansion_rate=6),
            MBConvBlock(40, 40, kernel_size=5, stride=1, expansion_rate=6)
        )

        self.down_block_4 = nn.Sequential(
            MBConvBlock(40, 80, kernel_size=3, stride=2, expansion_rate=6),
            MBConvBlock(80, 80, kernel_size=3, stride=1, expansion_rate=6),
            MBConvBlock(80, 80, kernel_size=3, stride=1, expansion_rate=6),
            MBConvBlock(80, 112, kernel_size=5, stride=1, expansion_rate=6)
        )

        self.down_block_5 = nn.Sequential(
            MBConvBlock(112, 112, kernel_size=5, stride=1, expansion_rate=6),
            MBConvBlock(112, 112, kernel_size=5, stride=1, expansion_rate=6),
            MBConvBlock(112, 192, kernel_size=5, stride=2, expansion_rate=6),
            MBConvBlock(192, 192, kernel_size=5, stride=1, expansion_rate=6),
            MBConvBlock(192, 192, kernel_size=5, stride=1, expansion_rate=6),
            MBConvBlock(192, 192, kernel_size=5, stride=1, expansion_rate=6),
            MBConvBlock(192, 320, kernel_size=3, stride=1, expansion_rate=6)
        )

        self.up_block_4 = DecoderBlock(432, 256)

        self.up_block_3 = DecoderBlock(296, 128)

        self.up_block_2 = DecoderBlock(152, 64)

        self.up_block_1a = DecoderBlock(96, 32)

        self.up_block_1b = DecoderBlock(32, 16)

        self.head_conv = nn.Conv2d(in_channels=16, out_channels=classes, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        x1 = self.start_conv(x)

        x2 = self.down_block_2(x1)

        x3 = self.down_block_3(x2)

        x4 = self.down_block_4(x3)

        x5 = self.down_block_5(x4)
        x5 = F.interpolate(x5, scale_factor=2)
        x5 = torch.cat([x5, x4], dim=1)

        x5 = self.up_block_4(x5)
        x5 = F.interpolate(x5, scale_factor=2)
        x5 = torch.cat([x5, x3], dim=1)

        x5 = self.up_block_3(x5)
        x5 = F.interpolate(x5, scale_factor=2)
        x5 = torch.cat([x5, x2], dim=1)

        x5 = self.up_block_2(x5)
        x5 = F.interpolate(x5, scale_factor=2)
        x5 = torch.cat([x5, x1], dim=1)

        x5 = self.up_block_1a(x5)
        x5 = F.interpolate(x5, scale_factor=2)

        x5 = self.up_block_1b(x5)
        output = self.head_conv(x5)

        return output
'''
model = EffUNet(in_channels=3, classes=3).to(config.DEVICE)
summary(model, input_size=(3,1024,1024))
'''