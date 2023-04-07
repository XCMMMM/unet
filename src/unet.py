from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

# 左边每一层的2个3x3卷积，不会改变特征层高宽，但改变channel大小
# 对应于UNet结构图中的蓝色箭头
class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv, self).__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

# 左边的 下采样+2个3x3卷积层
# 下采样通过卷积核大小为2x2的Maxpool来实现，对应于结构图中的红色箭头
# 2个3x3卷积层通过上面定义的DoubleConv类来实现，对应于结构图中的蓝色箭头
class Down(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__(
            nn.MaxPool2d(2, stride=2),
            DoubleConv(in_channels, out_channels)
        )

# 对应于右边的 上采样+conact拼接+2个3x3卷积层
# in_channels--对应于concat拼接之后的in_chanenels
# bilinear--对应于是否采用双线性插值替代转置卷积，默认会
class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            # scale_factor--上采样率设置成2
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            # 此时2个卷积层输入channel为concat拼接后的in_channels
            # mid_channels为in_channels的一半，即in_channels // 2
            # 输出channel为out_channels
            # 此时的最终输出out_channels和中间输出mid_channels并不相等
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            # 采用转置卷积的方式进行上采样
            # 由原始论文的结构图可以看出，经过上采样后，channel数会由1024->512，即减半
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            # 拼接后的channel又会由512->1024，即翻倍，所以输入channel是in_channels
            # 中间输出mid_channels和输出out_channels默认相等
            self.conv = DoubleConv(in_channels, out_channels)

    # x1--对应于要进行上采样的特征层
    # x2--对应于要进行concat拼接的特征层
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        # 先将x1进行上采样
        x1 = self.up(x1)
        # 左边进行了4次下采样，将特征层下采样为原来的16=2*2*2*2倍
        # 如果输入的图片不是16的整数倍，会面临向下取整的问题，这样concat拼接时可能两个拼接的特征层大小会不一样
        # [N, C, H, W]
        # 用 x2的高度-上采样后的x1的高度 得到高度方向上的差值
        diff_y = x2.size()[2] - x1.size()[2]
        # 用 x2的宽度-上采样后的x1的宽度 得到宽度方向上的差值
        diff_x = x2.size()[3] - x1.size()[3]
        # 对高度和宽度进行相应的padding
        # padding_left, padding_right, padding_top, padding_bottom
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])
        # 将上采样后的x1和x2进行拼接
        x = torch.cat([x2, x1], dim=1)
        # 经过两个3x3的卷积层
        x = self.conv(x)
        return x

# 最后一个1x1的卷积层，没有BN以及激活函数
# 输出channel数为类别数num_classes
class OutConv(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )


class UNet(nn.Module):
    def __init__(self,
                 # 输入图片的channel，彩色图片channel为3，灰度图channel为1
                 # 这里默认为1，但我们采用的是彩色图片，所以在创建UNet网络模型时传入的是3
                 in_channels: int = 1,
                 # 输出类别个数
                 num_classes: int = 2,
                 # 是否采用双线性插值替代转置卷积进行上采样
                 bilinear: bool = True,
                 # base_c--第一个卷积层的channel个数64
                 base_c: int = 64):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        # 第一个DoubleConv模块，对应于结构图中左边的第一层，经过2个3x3的卷积，输出channel数变为64
        self.in_conv = DoubleConv(in_channels, base_c)
        # 第一个down模块，即下采样+2个3x3卷积，共有4个down模块
        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)
        factor = 2 if bilinear else 1
        # 第四个down模块，注意这里
        # 如果上采样时采用双线性插值的话，这里的channel个数是没有发生变化的，factor=2
        # 因为双线性插值时不会改变channel大小，这样做是为了concat拼接时channel数一样
        # 如果采用的是转置卷积进行上采样，那么factor=1
        self.down4 = Down(base_c * 8, base_c * 16 // factor)
        # 定义up模块，即 上采样+concat拼接+2个3x3卷积层，一共有4个up模块
        # 同样，如果采用双线性插值的话，第一个up模块的输入channel为1024=64*16,
        # 输出channel为输入的1/4，以为会经过两个3x3卷积层，每经过一个channel数都会减半（factor2）
        self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up(base_c * 2, base_c, bilinear)
        # 最后经过一个1x1的卷积层，输出channel为类别数num_classes
        self.out_conv = OutConv(base_c, num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.out_conv(x)

        return {"out": logits}
