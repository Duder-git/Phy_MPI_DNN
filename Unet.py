import torch
import torch.nn as nn
import torch.nn.functional as F


# conv 3X3,ReLU same
class double_conv2d_bn(nn.Module):
    # 输入通道、输出通道：3彩色通道 1 黑白通道
    def __init__(self, in_channels, out_channels, kernel_size=3, strides=1, padding=1): # 通过padding选择实现
        super(double_conv2d_bn, self).__init__()    # 初始化父类

        # 第一次卷积
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=kernel_size,
                               stride=strides, padding=padding, bias=True)  # ’bias‘ 偏置项常数，开启
        # 第二次卷积
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=kernel_size,
                               stride=strides, padding=padding, bias=True)
        # 批标准化层
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    # 自动调用forward函数
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))   # 卷积-池化-激活
        out = F.relu(self.bn2(self.conv2(out))) # 卷积-池化-激活
        return out


# deconv 2X2,ReLU
class deconv2d_bn(nn.Module):
    # 当strides=1时，上采样图像与原图大小相同
    # 当strides=2时，上采样图像是原图的两倍大小
    def __init__(self, in_channels, out_channels, kernel_size=2, strides=2):
        super(deconv2d_bn, self).__init__()

        # 反卷积层
        self.conv1 = nn.ConvTranspose2d(in_channels, out_channels,
                                        kernel_size=kernel_size,
                                        stride=strides, bias=True)
        # 批标准化层
        self.bn1 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))   # 反卷积-池化-激活
        return out


class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        # 卷积
        # 特征提取用到
        self.layer1_conv = double_conv2d_bn(1, 8)       # 输入1通道，输出8通道
        self.layer2_conv = double_conv2d_bn(8, 16)      # 输入8通道，输出16通道
        self.layer3_conv = double_conv2d_bn(16, 32)     # 输入16通道，输出32通道
        self.layer4_conv = double_conv2d_bn(32, 64)
        self.layer5_conv = double_conv2d_bn(64, 128)

        # 解卷积 深度减小 图像增大
        self.deconv1 = deconv2d_bn(128, 64)
        self.deconv2 = deconv2d_bn(64, 32)
        self.deconv3 = deconv2d_bn(32, 16)
        self.deconv4 = deconv2d_bn(16, 8)

        # 上采样部分用到
        self.layer6_conv = double_conv2d_bn(128, 64)    # 输入128通道，输出64通道
        self.layer7_conv = double_conv2d_bn(64, 32)
        self.layer8_conv = double_conv2d_bn(32, 16)
        self.layer9_conv = double_conv2d_bn(16, 8)      # 输入16通道，输出8通道

        # 卷积层，最后输出图像
        self.layer10_conv = nn.Conv2d(8, 1, kernel_size=3, stride=1, padding=1, bias=True)
        # 激活函数 [0,1]
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 特征提取
        conv1 = self.layer1_conv(x)         # 输入
        pool1 = F.max_pool2d(conv1, 2)      # 最大池化函数 减采样 取最大值

        conv2 = self.layer2_conv(pool1)     # 第二层
        pool2 = F.max_pool2d(conv2, 2)

        conv3 = self.layer3_conv(pool2)
        pool3 = F.max_pool2d(conv3, 2)

        conv4 = self.layer4_conv(pool3)
        pool4 = F.max_pool2d(conv4, 2)

        conv5 = self.layer5_conv(pool4)     # 特征提取最后一层，无需最大池化

        # 上采样 恢复图像
        convt1 = self.deconv1(conv5)
        concat1 = torch.cat([convt1, conv4], dim=1)     # 拼接
        conv6 = self.layer6_conv(concat1)   # 卷积

        convt2 = self.deconv2(conv6)
        concat2 = torch.cat([convt2, conv3], dim=1)
        conv7 = self.layer7_conv(concat2)

        convt3 = self.deconv3(conv7)
        concat3 = torch.cat([convt3, conv2], dim=1)
        conv8 = self.layer8_conv(concat3)

        convt4 = self.deconv4(conv8)
        concat4 = torch.cat([convt4, conv1], dim=1)
        conv9 = self.layer9_conv(concat4)

        # 输出 最后卷积 激活
        outp = self.layer10_conv(conv9)
        outp = self.sigmoid(outp)
        return outp

'''
def main():
    model = Unet()
    inp = torch.rand(10, 1, 224, 224)
    outp = model(inp)
    print(outp.shape)
    # == > torch.Size([10, 1, 224, 224])


# 运行函数
main()
'''