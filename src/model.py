import torch
from torch import nn


class FeatureBlock(nn.Module):
    def __init__(self, cin, cout, dimgroup):
        super(FeatureBlock, self).__init__()
        print('#FeatureBlock:', cin, cout)

        layers = [nn.GroupNorm(cin//dimgroup,cin),nn.GELU(),
                  nn.Conv3d(cin, cout, kernel_size=1, stride=2)]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class ChannelAttention(nn.Module):
    def __init__(self, cin):
        super(ChannelAttention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.fc = nn.Sequential(nn.Conv3d(cin, cin // 16, kernel_size=1, bias=False),
                                nn.GELU(),
                                nn.Conv3d(cin // 16, cin, kernel_size=1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv3d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class ConvBlock(nn.Module):
    def __init__(self, cin, cout, dimgroup):
        super(ConvBlock, self).__init__()

        chid = cout // 4
        print('#ConvBlock:', cin, chid, cout)

        self.conv = nn.Sequential(
            nn.GroupNorm(cin//dimgroup,cin),
            nn.GELU(),
            nn.Conv3d(cin, chid, kernel_size=1)
        )
        self.expand1 = nn.Sequential(
            nn.GroupNorm(chid//dimgroup,chid),
            nn.GELU(),
            nn.Conv3d(chid, cout, kernel_size=1)
        )
        self.expand2 = nn.Sequential(
            nn.GroupNorm(chid//dimgroup,chid),
            nn.GELU(),
            nn.Conv3d(chid, cout, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x = self.conv(x)
        expand1 = self.expand1(x)
        expand2 = self.expand2(x)
        merge = torch.cat((expand1, expand2), 1)
        return merge


class BasicBlock(nn.Module):
    def __init__(self, cin, cout, dimgroup):
        super(BasicBlock, self).__init__()

        self.conv = ConvBlock(cin, cout, dimgroup)
        self.sa = SpatialAttention()
        # self.ca = ChannelAttention(cout * 2)
        self.conv1x1 = Conv1x1(cin,cout*2, dimgroup)

    def forward(self, x):
        conv1x1 = self.conv1x1(x)
        out = self.conv(x)
        out = self.sa(out) * out
        # out = self.ca(out) * out
        out += conv1x1
        return out


class ResBasicBlock(nn.Module):
    def __init__(self, cin, cout, dimgroup):
        super(ResBasicBlock, self).__init__()

        self.conv = ConvBlock(cin, cout, dimgroup)
        self.sa = SpatialAttention()
        # self.ca = ChannelAttention(cout * 2)

    def forward(self, x):
        residual = x
        out = self.conv(x)
        out = self.sa(out) * out
        # out = self.ca(out) * out
        out += residual
        return out


class Conv1x1(nn.Module):
    def __init__(self, cin, cout, dimgroup):
        super(Conv1x1, self).__init__()

        layers = [nn.GroupNorm(cin//dimgroup,cin), nn.GELU(),
                  nn.Conv3d(cin, cout, kernel_size=1)]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class DenseBlock(nn.Module):
    def __init__(self, cin, cout):
        super(DenseBlock, self).__init__()
        print('#DenseBlock:', cin*2*2*2, cin, cout)

        layers = [nn.LayerNorm(cin*2*2*2), nn.GELU(),
                  nn.Linear(cin*2*2*2, cin),
                  nn.LayerNorm(cin), nn.GELU(),
                  nn.Linear(cin, cout)]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class ConvAttention3D(nn.Module):
    def __init__(self, args, num_class=1):
        super(ConvAttention3D, self).__init__()

        cin = args.feature_num
        cout = args.cout_dim
        chid = cout // 8
        dimgroup = args.dimgroup

        self.conv1 = nn.Sequential(FeatureBlock(cin, cin*6, dimgroup))
        self.conv2 = nn.Sequential(BasicBlock(cin*6, chid, dimgroup))
        self.res2 = nn.Sequential(ResBasicBlock(chid*2, chid, dimgroup))
        self.conv3 = nn.Sequential(BasicBlock(chid*2, chid*2, dimgroup))
        self.res3 = nn.Sequential(ResBasicBlock(chid*4, chid*2, dimgroup))

        self.pool = nn.MaxPool3d(kernel_size=3, stride=2)

        self.conv4 = nn.Sequential(BasicBlock(chid*4, chid*3, dimgroup))
        self.res4 = nn.Sequential(ResBasicBlock(chid*6, chid*3, dimgroup))
        self.conv5 = nn.Sequential(BasicBlock(chid*6, chid*4, dimgroup))
        self.res5 = nn.Sequential(BasicBlock(chid*8, chid*4, dimgroup))

        self.avg_pool = nn.AvgPool3d(kernel_size=3, padding=1)

        self.flatten = nn.Flatten()
        self.dense = DenseBlock(cout, num_class)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.res2(self.conv2(conv1))
        conv3 = self.res3(self.conv3(conv2))
        pool = self.pool(conv3)
        conv4 = self.res4(self.conv4(pool))
        conv5 = self.res5(self.conv5(conv4))
        avg_pool = self.avg_pool(conv5)
        flatten = self.flatten(avg_pool)
        dense = self.dense(flatten)
        return dense
