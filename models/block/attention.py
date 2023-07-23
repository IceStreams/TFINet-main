from torch import nn


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        out = x * self.sigmoid(out)
        return out
class SPModel(nn.Module):
    def __init__(self, ch):
        super(SPModel, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1, count_include_pad=False)
        self.fc1 = nn.Conv2d(in_channels=ch, out_channels=ch//16, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(ch//16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels=ch//16, out_channels=ch, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(ch)
    def forward(self, x):
        identity = x
        x = self.pool(x)
        x = self.bn1(self.fc1(x))
        x = self.fc2(self.relu(x))
        x = self.bn2(x)
        out = self.relu(x + identity)
        return out