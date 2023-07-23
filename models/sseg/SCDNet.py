import torch
from torch import nn

from models.block.attention import ChannelAttention, SPModel
from models.sseg.resnet import resnet18


class ConvBNRelu(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ConvBNRelu, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.conv1(x)
        identity = x
        x = self.bn1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.bn2(x)
        output = self.activation(x + identity)
        return output

class Aggregation(nn.Module):
    def __init__(self):
        super(Aggregation, self).__init__()
        self.conv2_3 = nn.Sequential(ConvBNRelu(1024, 512),ConvBNRelu(512, 128),ConvBNRelu(128, 128)
                                     )
        self.conv2_4 = ConvBNRelu(128, 64)
        self.conv2_5 = ConvBNRelu(64, 32)
        self.head = nn.Sequential(nn.Conv2d(32, 1, kernel_size=1), nn.Upsample(scale_factor=2, mode='bilinear'))
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
    def forward(self,x0_3, x1_3):

        d3 = self.conv2_3(torch.cat([x0_3, x1_3], 1))# 512,64,64
        d2 = self.conv2_4(self.up(d3))# 64,128,128
        d1 = self.conv2_5(self.up(d2))  # 32,256,256
        change = self.head(d1)# 1,512,512
        return change

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.conv1_3 = ConvBNRelu(512, 128)
        self.conv1_4 = ConvBNRelu(128, 32)
        self.conv1_5 = ConvBNRelu(32, 6)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
    def forward(self, x):
        # 512,16,16
        x = self.up(self.conv1_3(x))
        x = self.up(self.conv1_4(x))
        x = self.up(self.conv1_5(x))
        return x
class Fea_Inter_Model(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0):
        super(Fea_Inter_Model, self).__init__()
        # self.conv = nn.Conv2d(512, 256, kernel_size=1)
        self.gen_mask = nn.Sequential(nn.Conv2d(512, 1, kernel_size=3, padding=1),
                                      nn.BatchNorm2d(1))
        self.sigmoid = nn.Tanh()
        self.conv_final = ConvBNRelu(512, 512)
        self.alpha = torch.nn.Parameter(torch.FloatTensor([alpha]), requires_grad=True)
        self.beta = torch.nn.Parameter(torch.FloatTensor([beta]), requires_grad=True)
        self.conv1 = nn.Conv2d(1024, 256, kernel_size=1)
        self.conv2 = nn.Conv2d(512, 256, kernel_size=1)
        self.sp = SPModel(512)
        self.relu = nn.ReLU()

    def forward(self, x0_3, x1_3):
        # x0_3 = self.conv(x0_3)# 256,64,64
        # x1_3 = self.conv(x1_3)
        d1 = self.conv1(torch.cat([x0_3, x1_3], 1))
        d2 = self.conv2(torch.abs(x0_3-x1_3))
        # 512,64,64->1,64,64
        mask1 = self.sp(torch.cat([d1, d2], 1)) #512,64,64
        mask2 = self.gen_mask(mask1)# 1,64,64

        x0_3 = self.conv_final(x0_3 * self.sigmoid(mask2) * self.beta) + x0_3
        x1_3 = self.conv_final(x1_3 * self.sigmoid(mask2) * self.beta) + x1_3

        return x0_3, x1_3, mask2
class SCDNet(nn.Module):
    def __init__(self):
        super(SCDNet, self).__init__()
        self.resnet = resnet18(pretrained=True)
        self.aggregation = Aggregation()

        self.classifier = Classifier()
        self.neck = Fea_Inter_Model(alpha=0.1, beta=1.0)

    def forward_single(self, x):
        # resnet layers
        # 3,512,512
        x1 = self.resnet.conv1(x)# 64,256,256
        x = self.resnet.bn1(x1)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)# 64,128,128

        x2 = self.resnet.layer1(x)# 64,128,128
        x3_1 = self.resnet.layer2(x2)# 128,64,64
        x3_2 = self.resnet.layer3(x3_1)# 256,64，64
        x3_3 = self.resnet.layer4(x3_2)# 512,64,64
        return x3_3

    def forward(self, x1, x2):
        x0_3 = self.forward_single(x1)
        x1_3 = self.forward_single(x2)
        x0_3, x1_3, mask = self.neck(x0_3, x1_3)
        change = self.aggregation(x0_3, x1_3)
        s1 = self.classifier(x0_3)
        s2 = self.classifier(x1_3)
        change = torch.sigmoid(change)
        return  s1, s2, change.squeeze(1), mask


if __name__ == '__main__':
    x1 = torch.rand(8, 3, 512, 512)
    x2 = torch.rand(8, 3, 512, 512)
    net = SCDNet()
    s1, s2, c = net(x1, x2)
    print(c.shape)


