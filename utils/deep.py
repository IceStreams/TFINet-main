from torch import nn
from torch.nn import BCELoss


class Criterion_d(nn.Module):
    def __init__(self):
        super(Criterion_d, self).__init__()
        self.criterion_bin = BCELoss(reduction='mean')
        self.up_d = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.sigmoid = nn.Sigmoid()
    def forward(self, d, label):
        loss = self.criterion_bin(self.sigmoid(self.up_d(d)).squeeze(1), label.cuda())
        return loss
