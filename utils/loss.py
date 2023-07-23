import torch
import torch.nn.functional as F
# from torch.autograd import Variable
import torch.nn as nn
class ChangeSimilarity(nn.Module):
    """input: x1, x2 multi-class predictions, c = class_num
       label_change: changed part
    """

    def __init__(self, reduction='mean'):
        super(ChangeSimilarity, self).__init__()
        self.loss_f = nn.CosineEmbeddingLoss(margin=0., reduction=reduction)

    def forward(self, x1, x2, label_change):
        b, c, h, w = x1.size()  # b,6,h,w
        x1 = F.softmax(x1, dim=1)
        x2 = F.softmax(x2, dim=1)
        x1 = x1.permute(0, 2, 3, 1)  # b,h,w,6
        x2 = x2.permute(0, 2, 3, 1)
        x1 = torch.reshape(x1, [b * h * w, c])  # bhw,c
        x2 = torch.reshape(x2, [b * h * w, c])

        label_unchange = ~label_change.bool()
        target = label_unchange.float()  # 0 1   1  0
        target = target - label_change.float()  # 1, -1
        target = torch.reshape(target, [b * h * w])  # b,h,w

        loss = self.loss_f(x1, x2, target)
        return loss