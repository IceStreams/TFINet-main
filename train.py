import time

from datasets.change_detection import ChangeDetection
from models.model_zoo import get_model, get_scheduler
# from utils.options import Options
# from utils.palette import color_map
# from utils.metric import IOUandSek
import sys

from utils.deep import Criterion_d

sys.path.append('../')
from utils.palette import *
from utils.options import *
from utils.metric import *

import numpy as np
import os
from PIL import Image
import shutil
import torch
from torch import nn
import torchcontrib
from torch.nn import CrossEntropyLoss, BCELoss, DataParallel
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

def _update_training_loss_curve(train_seg_loss, train_bn_loss, loss):
    # 语义分支loss保存
    if not os.path.exists(os.path.join("outdir", 'seg_loss.npy')):
        SEG_loss = np.array([], np.float32)
    else:
        SEG_loss = np.load(os.path.join("outdir", 'seg_loss.npy'))
    SEG_loss = np.append(SEG_loss, train_seg_loss)
    np.save(os.path.join("outdir", 'seg_loss.npy'), SEG_loss)
    # 变化分支loss保存
    if not os.path.exists(os.path.join("outdir", 'bn_loss.npy')):
        BN_loss = np.array([], np.float32)
    else:
        BN_loss = np.load(os.path.join("outdir", 'bn_loss.npy'))
    BN_loss = np.append(BN_loss, train_bn_loss)
    np.save(os.path.join("outdir", 'bn_loss.npy'), BN_loss)
    # 总分支loss保存
    if not os.path.exists(os.path.join("outdir", 'loss.npy')):
        S_loss = np.array([], np.float32)
    else:
        S_loss = np.load(os.path.join("outdir", 'loss.npy'))
    S_loss = np.append(S_loss, loss)
    np.save(os.path.join("outdir", 'loss.npy'), S_loss)
def _update_val_score_curve(score):
    if not os.path.exists(os.path.join("outdir", 'score.npy')):
        Score = np.array([], np.float32)
    else:
        Score = np.load(os.path.join("outdir", 'score.npy'))
    Score = np.append(Score, score)
    np.save(os.path.join("outdir", 'score.npy'), Score)
class Trainer:
    def __init__(self, args):
        self.args = args

        trainset = ChangeDetection(root=args.data_root, mode="train")
        valset = ChangeDetection(root=args.data_root, mode="val")
        self.trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                                      pin_memory=False, num_workers=4, drop_last=True)
        self.valloader = DataLoader(valset, batch_size=args.val_batch_size, shuffle=False,
                                    pin_memory=True, num_workers=4, drop_last=False)

        self.model = get_model(args.model, init_type='normal', init_gain=0.02)

        # self.model.load_state_dict(torch.load(args.load_from), strict=True)

        weight = torch.FloatTensor([2, 2, 3, 3, 2, 1]).cuda()
        self.criterion = CrossEntropyLoss(ignore_index=-1, weight=weight)
        self.criterion_bin = BCELoss(reduction='none')
        self.criterion_mask = Criterion_d()

        self.optimizer = Adam(self.model.parameters(),
                              lr=args.lr, weight_decay=args.weight_decay)
        self.exp_lr_scheduler_G = get_scheduler(self.optimizer, args)



        self.iters = 0
        self.total_iters = len(self.trainloader) * args.epochs
        self.previous_best = 0.0

    def training(self):
        tbar = tqdm(self.trainloader)
        self.model.train()
        total_loss = 0.0
        total_loss_sem = 0.0
        total_loss_bin = 0.0
        total_loss_mask =0.0

        for i, (img1, img2, mask1, mask2, mask_bin, id) in enumerate(tbar):
            img1, img2 = img1.cuda(), img2.cuda()
            mask1, mask2 = mask1.cuda(), mask2.cuda()
            mask_bin = mask_bin.cuda()

            out1, out2, out_bin, mask = self.model(img1, img2)

            loss1 = self.criterion(out1, mask1 - 1)
            loss2 = self.criterion(out2, mask2 - 1)
            loss_bin = self.criterion_bin(out_bin, mask_bin)
            loss_bin[mask_bin == 0] *= 2
            loss_bin = loss_bin.mean()
            loss_mask = self.criterion_mask(mask, mask_bin)

            loss = loss_bin * 2 + loss1 + loss2 + loss_mask * 2

            total_loss_sem += loss1.item() + loss2.item()
            total_loss_bin += loss_bin.item()
            total_loss_mask +=loss_mask.item()
            total_loss += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.iters += 1

            tbar.set_description("Loss: %.3f, Semantic Loss: %.3f, Binary Loss: %.3f, loss_mask: %.3f" %
                                 (total_loss / (i + 1), total_loss_sem / (i + 1), total_loss_bin / (i + 1), total_loss_mask / (i + 1)))
        self.exp_lr_scheduler_G.step()
        _update_training_loss_curve(total_loss_sem / (len(self.trainloader) + 1), total_loss_bin / (len(self.trainloader) + 1), total_loss / (len(self.trainloader) + 1))
    def validation(self):
        tbar = tqdm(self.valloader)
        self.model.eval()
        metric_bn = IOUandSek(num_classes=len(ChangeDetection.CLASSES))
        metric_mask = ScoreCalculation(num_classes=2)
        if self.args.save_mask:
            if not os.path.exists("outdir/masks/val/im1/"):  # 如果路径不存在则创建
                os.makedirs("outdir/masks/val/im1/")
            if not os.path.exists("outdir/masks/val/im2/"):  # 如果路径不存在则创建
                os.makedirs("outdir/masks/val/im2/")
            if not os.path.exists("outdir/masks/val/label/"):  # 如果路径不存在则创建
                os.makedirs("outdir/masks/val/label/")
            cmap = color_map()
            cmap1 = color_lab()
        with torch.no_grad():
            for img1, img2, mask1, mask2, mask_bin, id in tbar:
                img1, img2 = img1.cuda(), img2.cuda()

                out1, out2, out_bin, out_mask = self.model(img1, img2)
                out1 = torch.argmax(out1, dim=1).cpu().numpy() + 1
                out2 = torch.argmax(out2, dim=1).cpu().numpy() + 1
                out_bin = (out_bin > 0.5).cpu().numpy().astype(np.uint8)
                out1[out_bin == 1] = 0
                out2[out_bin == 1] = 0

                if self.args.save_mask:
                    for i in range(out1.shape[0]):
                        mask = Image.fromarray(out1[i].astype(np.uint8), mode="P")
                        mask.putpalette(cmap)
                        mask.save("outdir/masks/val/im1/" + id[i])

                        mask = Image.fromarray(out2[i].astype(np.uint8), mode="P")
                        mask.putpalette(cmap)
                        mask.save("outdir/masks/val/im2/" + id[i])

                        mask = Image.fromarray(out_bin[i].astype(np.uint8), mode="P")
                        mask.putpalette(cmap1)
                        mask.save("outdir/masks/val/label/" + id[i])


                metric_bn.add_batch(out1, mask1.numpy())
                metric_bn.add_batch(out2, mask2.numpy())

                score, miou, sek = metric_bn.evaluate()
                tbar.set_description("Score:%.2f, mIOU: %.2f, SeK: %.2f" %  (score*100, miou*100, sek*100))

        score *= 100.0
        if not os.path.exists("outdir/models/"):  # 如果路径不存在则创建
            os.makedirs("outdir/models/")
        if score >= self.previous_best:
            if self.previous_best != 0:
                model_path = "outdir/models/%s_%.2f.pth" % \
                             (self.args.model, self.previous_best)
                if os.path.exists(model_path):
                    os.remove(model_path)

            torch.save(self.model.state_dict(), "outdir/models/%s_%.2f.pth" %
                       (self.args.model, score))
            self.previous_best = score
        _update_val_score_curve(score)


if __name__ == "__main__":
    args = Options().parse()
    trainer = Trainer(args)
    begin_time = time.time()
    # if args.load_from:
    #     trainer.validation()

    for epoch in range(args.epochs):
        print("\n==> Epoches %i, learning rate = %.7f\t\t\t\t previous best = %.2f" %
              (epoch, trainer.optimizer.param_groups[0]["lr"], trainer.previous_best))
        trainer.training()
        trainer.validation()
        print('Total time: %.1fh' % (((time.time() - begin_time)/3600)))
