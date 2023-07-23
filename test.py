from datasets.change_detection import ChangeDetection
from models.model_zoo import get_model
from utils.options import Options
from utils.palette import color_map,color_lab

import numpy as np
import cv2
import os
from PIL import Image
import shutil
import time
import torch
from torch import nn
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.metric import *

if __name__ == "__main__":
    """
    Since the final evaluation is limited in 400 seconds in this challenge and the online inference speed 
    is hard to estimate accurately, we compute the inference speed in earlier iterations during inference 
    and choose not to use test-time augmentation in later iterations if time is not enough.
    """

    START_TIME = time.time()
    LIMIT_TIME = 400 - 20
    PAST_TIME = 0
    NO_TTA_TIME = 0
    TTA_TIME = 0

    args = Options().parse()

    torch.backends.cudnn.benchmark = True

    print(torch.cuda.is_available())
    testset = ChangeDetection(root=args.data_root, mode="val")
    testloader = DataLoader(testset, batch_size=1, shuffle=False,
                            pin_memory=True, num_workers=4, drop_last=False)

    model2 = get_model('SCDNet', init_type='normal', init_gain=0.02)
    # 这里需要进行更改
    model2.load_state_dict(torch.load('outdir/models/SCDNet_36.50.pth'), strict=True)


    models = model2.cuda()
    models.eval()
    metric = IOUandSek(num_classes=len(ChangeDetection.CLASSES))
    metric_bn = ScoreCalculation(num_classes=2)
    metric_mask = ScoreCalculation(num_classes=2)
    cmap = color_map()
    cmap1 = color_lab()

    tbar = tqdm(testloader)
    TOTAL_ITER = len(testloader)
    CHECK_ITER = TOTAL_ITER // 5
    NO_TTA_ITER = TOTAL_ITER

    with torch.no_grad():
        for img1, img2, mask1, mask2, mask_bin, id in tbar:

            img1, img2 = img1.cuda(non_blocking=True), img2.cuda(non_blocking=True)

            out1_list, out2_list, out_bin_list = [], [], []

            out1, out2, out_bin, out_mask = models(img1, img2)

            m = nn.Upsample(scale_factor=8, mode='bilinear')
            out_mask = m(out_mask)
            
            out_mask = out_mask.squeeze(1)
            
            mask_d = out_mask
            
            out1 = torch.argmax(out1, dim=1) + 1
            out2 = torch.argmax(out2, dim=1) + 1
            out_bin = (out_bin > 0.5)
            out1[out_bin == 1] = 0
            out2[out_bin == 1] = 0
            out1 = out1.cpu().numpy()
            out2 = out2.cpu().numpy()
            out_bin=out_bin.cpu().numpy()
            out_mask = (out_mask > 0.5)
            out_mask=out_mask.cpu().numpy()

            metric.add_batch(out1, mask1.numpy())
            metric.add_batch(out2, mask2.numpy())

            score, miou, sek = metric.evaluate()
        
            tbar.set_description(" Score:%.2f, mIOU: %.2f, SeK: %.2f" %   (score*100, miou*100, sek*100))
            s = nn.Sigmoid()
            mask_d = s(mask_d)
            mask_d = mask_d.squeeze(0)
            mask_d = mask_d.cpu().numpy()
            mask_d = mask_d * 255
            heatmap = np.uint8(mask_d)  # 将热力图转换为RGB格式
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            
            for i in range(out1.shape[0]):
                mask = Image.fromarray(out1[i].astype(np.uint8), mode="L")
                mask.putpalette(cmap)
                mask.save("outdir/masks/test/out1/" + id[i])

                mask = Image.fromarray(out2[i].astype(np.uint8), mode="L")
                mask.putpalette(cmap)
                mask.save("outdir/masks/test/out2/" + id[i])

                mask = Image.fromarray(out_bin[i].astype(np.uint8), mode="L")
                mask.putpalette(cmap1)
                mask.save("outdir/masks/test/label/" + id[i])

                mask = Image.fromarray(out_mask[i].astype(np.uint8), mode="P")
                mask.putpalette(cmap1)
                mask.save("outdir/masks/test/deep/" + id[i])

                
                mask_map = Image.fromarray(heatmap.astype(np.uint8))
                mask_map.save("outdir/masks/test/heatmap/" + id[i])


    END_TIME = time.time()
    print("Inference Time: %.1fs" % (END_TIME - START_TIME))
