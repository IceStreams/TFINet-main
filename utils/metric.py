import math
import numpy as np


def cal_kappa(hist):
    if hist.sum() == 0:
        po = 0
        pe = 1
        kappa = 0
    else:
        po = np.diag(hist).sum() / hist.sum()
        pe = np.matmul(hist.sum(1), hist.sum(0).T) / hist.sum() ** 2
        if pe == 1:
            kappa = 0
        else:
            kappa = (po - pe) / (1 - pe)
    return kappa


class IOUandSek:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))

    def _fast_hist(self, label_pred, label_true):
        mask = (label_true >= 0) & (label_true < self.num_classes)
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) +
            label_pred[mask], minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        return hist

    def add_batch(self, predictions, gts):
        for lp, lt in zip(predictions, gts):
            self.hist += self._fast_hist(lp.flatten(), lt.flatten())

    def evaluate(self):
        confusion_matrix = np.zeros((2, 2))
        confusion_matrix[0][0] = self.hist[0][0]
        confusion_matrix[0][1] = self.hist.sum(1)[0] - self.hist[0][0]
        confusion_matrix[1][0] = self.hist.sum(0)[0] - self.hist[0][0]
        confusion_matrix[1][1] = self.hist[1:, 1:].sum()
        # confusion_matrix[1][1] = self.hist.sum()

        hist_n0 = self.hist.copy()
        hist_n0[0][0] = 0
        kappa_n0 = cal_kappa(hist_n0)
        iu = np.diag(confusion_matrix) / (confusion_matrix.sum(1) + confusion_matrix.sum(0) - np.diag(confusion_matrix))
        IoU_fg = iu[1]
        IoU_mean = (iu[0] + iu[1]) / 2
        Sek = (kappa_n0 * math.exp(IoU_fg)) / math.e
        Score = 0.3 * IoU_mean + 0.7 * Sek

        return Score, IoU_mean, Sek

    def miou(self):
        confusion_matrix = self.hist[1:, 1:]
        iou = np.diag(confusion_matrix) / (confusion_matrix.sum(0) + confusion_matrix.sum(1) - np.diag(confusion_matrix))
        return iou, np.mean(iou)
class ScoreCalculation:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))

    def _fast_hist(self, label_pred, label_true):
        mask = (label_true >= 0) & (label_true < self.num_classes)
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) +
            label_pred[mask], minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        return hist

    def add_batch(self, predictions, gts):
        for lp, lt in zip(predictions, gts):
            self.hist += self._fast_hist(lp.flatten(), lt.flatten())

    def evaluate(self):
        confusion_matrix = self.hist.copy()
        tn = confusion_matrix[1][1]
        fp = confusion_matrix[1][0]
        fn = confusion_matrix[0][1]
        tp = confusion_matrix[0][0]
        Precision = tp/(tp+fp)
        Recall = tp/(tp+fn)
        F1 = (2*Precision*Recall)/(Precision+Recall)
        miou = (tp/(tp+fp+fn) + tn/(tn+fp+fn))/2
        OA = (tp+tn)/(tp+tn+fp+fn)
        FPR = fp/(fp+tn)
        FNR = fn/(tp+fn)




        return F1, OA, FPR, FNR
