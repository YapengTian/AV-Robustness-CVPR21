import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseLoss(nn.Module):
    def __init__(self):
        super(BaseLoss, self).__init__()

    def forward(self, preds, targets):
        if isinstance(preds, list):
            N = len(preds)

            errs = [self._forward(preds[n], targets[n])
                    for n in range(N)]
            err = torch.mean(torch.stack(errs))

        elif isinstance(preds, torch.Tensor):
            err = self._forward(preds, targets)

        return err


class BCELoss(BaseLoss):
    def __init__(self):
        super(BCELoss, self).__init__()

    def _forward(self, pred, target, weight):
        return F.binary_cross_entropy(pred, target, weight=weight)

class CELoss(BaseLoss):
    def __init__(self):
        super(CELoss, self).__init__()
        self.crit = nn.CrossEntropyLoss()

    def _forward(self, pred, target):
        return self.crit(pred, target)
