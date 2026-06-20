import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, preds, targets):
        preds = F.softmax(preds, dim=1)

        targets_one_hot = F.one_hot(targets.clamp(min=0), self.num_classes)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()

        mask = (targets != -1).unsqueeze(1).float()
        targets_one_hot = targets_one_hot * mask

        smooth = 1e-6

        intersection = (preds * targets_one_hot).sum(dim=(2,3))
        union = preds.sum(dim=(2,3)) + targets_one_hot.sum(dim=(2,3))

        dice = (2 * intersection + smooth) / (union + smooth)

        return 1 - dice.mean()


class CombinedLoss(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=-1)
        self.dice = DiceLoss(num_classes)

    def forward(self, preds, targets):
        return 0.5 * self.ce(preds, targets) + 0.5 * self.dice(preds, targets)


class PFCE_Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(reduction='none', ignore_index=-1)

    def forward(self, preds, targets):

        loss = self.ce(preds, targets)

        
        mask = (targets != -1).float()

        loss = loss * mask

        return loss.sum() / (mask.sum() + 1e-6)