import torch


def iou_score(pred, mask, num_classes=5):
    pred = torch.argmax(pred, dim=1)

    ious = []

    for c in range(num_classes):
        pred_c = (pred == c)
        mask_c = (mask == c)

        inter = (pred_c & mask_c).sum().float()
        union = (pred_c | mask_c).sum().float()

        if union == 0:
            continue

        ious.append(inter / union)

    return torch.mean(torch.tensor(ious))


def dice_score(pred, mask, num_classes=5):
    pred = torch.argmax(pred, dim=1)

    dices = []

    for c in range(num_classes):
        pred_c = (pred == c)
        mask_c = (mask == c)

        inter = (pred_c & mask_c).sum().float()

        dice = (2 * inter) / (pred_c.sum() + mask_c.sum() + 1e-6)

        dices.append(dice)

    return torch.mean(torch.tensor(dices))