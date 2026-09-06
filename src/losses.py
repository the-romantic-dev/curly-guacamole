import torch
import torch.nn.functional as F


def soft_dice_loss(logits, targets, smooth=1.0, valid_mask=None):
    probs = torch.sigmoid(logits).flatten(1)
    targets = targets.flatten(1)
    if valid_mask is not None:
        valid = valid_mask.flatten(1)
        probs = probs * valid
        targets = targets * valid
    inter = (probs * targets).sum(1)
    return (1.0 - (2.0 * inter + smooth) / (probs.sum(1) + targets.sum(1) + smooth)).mean()


def bce_loss(logits, targets, valid_mask=None):
    if valid_mask is not None:
        loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        weighted = (loss * valid_mask).flatten(1).sum(1)
        return (weighted / valid_mask.flatten(1).sum(1).clamp_min(1)).mean()
    return F.binary_cross_entropy_with_logits(logits, targets)

def compute_loss(out, batch, aux_weight: float = 0.0):
    """BCE + Dice по маске, BCE по гейту, плюс глубокая супервизия на страйде 4.

    Без последнего слагаемого голова aux4 не получает градиента вообще — она
    считается, но не учится, и вычисления уходят впустую.
    """
    valid = batch.get("valid_mask")
    bce = bce_loss(out["logits"], batch["mask"], valid_mask=valid)
    dice = soft_dice_loss(out["logits"], batch["mask"], valid_mask=valid)
    cls = bce_loss(out["cls_logits"], batch["label"])
    total = bce + dice + 0.3 * cls

    if aux_weight > 0 and "aux_logits" in out:
        total = total + aux_weight * (
            bce_loss(out["aux_logits"], batch["mask"], valid_mask=valid)
            + soft_dice_loss(out["aux_logits"], batch["mask"], valid_mask=valid)
        )
    return total
