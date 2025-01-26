import torch.nn.functional as F

def cross_entropy_loss(logits, gold, trg_pad_idx):
    ''' Calculate cross entropy loss'''
    n_classes = logits.shape[-1]

    logits = logits.view(-1, n_classes)
    gold = gold.unsqueeze(-1).view(-1)
    loss = F.cross_entropy(logits, gold, ignore_index=trg_pad_idx)
    return loss