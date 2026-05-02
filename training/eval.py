import torch

import config as conf


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()

    total_loss = 0
    count = 0

    for batch in loader:
        for k in batch:
            batch[k] = batch[k].to(conf.DEVICE, non_blocking=True)

        logits = model(batch)
        loss = criterion(logits, batch["labels"])

        total_loss += loss.item()
        count += 1

    model.train()

    return {"loss": total_loss / max(count, 1)}