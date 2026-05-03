import torch

import config as conf


@torch.no_grad()
def evaluate(model, loader, metrics: dict):
    """
    metrics: dict[str, callable]
        Each callable should take (logits, batch) and return a scalar (float or tensor)
    """
    model.eval()

    totals = {name: 0.0 for name in metrics}
    counts = {name: 0 for name in metrics}

    for batch in loader:
        for k in batch:
            batch[k] = batch[k].to(conf.DEVICE, non_blocking=True)

        logits = model(batch)

        batch_size = batch["labels"].size(0)

        for name, fn in metrics.items():
            value = fn(logits, batch["labels"])

            # convert tensors -> python float safely
            # if torch.is_tensor(value):
            #     value = value.item()

            totals[name] += value * batch_size
            counts[name] += batch_size

    model.train()

    return {
        name: totals[name] / max(counts[name], 1)
        for name in metrics
    }

# @torch.no_grad()
# def evaluate(model, loader, metrics: dict):
#     model.eval()

#     total_loss = 0
#     count = 0

#     tracked_met = {}

#     for batch in loader:
#         for k in batch:
#             batch[k] = batch[k].to(conf.DEVICE, non_blocking=True)



#         logits = model(batch)
#         loss = criterion(logits, batch["labels"])

#         total_loss += loss.item()
#         count += 1

#     model.train()

#     return {criterion_name: total_loss / max(count, 1)}