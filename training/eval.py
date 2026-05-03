import torch
import tqdm

import config as conf


@torch.no_grad()
def evaluate(model, loader, metrics: dict):
    """
    metrics: dict[str, callable]
        Each callable should take (logits, batch) and return a scalar (float or tensor)
    """
    model.eval()

    pbar = tqdm.tqdm(loader, dynamic_ncols=True, desc="Evaluating")

    totals = {name: 0.0 for name in metrics}
    counts = {name: 0 for name in metrics}

    for step, batch in enumerate(pbar):
        for k in batch:
            batch[k] = batch[k].to(conf.DEVICE, non_blocking=True)

        logits = model(batch)

        batch_size = batch["labels"].size(0)

        for name, fn in metrics.items():
            value = fn(logits, batch)

            totals[name] += value * batch_size
            counts[name] += batch_size

    model.train()

    return {
        name: totals[name] / max(counts[name], 1)
        for name in metrics
    }