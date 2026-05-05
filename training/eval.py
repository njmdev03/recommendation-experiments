import torch
import tqdm
import time

import config as conf


@torch.no_grad()
def evaluate(model, loader, metrics: dict = None, criterion=None):
    """
    metrics: dict[str, callable]
        Each callable should take (logits, batch) and return a scalar (float or tensor)
    """
    if not metrics and criterion is None:
        return

    model.eval()

    pbar = tqdm.tqdm(loader, dynamic_ncols=True, desc="Evaluating")

    totals = {name: 0.0 for name in metrics} if metrics else {}
    counts = {name: 0 for name in metrics} if metrics else {}

    total_loss = 0
    total_count = 0
    start_time = time.time()
    
    inference_times = []

    for step, batch in enumerate(pbar):
        for k in batch:
            batch[k] = batch[k].to(conf.DEVICE, non_blocking=True)

        t0 = time.time()
        logits = model(batch)
        t1 = time.time()
        inference_times.append(t1 - t0)

        if criterion:
            loss = criterion(logits, batch["labels"])
            total_loss += loss.item()
            total_count += 1

        batch_size = batch["labels"].size(0)

        if metrics:
            for name, fn in metrics.items():
                value = fn(logits, batch)

                totals[name] += value * batch_size
                counts[name] += batch_size

    eval_time = time.time() - start_time
    model.train()

    res = {
        name: totals[name] / max(counts[name], 1)
        for name in metrics
    } if metrics else {}

    if criterion:
        res["loss"] = total_loss / max(total_count, 1)

    res["eval_time"] = eval_time
    res["avg_batch_inference_time"] = sum(inference_times) / len(inference_times) if inference_times else 0

    return res