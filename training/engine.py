import torch
import tqdm
import time

import config as conf


def train_epoch(loader, model, optimizer, scaler, criterion, metrics={}):
    model.train()

    pbar = tqdm.tqdm(loader, dynamic_ncols=True)

    totals = {name: 0.0 for name in metrics}
    counts = {name: 0 for name in metrics}

    total_loss = 0
    count = 0

    epoch_time = 0

    for step, batch in enumerate(pbar):
        t_init = time.time()

        # Queue batch
        t_transfer = time.time()

        for k in batch:
            batch[k] = batch[k].to(conf.DEVICE, non_blocking=True)

        transfer_time = time.time() - t_transfer

        # Forward pass
        t_fwd = time.time()

        with torch.amp.autocast(str(conf.DEVICE) ,enabled=conf.USE_MIX_PRE):
            logits = model(batch)
            loss = criterion(logits, batch["labels"])

        if (step + 1) % conf.TRAINING_METRIC_EVERY_N == 0:
            batch_size = batch["labels"].size(0)

            for name, fn in metrics.items():
                value = fn(logits, batch)

                totals[name] += value * batch_size
                counts[name] += batch_size

        loss = loss / conf.ACCUM_STEPS

        total_loss += loss.item()
        count += 1

        scaler.scale(loss).backward()

        if conf.SYNC_PROFILES:
            torch.cuda.synchronize()
        fwd_time = time.time() - t_fwd

        # Optimizer step
        t_opt = time.time()

        if (step + 1) % conf.ACCUM_STEPS == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if conf.SYNC_PROFILES:
            torch.cuda.synchronize()
        opt_time = time.time() - t_opt

        # Final profiling metrics
        total_time = time.time() - t_init

        eff_batch = conf.BATCH_SIZE * conf.ACCUM_STEPS

        epoch_time += total_time

        pbar.set_postfix({
            "gpu_sps": f"{eff_batch / fwd_time:.1f}",
            "pipe_sps": f"{eff_batch / total_time:.1f}",
            "enqueue": f"{transfer_time:.3f}",
            "fwd": f"{fwd_time:.3f}",
            "opt": f"{opt_time:.3f}",
        })

    result = {
        name: totals[name] / max(counts[name], 1)
        for name in metrics
    }

    result["loss"] = total_loss / max(count, 1)
    result["epoch_time"] = epoch_time

    return result