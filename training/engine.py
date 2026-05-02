import torch
import tqdm
import time

import config as conf


def train_epoch(loader, model, optimizer, scaler, criterion):
    model.train()

    pbar = tqdm.tqdm(loader, dynamic_ncols=True)

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

        loss = loss / conf.ACCUM_STEPS
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

        pbar.set_postfix({
            "gpu_sps": f"{eff_batch / fwd_time:.1f}",
            "pipe_sps": f"{eff_batch / total_time:.1f}",
            "enqueue": f"{transfer_time:.3f}",
            "fwd": f"{fwd_time:.3f}",
            "opt": f"{opt_time:.3f}",
        })