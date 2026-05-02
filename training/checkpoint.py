import torch

import config as conf


def save(path, model, optimizer, epoch, step, scaler):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "step": step,
        "scaler": scaler.state_dict(),
    }, path)

def load(path, model, optimizer, scaler):
    ckpt = torch.load(path, map_location=conf.DEVICE)

    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scaler.load_state_dict(ckpt["scaler"])

    return ckpt["epoch"], ckpt["step"]