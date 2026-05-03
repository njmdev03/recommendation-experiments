import torch

import config as conf


def save(path, model, optimizer, scaler, epoch):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch
    }, path)

def load(path, model, optimizer, scaler):
    ckpt = torch.load(path, map_location=conf.DEVICE)

    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scaler.load_state_dict(ckpt["scaler"])

    return ckpt["epoch"]