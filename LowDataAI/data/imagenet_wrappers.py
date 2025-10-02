# data/imagenet_wrappers.py
from data.imagenet import (
    load_tiny_imagenet,
    load_tiny_imagenet_budget,
    load_imagenet100,
)

def tiny_full(batch_size=64, image_size=(64,64)):
    train_ds, val_ds, _ = load_tiny_imagenet(batch_size=batch_size, image_size=image_size)
    return train_ds, val_ds

def tiny_budget(pct, batch_size=64, image_size=(64,64), seed=42):
    if pct == 1:
        train_ds, val_ds, _ = load_tiny_imagenet(batch_size=batch_size, image_size=image_size)
        return train_ds, val_ds
    else:
        train_sub, val_ds = load_tiny_imagenet_budget(pct, batch_size=batch_size, image_size=image_size, seed=seed)
        return train_sub, val_ds

def imnet100_full(batch_size=64, image_size=(224,224)):
    train_ds, val_ds, _ = load_imagenet100(batch_size=batch_size, image_size=image_size)
    return train_ds, val_ds
