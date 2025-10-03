# data/imagenet_wrappers.py
from data.imagenet import (
    load_tiny_imagenet,
    load_tiny_imagenet_budget,
    load_imagenet100,
    load_imagenet100_kaggle, load_imagenet100_kaggle_budget
)

def tiny_full(batch_size=64, image_size=(64,64)):
    train_ds, test_ds, _ = load_tiny_imagenet(batch_size=batch_size, image_size=image_size)
    return train_ds, test_ds

def tiny_budget(pct, batch_size=64, image_size=(64,64), seed=42):
    if pct == 1:
        train_ds, test_ds, _ = load_tiny_imagenet(batch_size=batch_size, image_size=image_size)
        return train_ds, test_ds
    else:
        train_sub, test_ds = load_tiny_imagenet_budget(pct, batch_size=batch_size, image_size=image_size, seed=seed)
        return train_sub, test_ds

def imnet100_full(batch_size=64, image_size=(224,224)):
    train_ds, test_ds, _ = load_imagenet100(batch_size=batch_size, image_size=image_size)
    return train_ds, test_ds

def imnet100_kaggle_full(batch_size=64, image_size=(224,224)):
    train_ds, test_ds, _ = load_imagenet100_kaggle(batch_size=batch_size, image_size=image_size)
    return train_ds, test_ds

def imnet100_kaggle_budget(pct, batch_size=64, image_size=(224,224), seed=42):  # <-- ÚJ
    train_sub, test_ds = load_imagenet100_kaggle_budget(
        pct=pct, batch_size=batch_size, image_size=image_size, seed=seed
    )
    return train_sub, test_ds