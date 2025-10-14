# data/imagenet_wrappers.py
from data.imagenet import load_imagenet100_kaggle_budget, load_imagenet50_from_imagenet100


# def tiny_full(batch_size=64, image_size=(64,64)):
#     train_ds, test_ds, _ = load_tiny_imagenet(batch_size=batch_size, image_size=image_size)
#     return train_ds, test_ds
#
# def tiny_budget(pct, batch_size=64, image_size=(64,64), seed=42):
#     if pct == 1:
#         train_ds, test_ds, _ = load_tiny_imagenet(batch_size=batch_size, image_size=image_size)
#         return train_ds, test_ds
#     else:
#         train_sub, test_ds = load_tiny_imagenet_budget(pct, batch_size=batch_size, image_size=image_size, seed=seed)
#         return train_sub, test_ds
#
# def imnet100_full(batch_size=64, image_size=(224,224)):
#     train_ds, test_ds, _ = load_imagenet100(batch_size=batch_size, image_size=image_size)
#     return train_ds, test_ds

def imnet100_kaggle_budget(pct, batch_size=64, image_size=(224,224), seed=42):
    train_sub, val_ds = load_imagenet100_kaggle_budget(
        pct=pct, batch_size=batch_size, image_size=image_size, seed=seed
    )
    return train_sub, val_ds

def imnet50_from_100_budget(pct, batch_size=64, image_size=(224,224), seed=42):

    train_sub, val_ds = load_imagenet50_from_imagenet100(
        pct=pct, batch_size=batch_size, image_size=image_size, seed=seed
    )
    return train_sub, val_ds