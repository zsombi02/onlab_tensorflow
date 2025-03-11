from utils.data_utils import load_dataset

def load_cifar10(batch_size=32):
    """
    Loads CIFAR-10 dataset.

    Args:
        batch_size (int): Batch size for training.

    Returns:
        Tuple of (train_ds, test_ds)
    """
    train_ds, test_ds = load_dataset('cifar10', batch_size=batch_size, split=['train', 'test'])
    return train_ds, test_ds
