from utils.data_utils import load_dataset, create_subset, create_subset_fewer_frogs, \
    create_subset_more_critical_classes, create_augmented_dataset


def load_cifar10(batch_size=32):

    train_ds, test_ds = load_dataset('cifar10', batch_size=batch_size, split=['train', 'test'])
    return train_ds, test_ds


def load_cifar10_halved(batch_size=32):

    train_ds, test_ds = load_cifar10(batch_size=batch_size)
    decreased_train_ds = create_subset(train_ds, subset_fraction=0.5)
    return decreased_train_ds, test_ds


def load_cifar10_quartered(batch_size=32):

    train_ds, test_ds = load_cifar10(batch_size=batch_size)
    decreased_train_ds = create_subset(train_ds, subset_fraction=0.25)
    return decreased_train_ds, test_ds

def load_cifar10_quartered_fewer_frogs(batch_size=32):

    train_ds, test_ds = load_cifar10(batch_size=batch_size)
    decreased_train_ds = create_subset_fewer_frogs(train_ds, subset_fraction=0.25, frog_class=6, frog_fraction=0.5)
    return decreased_train_ds, test_ds

def load_cifar10_quartered_boost_critical_double(batch_size=32):

    train_ds, test_ds = load_cifar10(batch_size=batch_size)
    decreased_train_ds = create_subset_more_critical_classes(train_ds, subset_fraction=0.25, boost_classes=[2, 3, 5, 7], boost_factor=2)
    return decreased_train_ds, test_ds

def load_cifar10_eight(batch_size=32):

    train_ds, test_ds = load_cifar10(batch_size=batch_size)
    decreased_train_ds = create_subset(train_ds, subset_fraction=0.125)
    return decreased_train_ds, test_ds


def load_cifar10_quartered_augmented_double(batch_size=32):

    train_ds, test_ds = load_cifar10(batch_size=batch_size)
    decreased_train_ds = create_subset(train_ds, subset_fraction=0.25)
    boosted_dataset = create_augmented_dataset(decreased_train_ds, factor=2)
    return boosted_dataset, test_ds

def load_cifar10_quartered_fewer_frogs_augmented_double(batch_size=32):

    train_ds, test_ds = load_cifar10(batch_size=batch_size)
    decreased_train_ds = create_subset_fewer_frogs(train_ds, subset_fraction=0.25, frog_class=6, frog_fraction=0.5)
    boosted_dataset = create_augmented_dataset(decreased_train_ds, factor=2)
    return boosted_dataset, test_ds