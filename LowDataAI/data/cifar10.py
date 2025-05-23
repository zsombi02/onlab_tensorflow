from utils.data_utils import load_dataset, create_subset, create_subset_fewer_frogs, \
    create_subset_more_critical_classes, create_augmented_dataset, create_animal_subset, create_nonanimal_subset, \
    extract_numpy_data, generate_siamese_pairs, generate_siamese_pairs_dataset, save_dataset_snapshot


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

def load_cifar10_quartered_animals(batch_size=32):

    train_ds, test_ds = load_cifar10(batch_size=batch_size)
    animal_train_ds = create_animal_subset(train_ds, subset_fraction=0.25)
    animal_test_ds = create_animal_subset(test_ds, subset_fraction=1)

    return animal_train_ds, animal_test_ds

def load_cifar10_halved_animals(batch_size=32):

    train_ds, test_ds = load_cifar10(batch_size=batch_size)
    animal_train_ds = create_animal_subset(train_ds, subset_fraction=0.5)
    animal_test_ds = create_animal_subset(test_ds, subset_fraction=1)

    return animal_train_ds, animal_test_ds

def load_cifar10_full_animals(batch_size=32):

    train_ds, test_ds = load_cifar10(batch_size=batch_size)
    animal_train_ds = create_animal_subset(train_ds, subset_fraction=1)
    animal_test_ds = create_animal_subset(test_ds, subset_fraction=1)

    return animal_train_ds, animal_test_ds

def load_cifar10_quartered_animals_save(batch_size=32):

    train_ds, test_ds = load_cifar10(batch_size=batch_size)
    animal_train_ds = create_animal_subset(train_ds, subset_fraction=0.25)
    animal_test_ds = create_animal_subset(test_ds, subset_fraction=1)

    # 💾 Save train subset for analysis
    save_dataset_snapshot(animal_train_ds, name_prefix="Multirun_Quartered_Animals_V2_Run")

    return animal_train_ds, animal_test_ds

def load_cifar10_quartered_nonanimals(batch_size=32):

    train_ds, test_ds = load_cifar10(batch_size=batch_size)
    non_animal_train_ds = create_nonanimal_subset(train_ds, subset_fraction=0.25)  # 12.5% az összből
    non_animal_test_ds = create_nonanimal_subset(test_ds, subset_fraction=1)
    return non_animal_train_ds, non_animal_test_ds

def load_cifar10_full_animals_siamese(batch_size=32, num_pairs_per_class=1000):
    train_ds, test_ds = load_cifar10(batch_size=batch_size)

    animal_train_ds = create_animal_subset(train_ds, subset_fraction=1.0)
    animal_test_ds = create_animal_subset(test_ds, subset_fraction=1.0)

    X_train, y_train = extract_numpy_data(animal_train_ds)
    X_test, y_test = extract_numpy_data(animal_test_ds)

    train_dataset = generate_siamese_pairs_dataset(X_train, y_train, num_pairs_per_class=num_pairs_per_class, batch_size=batch_size)
    val_dataset = generate_siamese_pairs_dataset(X_test, y_test, num_pairs_per_class=num_pairs_per_class // 4, batch_size=batch_size)

    return train_dataset, val_dataset