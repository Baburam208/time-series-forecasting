import torch
import os
import re
from torch_geometric_temporal.signal import temporal_signal_split
from torch.utils.data import random_split

from torch.utils.data import Dataset, DataLoader


########################
## Custom Datasetloader
########################
def custom_collate(batch):
    # return Batch.from_data_list(batch)
    return batch


class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.file_list = sorted(os.listdir(data_dir), key=lambda x: int(re.search(r'\d+', x).group()))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = os.path.join(self.data_dir, self.file_list[idx])
        data = torch.load(file_name)

        return data


class CustomTrainDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.file_list = os.listdir(data_dir)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = os.path.join(self.data_dir, self.file_list[idx])
        data = torch.load(file_name)

        return data


def loader(bs_train: int = 64,
           bs_val: int = 64,
           bs_test: int = 64,
           num_workers: int = 0,
           data_dir: str = "./Saved_Data"):

    dataset = CustomDataset(data_dir)

    # Calculate the sizes for train, val, and test sets
    total_size = len(dataset)
    train_size = int(0.9 * total_size)  # 90% for training
    val_size = int(0.05 * total_size)    # 5% for validation
    test_size = total_size - train_size - val_size  # Remaining 5% for testing

    # Use random_split to create train, val, and test datasets
    train_dataset, val_dataset, test_dataset = random_split(dataset, lengths=[train_size, val_size, test_size])

    # Create DataLoader instances for train, val, and test sets
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=bs_train,
                                  shuffle=True,
                                  collate_fn=custom_collate,
                                  num_workers=num_workers)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=bs_val,
                                shuffle=False,
                                collate_fn=custom_collate,
                                num_workers=num_workers)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=bs_test,
                                 shuffle=False,
                                 collate_fn=custom_collate,
                                 num_workers=num_workers)

    return train_dataloader, val_dataloader, test_dataloader


def train_loader(batch_size: int = 4,
                 data_dir: str = './Train',
                 shuffle: bool = False,
                 num_workers: int = 8):
    dataset = CustomTrainDataset(data_dir)

    # Create DataLoader instances for train, val, and test sets
    train_dataloader = DataLoader(dataset,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  collate_fn=custom_collate,
                                  num_workers=num_workers)

    return train_dataloader


def validation_loader(batch_size: int = 4,
                      data_dir: str = './Validation',
                      shuffle: bool = False,
                      num_workers: int = 8):
    dataset = CustomDataset(data_dir)

    # Create DataLoader instances for train, val, and test sets
    validation_dataloader = DataLoader(dataset,
                                       batch_size=batch_size,
                                       shuffle=shuffle,
                                       collate_fn=custom_collate,
                                       num_workers=num_workers)

    return validation_dataloader


def test_loader(batch_size: int = 4,
                data_dir: str = './Test',
                shuffle: bool = False,
                num_workers: int = 8):
    dataset = CustomDataset(data_dir)

    # Create DataLoader instances for train, val, and test sets
    test_dataloader = DataLoader(dataset,
                                 batch_size=batch_size,
                                 shuffle=shuffle,
                                 collate_fn=custom_collate,
                                 num_workers=num_workers)

    return test_dataloader
