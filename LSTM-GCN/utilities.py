import os
import pandas as pd
import torch
from tqdm import tqdm
import shutil


class StandardScaler():
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def create_dir(dir_name: str):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def save_as_pt_file(dataset, path: str):
    for i, data in tqdm(enumerate(dataset)):
        file_path = os.path.join(path, f'data_{i}.pt')

        if os.path.exists(file_path):
            os.remove(file_path)

        torch.save(data, file_path)


def split_list_of_data(dataset, train_ratio, val_ratio):
    # Calculate sizes for each split
    train_size = int(len(dataset) * train_ratio)
    val_size = int(len(dataset) * val_ratio)
    # test_size = len(dataset) - train_size - val_size  # Ensure all samples are included

    # Split the dataset
    train_set = dataset[:train_size]
    val_set = dataset[train_size:train_size + val_size]
    test_set = dataset[train_size + val_size:]

    return train_set, val_set, test_set


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


def print_model_parameters(model):
    # Get the total number of parameters
    total_params = sum(p.numel() for p in model.parameters())

    # Get the number of trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Get the number of non-trainable parameters
    non_trainable_params = total_params - trainable_params

    print(f'Total parameters: {total_params}')
    print(f'Trainable parameters: {trainable_params}')
    print(f'Non-trainable parameters: {non_trainable_params}')


def create_empty_dir(directory_path):
    # Check if the directory exists
    if os.path.exists(directory_path):
        # Remove all contents of the directory
        shutil.rmtree(directory_path)
    # Create the directory again (empty)
    os.makedirs(directory_path)


def standard_scaler():
    mean_std_path = r'./MeanStd/mean_std.pkl'
    mean_std_df = pd.read_pickle(mean_std_path)
    # features = ['open', 'high', 'low', 'close']
    features = ['Close']
    mean = mean_std_df['OHLC_plus_mean'][features].values
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mean = torch.tensor(mean, device=device).view(1, 1, 1, -1)  # reshaped to = (1, 1, 4)
    std = mean_std_df['OHLC_plus_std'][features].values
    std = torch.tensor(std, device=device).view(1, 1, 1, -1)  # reshaped to = (1, 1, 4)
    scaler = StandardScaler(mean, std)

    return scaler
