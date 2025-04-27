import torch
import numpy as np


def mse_(preds, labels):
    loss = (labels - preds)**2
    return torch.mean(loss)


def rmse_(preds, labels):
    return torch.sqrt(mse_(preds=preds, labels=labels))


def mae_(preds, labels):
    loss = torch.abs(labels - preds)
    return torch.mean(loss)


def mape_(preds, labels):
    epsilon = 1e-8  # to prevent division by zero.
    loss = torch.abs(labels-preds)/(labels + epsilon)
    return torch.mean(loss) * 100


def metric(pred, real):
    # Combine lists of NumPy arrays into single arrays
    pred = np.concatenate(pred)  # Shape: (num_samples, num_features)
    real = np.concatenate(real)

    # print(f"{pred.shape = }")
    # print(f"{real.shape = }")
    # pred.shape = (164, 1, 46, 1)
    # real.shape = (164, 1, 46, 1)

    # Convert NumPy arrays to PyTorch tensors
    pred = torch.tensor(pred)
    real = torch.tensor(real)

    # pred = torch.tensor(np.array(pred))
    # real = torch.tensor(np.array(real))

    mse = mse_(pred, real).item()
    mae = mae_(pred,real).item()
    mape = mape_(pred,real).item()
    rmse = rmse_(pred,real).item()

    return mse, mae, mape, rmse
