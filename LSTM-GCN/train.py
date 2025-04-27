import random
import numpy as np
import warnings
from typing import List
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
import os
from torch.optim.lr_scheduler import LambdaLR

from model import LSTMGCN
import utilities
import Dataloader.dataloader as dataloader
# import GPUtil
from configuration import ModelConfiguration, TrainConfiguration

plt.rcParams["font.family"] = "Times New Roman"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()

config_model = ModelConfiguration()
config_train = TrainConfiguration()

model = LSTMGCN(input_size=14,
                hidden_size=config_model.hidden_size,
                num_layers=config_model.num_layers,
                lstm_dropout_rate=config_model.lstm_dropout_rate,
                in_channels=config_model.hidden_size,
                hidden_channels=config_model.hidden_channels,
                out_channels=config_model.out_channels,
                hidden_features=config_model.hidden_features,
                fc_dropout_rate=config_model.fc_dropout_rate,
                k=config_model.k,
                forecast_horizon=config_model.forecast_horizon).to(device)

utilities.print_model_parameters(model)

# optimizer = torch.optim.Adam(
#     model.parameters(),  # Parameters to optimize
#     lr=config_train.lr_rate,          # Learning rate
#     betas=(0.9, 0.999),  # Coefficients for moving averages of gradient and its square
#     eps=1e-8,            # Small epsilon to prevent division by zero
#     weight_decay=0       # L2 regularization strength
# )
optimizer = torch.optim.RMSprop(model.parameters(),
                                lr=0.001,
                                alpha=0.99,
                                eps=1e-8,
                                weight_decay=0,
                                momentum=0.9)

# optimizer = torch.optim.SGD(
#     model.parameters(),
#     lr=config_train.lr_rate,
#     momentum=0.9,
#     weight_decay=1e-6
# )

# optimizer = torch.optim.AdamW(model.parameters(),
#                               lr=config_train.lr_rate,
#                               weight_decay=0)

# Create DataLoader instances for train, val, and test sets
bs_train = config_train.bs_train
bs_val = config_train.bs_val
bs_test = config_train.bs_test
num_workers = 0

train_data_path = './Train'
valid_data_path = './Validation'
test_data_path = './Test'

train_loader = dataloader.train_loader(batch_size=bs_train, data_dir=train_data_path, num_workers=num_workers)
val_loader = dataloader.validation_loader(batch_size=bs_val, data_dir=valid_data_path, num_workers=num_workers)
test_loader = dataloader.test_loader(batch_size=bs_test, data_dir=test_data_path, num_workers=num_workers)

print(f"train_loader: {len(train_loader)}")
print(f"val_loader: {len(val_loader)}")
print(f"test_loader: {len(test_loader)}")

scaler = utilities.standard_scaler()


# Define the learning rate decay function
def lr_decay(epoch):
    """Returns the decay factor based on epoch."""
    return 0.7 ** (epoch // 5)  # Decay by 0.7 every 5 epochs


# Create the learning rate scheduler
# scheduler = LambdaLR(optimizer, lr_lambda=lr_decay)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)  # Reduce LR every 10 epochs by 10%

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

train_losses: List[float] = []
eval_losses: List[float] = []


def trainer():
    model.train()
    train_loss: float = 0.0
    for batch, train_batch in enumerate(train_loader, 1):
        cost: float = 0.0
        for time, data in enumerate(train_batch, 1):
            data.to(device)
            # y_hat = model(data.x, data.edge_index, data.edge_weight)
            y_hat = model(data.x, data.edge_index)
            cost = cost + F.mse_loss(y_hat, data.y)
            del data

        cost = cost / time
        train_loss += cost.item()

        # Backward pass and optimization
        cost.backward()

        # Gradient clipping (optional)
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        optimizer.zero_grad()
        del train_batch

    # Print epoch info
    # current_lr = scheduler.get_last_lr()[0]
    # current_lr = config_train.lr_rate
    current_lr = optimizer.param_groups[0]['lr']

    train_loss /= batch
    train_losses.append(train_loss)
    # return train_loss, current_lr

    model.eval()
    eval_loss = 0
    with torch.no_grad():
        for batch, val_batch in enumerate(val_loader, 1):
            val_cost = 0
            for time, data in enumerate(val_batch, 1):
                data.to(device)
                # y_hat_val = model(data.x, data.edge_index, data.edge_weight)
                y_hat_val = model(data.x, data.edge_index)
                val_cost = val_cost + F.mse_loss(y_hat_val, data.y)
                del data

            del val_batch
            val_cost /= time

            eval_loss += val_cost.item()
        eval_loss /= batch
        eval_losses.append(eval_loss)

    # Update the learning rate
    scheduler.step(eval_loss)

    return train_loss, eval_loss, current_lr


# Saving best model
utilities.create_dir("model_checkpoint")
timestamp = time.strftime("%Y-%m-%d-%H-%M")
print(f"Timestamp: {timestamp}")


def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours}h {minutes}m {seconds}s"


# Initialize early stopping
early_stopping = utilities.EarlyStopping(patience=config_train.patience, min_delta=0.01)


def training(num_epochs: int = 60):
    total_start_time = time.time()
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        train_loss, val_loss, current_lr = trainer()

        print(f'Epoch {epoch + 1:03d}/{num_epochs:03d}, {train_loss=:.6f}, {val_loss=:.6f}'
              f'\tLearning Rate: {current_lr:.6f}')

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time

        '''
        # Get GPU usage
        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            gpu_load = gpu.load * 100
            gpu_memory_used = gpu.memoryUsed
            gpu_memory_total = gpu.memoryTotal
            print(
                f"Time taken: {format_time(epoch_duration)} - GPU Load: {gpu_load:.2f}% - GPU Memory Usage: {gpu_memory_used:.2f}MB/{gpu_memory_total:.2f}MB")
        '''

        # Check early stopping condition
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping !!!")
            break

        # Save the model if the validation loss has decreased
        if val_loss < early_stopping.best_loss:
            best_model_path = os.path.join("model_checkpoint", f'{timestamp}_best_model_epoch_{epoch + 1}.pth')
            torch.save(model.state_dict(), best_model_path)
            print(
                f"\nSaved model at epoch {epoch + 1} with training loss={train_loss} and validation loss={val_loss}\n")
            print(f"Model weights file: {best_model_path}\n")
            early_stopping.best_loss = val_loss

    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    print(f"Total training time: {format_time(total_duration)}")

    return


def plot():
    # PLOTTING STUFFS
    time_stamp = time.strftime("%Y-%m-%d-%H-%M")

    # Save the trained model
    torch.save(model.state_dict(), f'last_trained_weight/Stock_Forecast_Model_{time_stamp}.pth')
    print(f'last_trained_weight/Stock_Forecast_Model_{time_stamp}.pth')

    # Plotting
    plt.title("Loss curve")
    plt.plot(range(len(train_losses)), train_losses, label="Training Loss")
    plt.plot(range(len(eval_losses)), eval_losses, label="Validation Loss")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Losses")

    # Saving the learning curve
    if not os.path.exists('plots'):
        os.makedirs(name='plots', exist_ok=True)
    plt.savefig(os.path.join('plots', f'learning_curve_{time_stamp}.png'), dpi=300, bbox_inches='tight')

    plt.show()


def set_env(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main():
    seed = 42
    set_env(seed)

    warnings.filterwarnings(action="ignore", category=FutureWarning)
    warnings.filterwarnings(action="ignore", category=UserWarning)

    print("Training Started")
    print('=' * 50)

    # For model checkpointing
    if not os.path.exists('model_checkpoint'):
        os.makedirs(name='model_checkpoint', exist_ok=True)

    if not os.path.exists('last_trained_weight'):
        os.makedirs(name='last_trained_weight', exist_ok=True)

    # Saving the learning curve
    if not os.path.exists('plots'):
        os.makedirs(name='plots', exist_ok=True)

    training(num_epochs=60)
    plot()
    print("Finished Training!!!")


if __name__ == '__main__':
    main()
