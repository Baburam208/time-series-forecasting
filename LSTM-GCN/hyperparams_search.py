from typing import List
import random
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
import os
from torch.optim.lr_scheduler import LambdaLR

from model import ASTGCNmodel
import utilities
import Dataloader.dataloader as dataloader
import GPUtil

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()

# Hyperparameter search space
nb_block_options = [2, 4]
nb_chev_filter_options = [32, 64]
nb_time_filter_options = [32, 64]
batch_size_options = [8, 16]
learning_rate_options = [1e-2, 1e-3]
optimizer_options = ['adam', 'rmsprop']

# Random search setup
num_combinations = 8
random_combinations = [
    {
        'nb_block': random.choice(nb_block_options),
        'nb_chev_filter': random.choice(nb_chev_filter_options),
        'nb_time_filter': random.choice(nb_time_filter_options),
        'batch_size': random.choice(batch_size_options),
        'learning_rate': random.choice(learning_rate_options),
        'optimizer': random.choice(optimizer_options)
    }
    for _ in range(num_combinations)
]

# Dataset paths
train_data_path = './Train'
valid_data_path = './Validation'
test_data_path = './Test'

# Define the learning rate decay function
def lr_decay(epoch):
    """Returns the decay factor based on epoch."""
    return 0.7 ** (epoch // 5)  # Decay by 0.7 every 5 epochs


def run_random_search():
    """Runs training and evaluation for each random combination of hyperparameters."""
    best_val_loss = float('inf')
    best_config = None

    for idx, config in enumerate(random_combinations):
        print(f"\nRunning configuration {idx + 1}/{num_combinations}: {config}")

        # Model initialization
        model = ASTGCNmodel(
            config['nb_block'],
            in_channels=5,
            K=3,
            nb_chev_filter=config['nb_chev_filter'],
            nb_time_filter=config['nb_time_filter'],
            time_strides=2,
            num_for_predict=1,
            len_input=40,
            num_of_vertices=38
        ).to(device)

        utilities.print_model_parameters(model)

        # Optimizer setup
        if config['optimizer'] == 'adam':
            optimizer = torch.optim.Adam(
                model.parameters(), lr=config['learning_rate'], weight_decay=1e-5
            )
        elif config['optimizer'] == 'rmsprop':
            optimizer = torch.optim.RMSprop(
                model.parameters(), lr=config['learning_rate'], weight_decay=1e-5
            )

        scheduler = LambdaLR(optimizer, lr_lambda=lr_decay)

        # DataLoader instances
        train_loader = dataloader.train_loader(batch_size=config['batch_size'], data_dir=train_data_path, num_workers=0)
        val_loader = dataloader.validation_loader(batch_size=config['batch_size'], data_dir=valid_data_path, num_workers=0)

        # Training and evaluation
        train_losses = []
        eval_losses = []

        def train():
            model.train()
            train_loss = 0.0
            for batch, train_batch in enumerate(train_loader, 1):
                cost = 0.0
                for time, data in enumerate(train_batch, 1):
                    data.to(device)
                    y_hat = model(data.x, data.edge_index)
                    cost += F.mse_loss(y_hat, data.y)
                    del data

                cost /= time
                train_loss += cost.item()
                cost.backward()
                optimizer.step()
                optimizer.zero_grad()
                del train_batch

            scheduler.step()
            train_loss /= batch
            train_losses.append(train_loss)
            return train_loss

        @torch.no_grad()
        def evaluate():
            model.eval()
            eval_loss = 0.0
            with torch.no_grad():
                for batch, val_batch in enumerate(val_loader, 1):
                    val_cost = 0.0
                    for time, data in enumerate(val_batch, 1):
                        data.to(device)
                        y_hat_val = model(data.x, data.edge_index)
                        val_cost += F.mse_loss(y_hat_val, data.y)
                        del data

                    del val_batch
                    val_cost /= time
                    eval_loss += val_cost.item()
                eval_loss /= batch
                eval_losses.append(eval_loss)
            return eval_loss

        # Training loop
        num_epochs = 30
        for epoch in range(num_epochs):
            train_loss = train()
            val_loss = evaluate()
            print(f"Epoch {epoch+1:03d}/{num_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_config = config
                print(f"New best validation loss: {val_loss:.6f} for config {config}")

    print("\nBest configuration:")
    print(best_config)
    print(f"Best validation loss: {best_val_loss:.6f}")


if __name__ == '__main__':
    run_random_search()
