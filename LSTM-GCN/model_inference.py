import os

import pandas as pd
import torch
import time
import matplotlib.pyplot as plt

import metrics
import utilities
from metrics import metric
from model import LSTMGCN
import Dataloader.dataloader as dataloader
from utilities import standard_scaler
from configuration import ModelConfiguration, TrainConfiguration
from utilities import standard_scaler

plt.rcParams["font.family"] = "Times New Roman"

mean_std = pd.read_pickle('./MeanStd/mean_std.pkl')

mean = mean_std['OHLC_plus_mean']['Close']
std = mean_std['OHLC_plus_std']['Close']

@torch.no_grad()
def model_inference(model,
                    dataloader,
                    scaler,
                    device,
                    sample_stock,
                    time_stamp,
                    plot_name):
    model.eval()
    all_preds, all_targets = [], []
    sample_preds, sample_targets = [], []
    with torch.no_grad():
        for batch, test_batch in enumerate(dataloader, 1):
            val_cost = 0
            for time, data in enumerate(test_batch, 1):
                data.to(device)
                # y_hat_val = model(data.x, data.edge_index, data.edge_attr)
                y_hat_val = model(data.x, data.edge_index)
                # y_hat_val = model(data.x, data.edge_index)

                y_hat_val = scaler.inverse_transform(y_hat_val)
                y_label = scaler.inverse_transform(data.y)

                all_preds.append(y_hat_val.detach().cpu().numpy())
                all_targets.append(y_label.detach().cpu().numpy())

                if sample_stock is not None:
                    assert isinstance(sample_stock, int), "`sample_stock` should be a single integer."
                    sample_preds.append(torch.squeeze(y_hat_val[:, :, sample_stock, :]).item())
                    sample_targets.append(torch.squeeze(y_label[:, :, sample_stock, :]).item())

                del data

            del test_batch

    mse, mae, mape, rmse = metrics.metric(all_preds, all_targets)

    plot_sample_stock((sample_preds, sample_targets), sample_stock, time_stamp, plot_name)

    # plot_actual_price((sample_preds, sample_targets), sample_stock, time_stamp, plot_name)

    return mse, mae, mape, rmse


def plot_sample_stock(y, sample_stock, time_stamp, plot_name):
    y_pred = y[0]
    y_actual = y[1]

    plt.title("Actual versus predicted stock price")

    plt.plot(range(len(y_actual)), y_actual, label="Actual")
    plt.plot(range(len(y_pred)), y_pred, label="Predicted")

    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Close price")

    # Saving the stock price plot
    if not os.path.exists('sample stock plot'):
        os.makedirs(name='sample stock plot', exist_ok=True)

    plt.savefig(os.path.join('sample stock plot',
                f'{sample_stock}_{plot_name}_{time_stamp}.png'),
                dpi=300, bbox_inches='tight')

    plt.show()


def plot_actual_price(y, sample_stock, time_stamp, plot_name):
    # y_pred = y[0]
    y_actual = y[1]

    plt.title("Actual stock price")

    plt.plot(range(len(y_actual)), y_actual, label="Actual")
    # plt.plot(range(len(y_pred)), y_pred, label="Predicted")

    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Close price")

    # # Saving the learning curve
    # if not os.path.exists('sample stock plot'):
    #     os.makedirs(name='sample stock plot', exist_ok=True)
    #
    # plt.savefig(os.path.join('sample stock plot',
    #             f'{sample_stock}_{plot_name}_{time_stamp}.png'),
    #             dpi=300, bbox_inches='tight')


    plt.show()


def save_metrics_to_csv(dst_dir, mse, mae, mape, rmse, evaluation_time, timestamp, metrics_csv_file):
    """
    Save model metrics to a CSV file in the specified destination directory.

    Parameters:
        dst_dir (str): Directory to save the CSV file.
        mse (float): Mean Squared Error value.
        mae (float): Mean Absolute Error value.
        mape (float): Mean Absolute Percentage Error value.
        rmse (float): Root Mean Squared Error value.
        training_time (float): Total training time in seconds.
        evaluation_time (float): Total inference time in seconds
    """
    # Ensure the destination directory exists
    os.makedirs(dst_dir, exist_ok=True)

    # Prepare metrics data as a dictionary
    metrics = {
        'MSE': [mse],
        'MAE': [mae],
        'MAPE': [mape],
        'RMSE': [rmse],
        'inference_time': [evaluation_time]
    }

    # Create DataFrame
    df = pd.DataFrame(metrics)

    # Construct the file path
    file_path = os.path.join(dst_dir, f'test_metrics_({metrics_csv_file})_{timestamp}.csv')

    try:
        # Save to CSV
        df.to_csv(file_path, mode='a', index=False)
        print(f"Metrics successfully saved to: {file_path}")
    except Exception as e:
        print(f"Error saving metrics: {e}")


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")

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

    best_model_weight_path = r'model_checkpoint\best_model_2025-01-05_14-49-04_epoch_25.pth'
    last_model_weight_path = r'last_trained_weight/Stock_Forecast_Model_2025-04-18-12-48.pth'

    is_best_model = False
    if is_best_model:
        model_weight_path = best_model_weight_path
        plot_name = "best_model_pred_and_target_price_plot"
        metrics_csv_file = 'best'
    else:
        model_weight_path = last_model_weight_path
        plot_name = "last_model_pred_and_target_price_plot"
        metrics_csv_file = 'last'

    model.load_state_dict(torch.load(model_weight_path, map_location=device))

    test_data_path = './Test'
    bs_test = 16
    num_workers = 0
    test_loader = dataloader.test_loader(batch_size=bs_test, data_dir=test_data_path, num_workers=num_workers)
    scaler = standard_scaler()

    eval_start = time.time()

    sample_stock = 0  # ADBL

    mse, mae, mape, rmse = model_inference(model,
                                           test_loader,
                                           scaler,
                                           device,
                                           sample_stock,
                                           timestamp,
                                           plot_name)
    eval_end = time.time()

    eval_time = eval_end - eval_start

    print(f"Test MSE: {mse:.4f}")
    print(f"Test MAE: {mae:.4f}")
    print(f"Test MAPE: {mape:.4f}%")
    print(f"Test RMSE: {rmse:.4f}")

    dst_dir = 'test evaluation metrics'
    save_metrics_to_csv(dst_dir, mse, mae, mape, rmse, eval_time, timestamp, metrics_csv_file)
