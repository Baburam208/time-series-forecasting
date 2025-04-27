import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
from typing import List

from torch_geometric_temporal.signal import StaticGraphTemporalSignal

from utilities import StandardScaler


def features_dataframe(file_path):
    MEAN_STD = dict()
    df = pd.read_csv(file_path)

    # All features
    # all_features = ['date', 'Symbol', 'Open', 'High', 'Low', 'Close', 'Vol', 'EMA_14',
    #                        'MACD_12_26', 'MACD_Signal_9', 'RSI_14', 'BB_Mid', 'BB_Upper',
    #                        'BB_Lower', 'ATR_14', 'VWAP']

    # Selected features
    selected_features = ['date', 'Symbol', 'Open', 'High', 'Low', 'Close', 'Vol', 'EMA_14',
                         'MACD_12_26', 'MACD_Signal_9', 'RSI_14', 'BB_Mid', 'BB_Upper',
                         'BB_Lower', 'ATR_14', 'VWAP']

    # Define OHLC and traded volume separately
    ohlc_plus_features = ['Open', 'High', 'Low', 'Close', 'EMA_14',
                          'MACD_12_26', 'MACD_Signal_9', 'RSI_14', 'BB_Mid', 'BB_Upper',
                          'BB_Lower', 'ATR_14', 'VWAP']
    volume_feature = ['Vol']

    df_ohlc_plus = df[ohlc_plus_features].copy()
    df_vol = df[volume_feature].copy()

    # Normalize features, except Vol with Z-score normalization
    ohlc_mean = df_ohlc_plus[ohlc_plus_features].mean()
    ohlc_std = df_ohlc_plus[ohlc_plus_features].std()
    # df_selected[ohlc_plus_features] = (df_selected[ohlc_plus_features] - ohlc_mean) / ohlc_std
    df_ohlc_plus[ohlc_plus_features] = df_ohlc_plus[ohlc_plus_features].apply(lambda x: (x - ohlc_mean) / ohlc_std,
                                                                              axis=1)

    # Log normalize Vol, then standardize
    df_vol[volume_feature] = np.log1p(df_vol[volume_feature])  # Log normalization
    volume_mean = df_vol[volume_feature].mean()
    volume_std = df_vol[volume_feature].std()
    df_vol[volume_feature] = df_vol[volume_feature].apply(lambda x: (x - volume_mean) / volume_std, axis=1)

    # Store the means and standard deviations for scaling back or future use
    MEAN_STD['OHLC_plus_mean'] = ohlc_mean
    MEAN_STD['OHLC_plus_std'] = ohlc_std
    MEAN_STD['Vol_mean'] = volume_mean
    MEAN_STD['Vol_std'] = volume_std

    df_final = df_ohlc_plus.join(df['Symbol'])
    df_final = df_final.join(df_vol)

    features = ['Open', 'High', 'Low', 'Close', 'Vol', 'EMA_14',
                'MACD_12_26', 'MACD_Signal_9', 'RSI_14', 'BB_Mid', 'BB_Upper',
                'BB_Lower', 'ATR_14', 'VWAP', 'Symbol']

    df_final = df_final[features]

    return df_final, MEAN_STD


def get_features(df, stock_list):
    """
    Groups a DataFrame based on a predefined order of 'stock' values
    and extracts selected features for each group.

    Args:
        df (pd.DataFrame): Input DataFrame containing a 'stock' column.
        stock_list (list): Predefined order of stocks to group by.

    Returns:
        list: List of lists containing the target features grouped by 'stock'.
    """
    # Validate input types
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input 'df' must be a pandas DataFrame")
    if not isinstance(stock_list, list):
        raise ValueError("Input 'stock_list' must be a list")

    # Define the target features to extract
    target_features = ['Open', 'High', 'Low', 'Close', 'Vol', 'EMA_14',
                       'MACD_12_26', 'MACD_Signal_9', 'RSI_14', 'BB_Mid', 'BB_Upper',
                       'BB_Lower', 'ATR_14', 'VWAP']

    # Ensure 'Symbol' column respects the predefined order
    df['Symbol'] = pd.Categorical(df['Symbol'], categories=stock_list, ordered=True)

    # Group the dataframe by the 'Symbol' column
    grouped_df = df.groupby('Symbol')

    # Earlier
    # Extract features for each group
    # stock_snapshots = grouped_df[target_features].apply(lambda x: x.values.tolist()).tolist()

    STOCKS_SNAPSHOTS = []

    for _, group in tqdm(grouped_df):
        # Append the features for each station to the list
        snapshot = group[target_features].values.tolist()
        STOCKS_SNAPSHOTS.append(snapshot)

    return STOCKS_SNAPSHOTS


def create_snapshots(file_path):
    # file_path = r"./modified_working_datasets"

    dataframe, mu_rho = features_dataframe(file_path)

    # Saving the `dataframe` to file `Std_Dataframe`.
    # `mu_rho` to file `Mu_Rho`
    if not os.path.exists('StandardizedDataframe'):
        os.makedirs('StandardizedDataframe', exist_ok=True)
    dataframe.to_csv('./StandardizedDataframe/Standard_Dataframe.csv', index=False)

    # Save the dictionary to a file
    if not os.path.exists('MeanStd'):
        os.makedirs('MeanStd', exist_ok=True)
    with open('./MeanStd/mean_std.pkl', 'wb') as f:
        pickle.dump(mu_rho, f)

    stock_list_path = r'F:\New Training\New Data Preparation - Updated\preprocess_logging\stock_list.txt'
    with open(stock_list_path, 'r') as file:
        stock_list = file.read().splitlines()

    snapshots = get_features(dataframe, stock_list=stock_list)

    print(f"Stocks: {len(snapshots)}")

    snap = np.array(snapshots)

    # np.save(os.path.join(path_snapshots, "snaps.npy"), snap)
    # print(f"Saved successfully `snap` !!!")

    print(f"snap shape: {snap.shape}")
    snap_transpose = np.transpose(snap, axes=(0, 2, 1))

    # np.save(os.path.join(path_snapshots, "snaps_transpose.npy"), snap_transpose)
    # print(f"Saved successfully `snap_transpose` !!!")

    print(f"snap_transpose shape: {snap_transpose.shape}")

    return snap_transpose


class StockDatasetLoader(object):
    def __init__(self, snapshots, edge_index, edge_weight, scalar):
        self._snapshots = snapshots
        self._snapshots = self._snapshots
        self._edge_index = edge_index
        self._edge_weight = edge_weight
        self.scalar = scalar

    def _get_edge_index(self):
        self._edges = torch.load(self._edge_index)

    def _get_edge_weights(self):
        self._edge_weights = torch.load(self._edge_weight).to(torch.float32)

    def _get_targets_and_features(self):
        stacked_target = self._snapshots
        self.features = [
            np.transpose(stacked_target[:, :, i: (i + self.lags)], axes=(0, 2, 1))
            for i in range(stacked_target.shape[-1] - self.lags - self._pred_seq)
        ]

        # ['open', 'high', 'low',	'close', 'traded_quantity']

        self.targets = [
            np.squeeze(
                stacked_target[:, [3], (i + self.lags):(i + self.lags + self._pred_seq)],
                axis=-2)  # only 'close' price

            for i in range(stacked_target.shape[-1] - self.lags - self._pred_seq)
        ]

    def get_dataset(self, lags: int = 30, pred_seq: int = 6) -> StaticGraphTemporalSignal:
        self.lags = lags
        self._pred_seq = pred_seq
        self._get_edge_index()
        self._get_edge_weights()
        self._get_targets_and_features()
        dataset = StaticGraphTemporalSignal(
            self._edges, self._edge_weights, self.features, self.targets
        )
        return dataset


def create_graph_snapshot(file_path: str, edge_index_path: str, edge_weight_path: str, pred=1, lags=37) -> List[
    StaticGraphTemporalSignal]:
    snaps = create_snapshots(file_path)

    print(f"{snaps.shape = }")

    mean_std_path = r"./MeanStd/mean_std.pkl"

    mean_std_df = pd.read_pickle(mean_std_path)
    features = ['Open', 'High', 'Low', 'Close', 'EMA_14', 'MACD_12_26', 'MACD_Signal_9',
                'RSI_14', 'BB_Mid', 'BB_Upper', 'BB_Lower', 'ATR_14', 'VWAP']
    close_feature = ['Close']
    mean = mean_std_df['OHLC_plus_mean'][close_feature].values
    std = mean_std_df['OHLC_plus_std'][close_feature].values
    scalar = StandardScaler(mean, std)
    loader = StockDatasetLoader(
        snapshots=snaps,
        edge_index=edge_index_path,
        edge_weight=edge_weight_path,
        scalar=scalar
    )

    dataset = loader.get_dataset(lags=lags, pred_seq=pred)
    # print(f"{dataset = }")
    # print(f"{type(dataset) = }")

    datasets = [data for data in dataset]

    # print(f"Shape of data: {datasets[0]}")
    # print(f"{datasets[0].x = }")
    # print(f"{datasets[0].y = }")

    return datasets

