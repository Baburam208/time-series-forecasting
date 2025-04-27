from dataclasses import dataclass


@dataclass
class ModelConfiguration:
    # hyperparameters for TemporalEncoder
    num_layers = 4
    hidden_size = 64
    lstm_dropout_rate = 0.0

    # hyperparameters for SpatialEncoder
    hidden_channels = 128
    out_channels = 64
    k = 3

    # hyperparameter for FC Layer
    hidden_features = 32
    fc_dropout_rate = 0.0
    forecast_horizon = 1


@dataclass
class TrainConfiguration:
    bs_train = 4
    bs_val = 4
    bs_test = 4
    lr_rate = 0.001
    patience = 60
