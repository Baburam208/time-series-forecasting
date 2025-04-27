import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import ChebConv


class TemporalEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, lstm_dropout_prob):
        super(TemporalEncoder, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=lstm_dropout_prob)

        # self.fc_dropout = nn.Dropout(0.2)
        #
        # self.fc1 = nn.Linear(in_features=hidden_size, out_features=64)
        # self.fc2 = nn.Linear(in_features=64, out_features=horizon)

    def forward(self, x):
        out, _ = self.lstm(x)  # x has shape: (B, T, F_in)

        out = out[:, -1, :]

        # out = self.fc1(out[:, -1, :])

        # out = self.fc2(out)
        #
        # out = out.unsqueeze(0)

        return out  # (B, F_out)


# class SpatialEncoder(nn.Module):
#     def __init__(self,
#                  in_channels,
#                  hidden_channels,
#                  out_channels,
#                  k=1):
#         super(SpatialEncoder, self).__init__()
#
#         self.chebconv1 = ChebConv(in_channels, hidden_channels, K=k)
#         self.chebconv2 = ChebConv(hidden_channels, hidden_channels, K=k)
#         self.chebconv3 = ChebConv(hidden_channels, hidden_channels, K=k)
#         self.chebconv4 = ChebConv(hidden_channels, out_channels, K=k)
#
#     def forward(self, x, edge_index, edge_weight):
#         out = self.chebconv1(x, edge_index, edge_weight)
#         out = F.relu(out)
#         out = self.chebconv2(out, edge_index, edge_weight)
#         out = F.relu(out)
#         # out = self.chebconv3(out, edge_index, edge_weight)
#         # out = F.relu(out)
#         out = self.chebconv4(out, edge_index, edge_weight)
#
#         # out = self.fc1(out)
#         # out = F.relu(out)
#         # out = self.fc2(out)
#
#         return out


class SpatialEncoder(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 k=1):
        super(SpatialEncoder, self).__init__()

        self.chebconv1 = ChebConv(in_channels, hidden_channels, K=k)
        # self.chebconv2 = ChebConv(hidden_channels, hidden_channels, K=k)
        # self.chebconv3 = ChebConv(hidden_channels, hidden_channels, K=k)
        self.chebconv4 = ChebConv(hidden_channels, out_channels, K=k)

    def forward(self, x, edge_index):
        out = self.chebconv1(x, edge_index)
        out = F.relu(out)
        # out = self.chebconv2(out, edge_index)
        # out = F.relu(out)
        # out = self.chebconv3(out, edge_index)
        # out = F.relu(out)
        out = self.chebconv4(out, edge_index)

        # out = self.fc1(out)
        # out = F.relu(out)
        # out = self.fc2(out)

        return out


class LSTMGCN(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers,
                 lstm_dropout_rate,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 hidden_features,
                 fc_dropout_rate,
                 k,
                 forecast_horizon):
        super(LSTMGCN, self).__init__()
        self.temporal_encoder = TemporalEncoder(input_size=input_size,
                                                hidden_size=hidden_size,
                                                num_layers=num_layers,
                                                lstm_dropout_prob=lstm_dropout_rate)

        self.spatial_encoder = SpatialEncoder(in_channels,
                                              hidden_channels,
                                              out_channels,
                                              k=k)

        self.lin1 = nn.Linear(in_features=out_channels, out_features=hidden_features)
        self.lin2 = nn.Linear(in_features=hidden_features, out_features=forecast_horizon)

        self.fc_dropout = nn.Dropout(fc_dropout_rate)

    def forward(self, x, edge_index):
        out = self.temporal_encoder(x)
        out = self.spatial_encoder(out, edge_index)

        out = self.lin1(out)
        out = F.relu(out)
        out = self.fc_dropout(out)
        out = self.lin2(out)

        return out
