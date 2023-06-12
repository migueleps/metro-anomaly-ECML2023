from torch import nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.padding = padding
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.batch_norm1 = nn.BatchNorm1d(n_outputs)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.batch_norm2 = nn.BatchNorm1d(n_outputs)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        res = x if self.downsample is None else self.downsample(x)
        x = self.conv1(x)
        x = x[:, :, :-self.padding].contiguous()
        x = self.dropout(self.batch_norm1(self.relu(x)))
        x = self.conv2(x)
        x = x[:, :, :-self.padding].contiguous()
        out = self.dropout(self.batch_norm2(self.relu(x)))

        return self.relu(out + res)


class Encoder(nn.Module):

    def __init__(self,
                 input_dim,
                 embedding_dim,
                 hidden_dim,
                 dropout,
                 num_layers=8,
                 kernel_size=5):

        super(Encoder, self).__init__()

        channels = num_layers * [hidden_dim]
        TCN_layers = []

        for i in range(num_layers):
            dilation = 2 ** i
            in_channels = input_dim if i == 0 else channels[i - 1]
            out_channels = channels[i]
            padding = (kernel_size - 1) * dilation
            TCN_layers.append(TemporalBlock(in_channels, out_channels, kernel_size, stride=1,
                                            dilation=dilation, padding=padding, dropout=dropout))

        self.TCN_layers = nn.ModuleList(TCN_layers)
        self.output_layer = nn.Linear(in_features=channels[-1],
                                      out_features=embedding_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)

        for layer in self.TCN_layers:
            x = layer(x)

        x = x.permute(0, 2, 1)

        return self.output_layer(x)


class Decoder(nn.Module):

    def __init__(self,
                 embedding_dim,
                 output_dim,
                 hidden_dim,
                 dropout,
                 num_layers=8,
                 kernel_size=5):

        super(Decoder, self).__init__()

        channels = num_layers * [hidden_dim]
        TCN_layers = []

        for i in range(num_layers):
            dilation = 2 ** i
            in_channels = embedding_dim if i == 0 else channels[i - 1]
            out_channels = channels[i]
            padding = (kernel_size - 1) * dilation
            TCN_layers.append(TemporalBlock(in_channels, out_channels, kernel_size, stride=1,
                                            dilation=dilation, padding=padding, dropout=dropout))

        self.TCN_layers = nn.ModuleList(TCN_layers)
        self.output_layer = nn.Linear(in_features=channels[-1],
                                      out_features=output_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)

        for layer in self.TCN_layers:
            x = layer(x)

        x = x.permute(0, 2, 1)

        return self.output_layer(x)


class TCN_AE(nn.Module):

    def __init__(self,
                 n_features,
                 emb_dim,
                 dropout,
                 num_layers,
                 device,
                 hidden_dim,
                 kernel_size):

        super(TCN_AE, self).__init__()

        self.embedding_dim = emb_dim
        self.dropout = dropout
        self.n_features = n_features
        self.device = device
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size

        self.encode = Encoder(input_dim=self.n_features,
                              embedding_dim=self.embedding_dim,
                              hidden_dim=self.hidden_dim,
                              dropout=self.dropout,
                              num_layers=self.num_layers,
                              kernel_size=self.kernel_size).to(device)

        self.decode = Decoder(embedding_dim=self.embedding_dim,
                              output_dim=self.n_features,
                              hidden_dim=self.hidden_dim,
                              dropout=self.dropout,
                              num_layers=self.num_layers,
                              kernel_size=self.kernel_size).to(device)

    def forward(self, x):

        reconstructed_x = self.decode(self.encode(x))

        loss = F.mse_loss(reconstructed_x, x)

        return loss, reconstructed_x
