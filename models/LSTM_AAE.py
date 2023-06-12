from torch import nn
import torch as th
from torch.nn.utils import weight_norm


class Encoder(nn.Module):

    def __init__(self,
                 input_dim,
                 embedding_dim,
                 dropout,
                 lstm_layers, **kwargs):
        super(Encoder, self).__init__()

        self.lstm_layers = nn.LSTM(input_size=input_dim,
                                   hidden_size=embedding_dim,
                                   batch_first=True,
                                   num_layers=lstm_layers,
                                   dropout=dropout)

    def forward(self, x):
        hidden_outs, (hidden, _) = self.lstm_layers(x)
        return hidden[-1]


class Decoder(nn.Module):

    def __init__(self,
                 embedding_dim,
                 output_dim,
                 dropout,
                 lstm_layers, **kwargs):
        super(Decoder, self).__init__()

        self.lstm_layers = nn.LSTM(input_size=embedding_dim,
                                   hidden_size=embedding_dim,
                                   batch_first=True,
                                   num_layers=lstm_layers,
                                   bidirectional=True,
                                   dropout=dropout)

        self.output_layer = nn.Linear(in_features=2 * embedding_dim,
                                      out_features=output_dim)

    def forward(self, x):
        x, (_, _) = self.lstm_layers(x)
        return self.output_layer(x)


class SimpleDiscriminator(nn.Module):

    def __init__(self, input_dim, dropout, **kwargs):
        super(SimpleDiscriminator, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.activation_func = nn.LeakyReLU()

        n_layers = kwargs["n_layers"]
        first_layer_hidden_dim = kwargs["disc_hidden"]

        inputs = [input_dim] + [int(first_layer_hidden_dim/2**i) for i in range(n_layers-1)]
        outputs = [int(first_layer_hidden_dim/2**i) for i in range(n_layers)]

        self.fcs = nn.Sequential(*[nn.Sequential(nn.Linear(in_features=inputs[i],
                                                           out_features=outputs[i]),
                                                 self.activation_func,
                                                 self.dropout) for i in range(n_layers)])

        self.output_layer = nn.Linear(in_features=outputs[-1],
                                      out_features=1)

    def forward(self, x):
        x = self.fcs(x)
        return th.sigmoid(self.output_layer(x))


class LSTMDiscriminator(nn.Module):

    def __init__(self, input_dim, dropout, **kwargs):
        super(LSTMDiscriminator, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim,
                            hidden_size=kwargs["disc_hidden"],
                            dropout=dropout,
                            batch_first=True,
                            num_layers=kwargs["n_layers"])

        self.output_layer = nn.Linear(in_features=kwargs["disc_hidden"],
                                      out_features=1)

    def forward(self, x):
        x, _ = self.lstm(x)
        return th.sigmoid(self.output_layer(x))


######################################################################################
#
#    TCN from https://github.com/locuslab/TCN/
#
######################################################################################


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


class ConvDiscriminator(nn.Module):

    def __init__(self, input_dim, dropout, **kwargs):
        super(ConvDiscriminator, self).__init__()
        # use same default parameters as TCN repo

        hidden_dim = kwargs["disc_hidden"]
        num_layers = kwargs["n_layers"]
        channels = num_layers * [hidden_dim]
        TCN_layers = []
        kernel_size = kwargs["kernel_size"]

        for i in range(num_layers):
            dilation = 2 ** i
            in_channels = input_dim if i == 0 else channels[i - 1]
            out_channels = channels[i]
            padding = (kernel_size - 1) * dilation
            TCN_layers.append(TemporalBlock(in_channels, out_channels, kernel_size, stride=1,
                                            dilation=dilation, padding=padding, dropout=dropout))

        self.TCN_layers = nn.ModuleList(TCN_layers)
        self.output_layer = nn.Linear(in_features=channels[-1],
                                      out_features=1)

    def forward(self, x):
        x = x.permute(0, 2, 1)

        for layer in self.TCN_layers:
            x = layer(x)

        x = x.permute(0, 2, 1)
        x = self.output_layer(x)

        return th.sigmoid(x)
