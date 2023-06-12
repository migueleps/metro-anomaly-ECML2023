import torch as th
from torch import nn
import torch.nn.functional as F


class Encoder(nn.Module):

    def __init__(self,
                 input_dim,
                 embedding_dim,
                 dropout,
                 lstm_layers):
        super(Encoder, self).__init__()

        self.lstm_layers = nn.LSTM(input_size=input_dim,
                                   hidden_size=embedding_dim,
                                   batch_first=True,
                                   num_layers=lstm_layers,
                                   dropout=dropout)

    def forward(self, x):
        hidden_outs, (hidden, _) = self.lstm_layers(x)
        return hidden[-1], hidden_outs


class Decoder(nn.Module):

    def __init__(self,
                 embedding_dim,
                 output_dim,
                 dropout,
                 lstm_layers):
        super(Decoder, self).__init__()

        self.lstm_layers = nn.LSTM(input_size=embedding_dim,
                                   hidden_size=embedding_dim,
                                   batch_first=True,
                                   num_layers=lstm_layers,
                                   dropout=dropout)

        self.output_layer = nn.Linear(in_features=embedding_dim,
                                      out_features=output_dim)

    def forward(self, x):
        x, (_, _) = self.lstm_layers(x)
        return self.output_layer(x)


class LSTM_AE_MultiComp(nn.Module):

    def __init__(self,
                 n_features,
                 emb_dim,
                 dropout,
                 lstm_layers,
                 device,
                 sparsity_weight,
                 sparsity_parameter):
        super(LSTM_AE_MultiComp, self).__init__()

        self.embedding_dim = emb_dim
        self.dropout = dropout
        self.n_features = n_features
        self.device = device
        self.lstm_layers = lstm_layers
        self.sparsity_weight = sparsity_weight
        self.sparsity_parameter = sparsity_parameter

        self.encode_comp0 = Encoder(input_dim=self.n_features,
                                    embedding_dim=self.embedding_dim,
                                    dropout=self.dropout,
                                    lstm_layers=self.lstm_layers).to(device)

        self.encode_comp1 = Encoder(input_dim=self.n_features,
                                    embedding_dim=self.embedding_dim,
                                    dropout=self.dropout,
                                    lstm_layers=self.lstm_layers).to(device)

        self.decode_comp0 = Decoder(embedding_dim=self.embedding_dim,
                                    output_dim=self.n_features,
                                    dropout=self.dropout,
                                    lstm_layers=self.lstm_layers).to(device)

        self.decode_comp1 = Decoder(embedding_dim=self.embedding_dim,
                                    output_dim=self.n_features,
                                    dropout=self.dropout,
                                    lstm_layers=self.lstm_layers).to(device)

    def forward(self, x):
        comp0, comp1 = x

        n_examples_comp0 = comp0.shape[1]
        assert comp0.shape[2] == self.n_features

        n_examples_comp1 = comp1.shape[1]
        assert comp1.shape[2] == self.n_features

        latent_vector0, _ = self.encode_comp0(comp0)
        latent_vector1, _ = self.encode_comp1(comp1)

        stacked_LV0 = th.repeat_interleave(latent_vector0, n_examples_comp0,
                                           dim=1).reshape(-1, n_examples_comp0, self.embedding_dim).to(self.device)
        stacked_LV1 = th.repeat_interleave(latent_vector1, n_examples_comp1,
                                           dim=1).reshape(-1, n_examples_comp1, self.embedding_dim).to(self.device)

        reconstructed_comp0 = self.decode_comp0(stacked_LV0)
        reconstructed_comp1 = self.decode_comp1(stacked_LV1)

        loss = F.mse_loss(reconstructed_comp0, comp0) + F.mse_loss(reconstructed_comp1, comp1)

        return loss, th.cat((reconstructed_comp0, reconstructed_comp1), dim=1)
