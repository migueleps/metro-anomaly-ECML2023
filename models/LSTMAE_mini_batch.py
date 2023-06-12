import torch as th
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F


class Encoder(nn.Module):

    def __init__(self, input_dim, embedding_dim, dropout, lstm_layers):
        super(Encoder, self).__init__()

        self.input_dim = input_dim
        self.emb_dim = embedding_dim

        self.lstm = nn.LSTM(input_size=self.input_dim,
                            hidden_size=self.emb_dim,
                            batch_first=True,
                            num_layers=lstm_layers,
                            dropout=dropout)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        return hidden[-1]


class Decoder(nn.Module):

    def __init__(self, embedding_dim, dropout, lstm_layers):
        super(Decoder, self).__init__()

        self.emb_dim = embedding_dim

        self.lstm = nn.LSTM(input_size=self.emb_dim,
                            hidden_size=self.emb_dim,
                            batch_first=True,
                            num_layers=lstm_layers,
                            dropout=dropout)

    def forward(self, x):
        x, (_, _) = self.lstm(x)
        return x


class LSTM_AE(nn.Module):

    def __init__(self, n_features, emb_dim, dropout, lstm_layers, device, *args):
        super(LSTM_AE, self).__init__()

        self.embedding_dim = emb_dim
        self.dropout = dropout
        self.n_features = n_features
        self.device = device

        self.encode = Encoder(input_dim=n_features,
                              embedding_dim=self.embedding_dim,
                              dropout=self.dropout,
                              lstm_layers=lstm_layers).to(device)

        self.decode = Decoder(embedding_dim=self.embedding_dim,
                              dropout=self.dropout,
                              lstm_layers=lstm_layers).to(device)

        self.output_layer = nn.Linear(in_features=self.embedding_dim,
                                      out_features=self.n_features)

    @staticmethod
    def compute_loss(reconstruction, original):
        return F.mse_loss(reconstruction, original)

    def forward(self, x):

        unpacked_original, seq_lengths = pad_packed_sequence(x, batch_first=True)

        latent_vector = self.encode(x)
        max_seq_length = max(seq_lengths)

        new_mini_batch = []
        for i, tensor in enumerate(latent_vector):
            padded_tensor = th.cat([tensor.repeat(seq_lengths[i], 1),
                                    th.zeros(max_seq_length - seq_lengths[i], self.embedding_dim).to(self.device)])
            new_mini_batch.append(padded_tensor)

        mini_batch_LV = th.stack(new_mini_batch).to(self.device)

        packed_LV = pack_padded_sequence(mini_batch_LV, seq_lengths, batch_first=True, enforce_sorted=False)

        decoded_x = self.decode(packed_LV)

        unpacked_decoded, _ = pad_packed_sequence(decoded_x, batch_first=True)

        loss = 0
        reconstructions = []
        for i, tensor in enumerate(unpacked_decoded):
            linear_out = self.output_layer(tensor[:seq_lengths[i]])
            loss += self.compute_loss(linear_out, unpacked_original[i][:seq_lengths[i]])
            reconstructions.append(linear_out)

        return loss / seq_lengths.shape[0], reconstructions
