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
                 lstm_layers,
                 device=th.device("cuda")):
        super(Decoder, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.embedding_dim = embedding_dim
        self.device = device
        self.n_layers = lstm_layers
        self.lstm_layers = nn.LSTM(input_size=output_dim,
                                   hidden_size=embedding_dim,
                                   dropout=dropout,
                                   batch_first=True,
                                   num_layers=lstm_layers)

        self.output_layer = nn.Linear(in_features=embedding_dim,
                                      out_features=output_dim)

    def init_hidden(self, latent_space):
        hidden_states = th.stack([latent_space for _ in range(self.n_layers)]).to(self.device)
        cell_states = th.zeros(self.n_layers, self.embedding_dim, device=self.device)
        return hidden_states, cell_states.unsqueeze(1)

    def forward(self, latent_space, number_outputs):

        hidden_states, cell_states = self.init_hidden(latent_space)
        iter_input = self.output_layer(latent_space).unsqueeze(0)
        output = [iter_input]

        for _ in range(number_outputs-1):
            lstm_outs, (hidden_states, cell_states) = self.lstm_layers(iter_input, (hidden_states, cell_states))
            new_output = self.output_layer(lstm_outs)
            output.append(new_output)
            iter_input = new_output

        output = th.cat(output, dim=1)
        return th.flip(output, [0])


class LSTM_SAE_MultiEncoder(nn.Module):

    def __init__(self,
                 n_features,
                 emb_dim,
                 dropout,
                 lstm_layers,
                 device,
                 sparsity_weight,
                 sparsity_parameter):

        super(LSTM_SAE_MultiEncoder, self).__init__()

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

        self.decode = Decoder(embedding_dim=2*self.embedding_dim,
                              output_dim=self.n_features,
                              dropout=self.dropout,
                              lstm_layers=self.lstm_layers).to(device)

    def sparsity_penalty(self, activations):
        average_activation = th.mean(th.abs(activations), 1)
        target_activations = th.tensor([self.sparsity_parameter] * average_activation.shape[0]).to(self.device)
        kl_div_part1 = th.log(target_activations/average_activation)
        kl_div_part2 = th.log((1-target_activations)/(1-average_activation))
        return th.sum(self.sparsity_parameter * kl_div_part1 + (1-self.sparsity_parameter) * kl_div_part2)

    def forward(self, x):

        comp0, comp1 = x

        n_examples_comp0 = comp0.shape[1]
        assert comp0.shape[2] == self.n_features

        n_examples_comp1 = comp1.shape[1]
        assert comp1.shape[2] == self.n_features

        total_n_examples = n_examples_comp0 + n_examples_comp1

        latent_vector0, hidden_outs0 = self.encode_comp0(comp0)
        latent_vector1, hidden_outs1 = self.encode_comp1(comp1)

        latent_vector = th.cat((latent_vector0, latent_vector1), dim=1)

        reconstructed_x = self.decode(latent_vector, total_n_examples)

        original_cycle = th.cat((comp0, comp1), dim=1)
        loss = F.mse_loss(reconstructed_x, original_cycle)
        if self.training:
            loss += self.sparsity_weight * (self.sparsity_penalty(hidden_outs0) + self.sparsity_penalty(hidden_outs1))

        return loss, reconstructed_x
