import os
import numpy as np
import torch as th
from torch import optim
from torch import nn
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal
import pickle as pkl
import tqdm
from ArgumentParser import parse_arguments
from models.LSTM_AAE import Encoder, Decoder, SimpleDiscriminator, LSTMDiscriminator, ConvDiscriminator
from torch.utils.data import Dataset, DataLoader
from models.LSTMAE import LSTM_AE
from models.LSTM_SAE import LSTM_SAE
from models.TCN_AE import TCN_AE
from models.TCN_AAE import Encoder_TCN, Decoder_TCN, SimpleDiscriminator_TCN, LSTMDiscriminator_TCN, ConvDiscriminator_TCN


class ChunkDataset(Dataset):
    def __init__(self, data_location):
        with open(data_location, "rb") as pklfile:
            self.data = pkl.load(pklfile)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, ind):
        return self.data[ind, :, :].float()


####################
#
# Based on the implementation: https://github.com/schelotto/Wasserstein-AutoEncoders
#
####################

#th.autograd.set_detect_anomaly(True)


def free_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = True


def frozen_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = False


def train_discriminator(optimizer_discriminator, multivariate_normal, epoch, args):

    frozen_params(args.encoder)
    frozen_params(args.decoder)
    free_params(args.discriminator)

    losses = []
    with tqdm.tqdm(args.train_dataloader, unit="batches") as tqdm_epoch:
        for train_batch in tqdm_epoch:
            tqdm_epoch.set_description(f"Discriminator Epoch {epoch + 1}")
            optimizer_discriminator.zero_grad()
            train_batch = train_batch.to(args.device)

            real_latent_space = args.encoder(train_batch)

            if len(real_latent_space.shape) == 2:
                real_latent_space = real_latent_space.unsqueeze(1)

            random_latent_space = multivariate_normal.sample(real_latent_space.shape[:2]).to(args.device)

            discriminator_real = args.discriminator(real_latent_space)
            discriminator_random = args.discriminator(random_latent_space)

            loss_random_term = th.log(discriminator_random)
            loss_real_term = th.log(1-discriminator_real)

            loss = args.WAE_regularization_term * -th.mean(loss_real_term + loss_random_term)
            loss.backward()

            nn.utils.clip_grad_norm_(args.discriminator.parameters(), 1)
            optimizer_discriminator.step()
            losses.append(loss.item())

    return losses


def train_reconstruction(optimizer_encoder, optimizer_decoder, epoch, args):

    free_params(args.encoder)
    free_params(args.decoder)
    frozen_params(args.discriminator)

    losses = []
    with tqdm.tqdm(args.train_dataloader, unit="batches") as tqdm_epoch:
        for i, train_batch in enumerate(tqdm_epoch):
            tqdm_epoch.set_description(f"Encoder/Decoder Epoch {epoch + 1}")
            optimizer_encoder.zero_grad()
            optimizer_decoder.zero_grad()
            train_batch = train_batch.to(args.device)

            real_latent_space = args.encoder(train_batch)
            if len(real_latent_space.shape) == 2:
                real_latent_space = real_latent_space.unsqueeze(1)

            discriminator_real_latent = args.discriminator(real_latent_space)

            if "TCN" not in args.MODEL_NAME:
                real_latent_space = real_latent_space.repeat(1, train_batch.shape[1], 1).to(args.device)

            reconstructed_input = args.decoder(real_latent_space)
            reconstruction_loss = F.mse_loss(reconstructed_input, train_batch,
                                             reduction="none").mean(dim=(1, 2)).reshape(-1, 1)

            discriminator_loss = args.WAE_regularization_term * (th.log(discriminator_real_latent))

            loss = th.mean(reconstruction_loss - discriminator_loss)

            loss.backward()

            nn.utils.clip_grad_norm_(args.encoder.parameters(), 1)
            nn.utils.clip_grad_norm_(args.decoder.parameters(), 1)

            optimizer_encoder.step()
            optimizer_decoder.step()
            losses.append(loss.item())

    return losses


def train_non_gan(optimizer, epoch, args):

    losses = []
    with tqdm.tqdm(args.train_dataloader, unit="batches") as tqdm_epoch:
        for i, train_batch in enumerate(tqdm_epoch):
            tqdm_epoch.set_description(f"Autoencoder Epoch {epoch + 1}")
            optimizer.zero_grad()
            train_batch = train_batch.to(args.device)
            loss, _ = args.model(train_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(args.model.parameters(), 1)
            optimizer.step()
            losses.append(loss.item())

    return losses


def train_model(epochs,
                args):

    if args.use_discriminator:
        optimizer_discriminator = optim.Adam(args.discriminator.parameters(), lr=args.disc_lr)
        optimizer_encoder = optim.Adam(args.encoder.parameters(), lr=args.LR)
        optimizer_decoder = optim.Adam(args.decoder.parameters(), lr=args.LR)

    else:
        optimizer_autoencoder = optim.Adam(args.model.parameters(), lr=args.LR)

    loss_over_time = {"discriminator": [], "encoder/decoder": []}

    multivariate_normal = MultivariateNormal(th.zeros(args.EMBEDDING), th.eye(args.EMBEDDING))

    for epoch in range(epochs):

        if args.use_discriminator:
            discriminator_losses = train_discriminator(optimizer_discriminator,
                                                       multivariate_normal, epoch, args)
            loss_over_time['discriminator'].append(np.mean(discriminator_losses))

            encoder_decoder_losses = train_reconstruction(optimizer_encoder, optimizer_decoder, epoch, args)
            loss_over_time['encoder/decoder'].append(np.mean(encoder_decoder_losses))
            print(f'Epoch {epoch + 1}: discriminator loss {np.mean(discriminator_losses)} encoder/decoder loss {np.mean(encoder_decoder_losses)}')

        else:
            autoencoder_losses = train_non_gan(optimizer_autoencoder, epoch, args)
            loss_over_time['encoder/decoder'].append(np.mean(autoencoder_losses))
            print(f'Epoch {epoch + 1}: encoder/decoder loss {np.mean(autoencoder_losses)}')

    return loss_over_time


def predict_gan(args, test_dataloader, tqdm_desc):
    reconstruction_errors = []
    critic_scores = []
    with th.no_grad():
        args.encoder.eval()
        args.decoder.eval()
        args.discriminator.eval()
        with tqdm.tqdm(test_dataloader, unit="cycles") as tqdm_epoch:
            for test_batch in tqdm_epoch:
                tqdm_epoch.set_description(tqdm_desc)
                test_batch = test_batch.to(args.device)

                latent_space = args.encoder(test_batch)
                if len(latent_space.shape) == 2:
                    latent_space = latent_space.unsqueeze(1)

                critic_score = th.mean(args.discriminator(latent_space))
                critic_scores.append(critic_score.item())

                if "TCN" not in args.MODEL_NAME:
                    latent_space = latent_space.repeat(1, test_batch.shape[1], 1).to(args.device)

                reconstruction = args.decoder(latent_space)
                reconstruction_errors.append(F.mse_loss(reconstruction, test_batch).item())

    return reconstruction_errors, critic_scores


def predict_non_gan(args, test_dataloader, tqdm_desc):
    test_losses = []
    with th.no_grad():
        args.model.eval()
        with tqdm.tqdm(test_dataloader, unit="cycles") as tqdm_epoch:
            for test_batch in tqdm_epoch:
                tqdm_epoch.set_description(tqdm_desc)
                test_batch = test_batch.to(args.device)
                loss, _ = args.model(test_batch)
                test_losses.append(loss.item())

    return test_losses


def offline_train(args):
    print(f"Starting offline training")

    loss_over_time = train_model(epochs=args.EPOCHS,
                                 args=args)

    if args.use_discriminator:
        results_string = args.results_string("offline", "WAE")
        th.save(args.decoder.state_dict(), args.model_saving_string("WAE_decoder"))
        th.save(args.encoder.state_dict(), args.model_saving_string("WAE_encoder"))
        th.save(args.discriminator.state_dict(), args.model_saving_string("WAE_discriminator"))
    else:
        results_string = args.results_string("offline", "AE")
        th.save(args.model.state_dict(), args.model_saving_string("AE"))

    with open(results_string, "wb") as loss_file:
        pkl.dump(loss_over_time, loss_file)

    return


def calculate_train_losses(args):

    if args.use_discriminator:
        reconstruction_error, critic_scores = predict_gan(args, args.train_scores,
                                                          "Calculating training error distribution")
        args.train_reconstruction_errors = reconstruction_error
        args.train_critic_scores = critic_scores
    else:
        reconstruction_error = predict_non_gan(args, args.train_scores, "Calculating training error distribution")
        args.train_reconstruction_errors = reconstruction_error

    return


def calculate_test_losses(args):

    if args.use_discriminator:
        results_string = args.results_string("complete", "WAE")
        reconstruction_errors, critic_scores = predict_gan(args, args.test_dataloader, "Testing on new data")

        results = {"test": {"reconstruction": reconstruction_errors,
                            "critic": critic_scores},
                   "train": {"reconstruction": args.train_reconstruction_errors,
                             "critic": args.train_critic_scores}}

    else:
        results_string = args.results_string("complete", "AE")
        reconstruction_errors = predict_non_gan(args, args.test_dataloader, "Testing on new data")

        results = {"test": reconstruction_errors,
                   "train": args.train_reconstruction_errors}

    with open(results_string, "wb") as loss_file:
        pkl.dump(results, loss_file)

    return


def load_parameters(arguments):

    FEATS_TO_NUMBER = {"analog_feats": 8, "digital_feats": 8, "all_feats": 16, "noflow_feats": 7}

    arguments.device = th.device('cuda' if th.cuda.is_available() else 'cpu')

    arguments.FEATS = f"{arguments.FEATS}_feats"
    arguments.NUMBER_FEATURES = FEATS_TO_NUMBER[arguments.FEATS]

    arguments.results_folder = "results/"
    arguments.data_folder = "data/"

    if arguments.FEATS == "noflow_feats":
        train_set = ChunkDataset("data/training_chunks_noflow.pkl")
        test_set = ChunkDataset("data/test_chunks_noflow.pkl")
    else:
        train_set = ChunkDataset("data/training_chunks.pkl")
        test_set = ChunkDataset("data/test_chunks.pkl")

    arguments.train_dataloader = DataLoader(train_set, batch_size=arguments.BATCH_SIZE, shuffle=True)
    arguments.train_scores = DataLoader(train_set, batch_size=1, shuffle=False)
    arguments.test_dataloader = DataLoader(test_set, batch_size=1, shuffle=False)

    if arguments.use_discriminator:
        first_part = f"{arguments.MODEL_NAME}_{arguments.FEATS}_{arguments.EMBEDDING}_{arguments.LSTM_LAYERS}"
        last_part = f"{arguments.WAE_regularization_term}_{arguments.disc_layers}_{arguments.disc_hidden}"
        if arguments.DECODER_NAME == "TCN":
            model_specific_part = f"_{arguments.tcn_hidden}_{arguments.tcn_kernel}"
        else:
            model_specific_part = ""
        arguments.model_string = lambda model: f"{model}_{first_part}{model_specific_part}_{last_part}"

    elif "tcn" in arguments.MODEL_NAME:
        arguments.model_string = lambda model: f"{model}_{arguments.MODEL_NAME}_{arguments.FEATS}_{arguments.EMBEDDING}_{arguments.tcn_layers}_{arguments.tcn_hidden}_{arguments.tcn_kernel}"
    else:
        arguments.model_string = lambda model: f"{model}_{arguments.MODEL_NAME}_{arguments.FEATS}_{arguments.EMBEDDING}_{arguments.LSTM_LAYERS}"

    if arguments.use_discriminator:
        print(f"Starting execution of model: {arguments.model_string('WAE')}")
    else:
        print(f"Starting execution of model: {arguments.model_string('AE')}")

    arguments.results_string = lambda loop_no, model_label: f"{arguments.results_folder}final_chunks_{loop_no}_losses_{arguments.model_string(model_label)}_{arguments.EPOCHS}_{arguments.LR}_{arguments.disc_lr}_{arguments.BATCH_SIZE}.pkl"
    arguments.model_saving_string = lambda model: f"{arguments.results_folder}final_chunks_offline_{arguments.model_string(model)}_{arguments.EPOCHS}_{arguments.LR}_{arguments.disc_lr}_{arguments.BATCH_SIZE}.pt"

    if arguments.use_discriminator:

        encoders = dict(LSTM=Encoder,
                        TCN=Encoder_TCN)

        decoders = dict(LSTM=Decoder,
                        TCN=Decoder_TCN)

        arguments.decoder = decoders[arguments.DECODER_NAME](arguments.EMBEDDING,
                                                             arguments.NUMBER_FEATURES,
                                                             arguments.DROPOUT,
                                                             arguments.LSTM_LAYERS,
                                                             hidden_dim=arguments.tcn_hidden,
                                                             kernel_size=arguments.tcn_kernel).to(arguments.device)

        arguments.encoder = encoders[arguments.ENCODER_NAME](arguments.NUMBER_FEATURES,
                                                             arguments.EMBEDDING,
                                                             arguments.DROPOUT,
                                                             arguments.LSTM_LAYERS,
                                                             hidden_dim=arguments.tcn_hidden,
                                                             kernel_size=arguments.tcn_kernel).to(arguments.device)

        models = dict(SimpleDiscriminator=SimpleDiscriminator,
                      LSTMDiscriminator=LSTMDiscriminator,
                      ConvDiscriminator=ConvDiscriminator,
                      SimpleDiscriminator_TCN=SimpleDiscriminator_TCN,
                      LSTMDiscriminator_TCN=LSTMDiscriminator_TCN,
                      ConvDiscriminator_TCN=ConvDiscriminator_TCN)

        arguments.discriminator = models[arguments.MODEL_NAME](arguments.EMBEDDING,
                                                               arguments.DROPOUT,
                                                               n_layers=arguments.disc_layers,
                                                               disc_hidden=arguments.disc_hidden,
                                                               kernel_size=arguments.tcn_kernel,
                                                               window_size=1800).to(arguments.device)

    else:
        MODELS = {"lstm_ae": LSTM_AE, "lstm_sae": LSTM_SAE, "tcn_ae": TCN_AE}

        if "tcn" in arguments.MODEL_NAME:
            arguments.model = MODELS[arguments.MODEL_NAME](arguments.NUMBER_FEATURES,
                                                           arguments.EMBEDDING,
                                                           arguments.DROPOUT,
                                                           arguments.tcn_layers,
                                                           arguments.device,
                                                           arguments.tcn_hidden,
                                                           arguments.tcn_kernel).to(arguments.device)
        else:
            arguments.model = MODELS[arguments.MODEL_NAME](arguments.NUMBER_FEATURES,
                                                           arguments.EMBEDDING,
                                                           arguments.DROPOUT,
                                                           arguments.LSTM_LAYERS,
                                                           arguments.device,
                                                           arguments.sparsity_weight,
                                                           arguments.sparsity_parameter).to(arguments.device)

    return arguments


def main(arguments):

    if arguments.use_discriminator:
        models_exist = all([os.path.exists(arguments.model_saving_string(model)) for model in ["WAE_encoder",
                                                                                               "WAE_decoder",
                                                                                               "WAE_discriminator"]])
    else:
        models_exist = os.path.exists(arguments.model_saving_string("AE"))

    if models_exist and not arguments.force_training:
        if arguments.use_discriminator:
            arguments.decoder.load_state_dict(th.load(arguments.model_saving_string("WAE_decoder")))
            arguments.encoder.load_state_dict(th.load(arguments.model_saving_string("WAE_encoder")))
            arguments.discriminator.load_state_dict(th.load(arguments.model_saving_string("WAE_discriminator")))
        else:
            arguments.model.load_state_dict(th.load(arguments.model_saving_string("AE")))

    else:
        offline_train(arguments)

    calculate_train_losses(arguments)
    calculate_test_losses(arguments)


if __name__ == "__main__":
    argument_dict = parse_arguments()
    argument_dict = load_parameters(argument_dict)
    main(argument_dict)
