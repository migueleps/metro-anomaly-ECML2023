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

####################
#
# Based on the implementation: https://github.com/schelotto/Wasserstein-AutoEncoders
#
####################


def free_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = True


def frozen_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = False


def train_discriminator(optimizer_discriminator, train_tensors, multivariate_normal, epoch, args):

    frozen_params(args.encoder)
    frozen_params(args.decoder)
    free_params(args.discriminator)

    losses = []
    with tqdm.tqdm(train_tensors, unit="cycles") as tqdm_epoch:
        for train_tensor in tqdm_epoch:
            tqdm_epoch.set_description(f"Discriminator Epoch {epoch + 1}")
            optimizer_discriminator.zero_grad()

            real_latent_space = args.encoder(train_tensor)

            random_latent_space = multivariate_normal.sample(real_latent_space.shape[:-1]).to(args.device)

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


def train_reconstruction(optimizer_encoder, optimizer_decoder, train_tensors, epoch, args):

    free_params(args.encoder)
    free_params(args.decoder)
    frozen_params(args.discriminator)

    losses = []
    with tqdm.tqdm(train_tensors, unit="cycles") as tqdm_epoch:
        for i, train_tensor in enumerate(tqdm_epoch):
            tqdm_epoch.set_description(f"Encoder/Decoder Epoch {epoch + 1}")
            optimizer_encoder.zero_grad()
            optimizer_decoder.zero_grad()

            real_latent_space = args.encoder(train_tensor)
            stacked_LV = th.repeat_interleave(real_latent_space,
                                              train_tensor.shape[1],
                                              dim=1).reshape(-1,
                                                             train_tensor.shape[1],
                                                             real_latent_space.shape[-1]).to(args.device)

            reconstructed_input = args.decoder(stacked_LV)
            discriminator_real_latent = args.discriminator(real_latent_space)

            reconstruction_loss = F.mse_loss(reconstructed_input, train_tensor)
            discriminator_loss = args.WAE_regularization_term * (th.log(discriminator_real_latent))

            loss = th.mean(reconstruction_loss - discriminator_loss)
            loss.backward()

            nn.utils.clip_grad_norm_(args.encoder.parameters(), 1)
            nn.utils.clip_grad_norm_(args.decoder.parameters(), 1)

            optimizer_encoder.step()
            optimizer_decoder.step()
            losses.append(loss.item())

    return losses


def train_model(train_tensors,
                epochs,
                args):

    optimizer_discriminator = optim.Adam(args.discriminator.parameters(), lr=args.disc_lr)
    optimizer_encoder = optim.Adam(args.encoder.parameters(), lr=args.LR)
    optimizer_decoder = optim.Adam(args.decoder.parameters(), lr=args.LR)

    loss_over_time = {"discriminator": [], "encoder/decoder": []}

    multivariate_normal = MultivariateNormal(th.zeros(args.EMBEDDING), th.eye(args.EMBEDDING))

    for epoch in range(epochs):

        discriminator_losses = train_discriminator(optimizer_discriminator, train_tensors,
                                                   multivariate_normal, epoch, args)
        encoder_decoder_losses = train_reconstruction(optimizer_encoder, optimizer_decoder, train_tensors, epoch, args)

        loss_over_time['discriminator'].append(np.mean(discriminator_losses))
        loss_over_time['encoder/decoder'].append(np.mean(encoder_decoder_losses))

        print(f'Epoch {epoch + 1}: discriminator loss {np.mean(discriminator_losses)} encoder/decoder loss {np.mean(encoder_decoder_losses)}')

    return loss_over_time


def predict(args, test_tensors, tqdm_desc):
    reconstruction_errors = []
    critic_scores = []
    with th.no_grad():
        args.encoder.eval()
        args.decoder.eval()
        args.discriminator.eval()
        with tqdm.tqdm(test_tensors, unit="cycles") as tqdm_epoch:
            for test_tensor in tqdm_epoch:
                tqdm_epoch.set_description(tqdm_desc)
                test_tensor = test_tensor.to(args.device)
                latent_vector = args.encoder(test_tensor)

                stacked_LV = th.repeat_interleave(latent_vector,
                                                  test_tensor.shape[1],
                                                  dim=1).reshape(-1,
                                                                 test_tensor.shape[1],
                                                                 latent_vector.shape[-1]).to(args.device)

                reconstruction = args.decoder(stacked_LV)
                reconstruction_errors.append(F.mse_loss(reconstruction, test_tensor).item())
                critic_score = th.mean(args.discriminator(latent_vector))
                critic_scores.append(critic_score.item())

    return reconstruction_errors, critic_scores


def offline_train(args):
    print(f"Starting offline training")

    with open(f"{args.data_folder}final_train_tensors_{args.FEATS}.pkl", "rb") as tensor_pkl:
        train_tensors = pkl.load(tensor_pkl)
        train_tensors = [tensor.to(args.device) for tensor in train_tensors]

    loss_over_time = train_model(train_tensors,
                                 epochs=args.EPOCHS,
                                 args=args)

    with open(args.results_string("offline"), "wb") as loss_file:
        pkl.dump(loss_over_time, loss_file)

    th.save(args.decoder.state_dict(), args.model_saving_string("WAE_decoder"))
    th.save(args.encoder.state_dict(), args.model_saving_string("WAE_encoder"))
    th.save(args.discriminator.state_dict(), args.model_saving_string("WAE_discriminator"))

    return


def calculate_train_losses(args):

    with open(f"{args.data_folder}final_train_tensors_{args.FEATS}.pkl", "rb") as tensor_pkl:
        train_tensors = pkl.load(tensor_pkl)
        train_tensors = [tensor.to(args.device) for tensor in train_tensors]

    reconstruction_error, critic_scores = predict(args, train_tensors, "Calculating training error distribution")
    args.train_reconstruction_errors = reconstruction_error
    args.train_critic_scores = critic_scores
    return


def calculate_test_losses(args):

    with open(f"{args.data_folder}final_test_tensors_{args.FEATS}.pkl", "rb") as tensor_pkl:
        test_tensors = pkl.load(tensor_pkl)
        test_tensors = [tensor.to(args.device) for tensor in test_tensors]

    reconstruction_errors, critic_scores = predict(args, test_tensors, "Testing on new data")

    results = {"test": {"reconstruction": reconstruction_errors,
                        "critic": critic_scores},
               "train": {"reconstruction": args.train_reconstruction_errors,
                         "critic": args.train_critic_scores}}

    with open(args.results_string("complete"), "wb") as loss_file:
        pkl.dump(results, loss_file)

    return


def load_parameters(arguments):
    FEATS_TO_NUMBER = {"analog_feats": 8, "digital_feats": 8, "all_feats": 16}

    arguments.device = th.device('cuda' if th.cuda.is_available() else 'cpu')

    arguments.FEATS = f"{arguments.FEATS}_feats"
    arguments.NUMBER_FEATURES = FEATS_TO_NUMBER[arguments.FEATS]

    arguments.results_folder = "results/"
    arguments.data_folder = "data/"

    arguments.model_string = lambda model: f"{model}_{arguments.MODEL_NAME}_{arguments.FEATS}_{arguments.EMBEDDING}_{arguments.LSTM_LAYERS}_{arguments.WAE_regularization_term}"

    print(f"Starting execution of model: {arguments.model_string('WAE')}")

    arguments.results_string = lambda loop_no: f"{arguments.results_folder}final_{loop_no}_losses_{arguments.model_string('WAE')}_{arguments.EPOCHS}_{arguments.LR}_{arguments.disc_lr}.pkl"
    arguments.model_saving_string = lambda model: f"{arguments.results_folder}final_offline_{arguments.model_string(model)}_{arguments.EPOCHS}_{arguments.LR}_{arguments.disc_lr}.pt"

    arguments.decoder = Decoder(arguments.EMBEDDING,
                                arguments.NUMBER_FEATURES,
                                arguments.DROPOUT,
                                arguments.LSTM_LAYERS).to(arguments.device)

    arguments.encoder = Encoder(arguments.NUMBER_FEATURES,
                                arguments.EMBEDDING,
                                arguments.DROPOUT,
                                arguments.LSTM_LAYERS).to(arguments.device)

    models = dict(SimpleDiscriminator=SimpleDiscriminator,
                  LSTMDiscriminator=LSTMDiscriminator,
                  ConvDiscriminator=ConvDiscriminator)

    arguments.discriminator = models[arguments.MODEL_NAME](arguments.EMBEDDING,
                                                           arguments.DROPOUT,
                                                           n_layers=arguments.LSTM_LAYERS,
                                                           disc_hidden=arguments.disc_hidden,
                                                           kernel_size=arguments.tcn_kernel).to(arguments.device)

    return arguments


def main(arguments):
    if all([os.path.exists(arguments.model_saving_string(model)) for model in ["WAE_encoder",
                                                                               "WAE_decoder",
                                                                               "WAE_discriminator"]]) \
            and not arguments.force_training:

        arguments.decoder.load_state_dict(th.load(arguments.model_saving_string("WAE_decoder")))
        arguments.encoder.load_state_dict(th.load(arguments.model_saving_string("WAE_encoder")))
        arguments.discriminator.load_state_dict(th.load(arguments.model_saving_string("WAE_discriminator")))
    else:
        offline_train(arguments)

    calculate_train_losses(arguments)
    calculate_test_losses(arguments)


if __name__ == "__main__":
    argument_dict = parse_arguments()
    argument_dict = load_parameters(argument_dict)
    main(argument_dict)
