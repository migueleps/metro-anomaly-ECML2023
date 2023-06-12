import os
import numpy as np
import torch as th
from torch import nn, optim
import pickle as pkl
from models.LSTMAE import LSTM_AE
from models.LSTM_SAE import LSTM_SAE
from models.LSTM_SAE_multi_encoder import LSTM_SAE_MultiEncoder
from models.LSTM_AE_multi_encoder import LSTM_AE_MultiEncoder
from models.LSTM_AE_diff_comp import LSTM_AE_MultiComp
from models.LSTM_SAE_diff_comp import LSTM_SAE_MultiComp
from models.TCN_AE import TCN_AE
import tqdm
from ArgumentParser import parse_arguments


def train_model(model,
                train_tensors,
                epochs,
                lr,
                args):

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    loss_over_time = {"train": []}

    for epoch in range(epochs):
        model.train()
        train_losses = []
        with tqdm.tqdm(train_tensors, unit="cycles") as tqdm_epoch:
            for train_tensor in tqdm_epoch:
                tqdm_epoch.set_description(f"Epoch {epoch+1}")
                optimizer.zero_grad()
                loss, _ = model(train_tensor)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                train_losses.append(loss.item())

        train_loss = np.mean(train_losses)

        loss_over_time['train'].append(train_loss)

        print(f'Epoch {epoch+1}: train loss {train_loss}')

    return model, loss_over_time


def predict(model, test_tensors, tqdm_desc):
    test_losses = []
    with th.no_grad():
        model.eval()
        with tqdm.tqdm(test_tensors, unit="cycles") as tqdm_epoch:
            for test_tensor in tqdm_epoch:
                tqdm_epoch.set_description(tqdm_desc)
                loss, _ = model(test_tensor)
                test_losses.append(loss.item())
    return test_losses


def calculate_train_losses(model, args):

    with open(f"{args.data_folder}final_train_tensors_{args.FEATS}.pkl", "rb") as tensor_pkl:
        train_tensors = pkl.load(tensor_pkl)
        train_tensors = [tensor.to(args.device) for tensor in train_tensors]

    train_losses = predict(model, train_tensors, "Calculating training error distribution")

    return train_losses


def offline_train(model, args):

    print(f"Starting offline training")

    with open(f"{args.data_folder}final_train_tensors_{args.FEATS}.pkl", "rb") as tensor_pkl:
        train_tensors = pkl.load(tensor_pkl)
        train_tensors = [tensor.to(args.device) for tensor in train_tensors]

    model, loss_over_time = train_model(model,
                                        train_tensors,
                                        epochs=args.EPOCHS,
                                        lr=args.LR,
                                        args=args)

    train_losses = predict(model, train_tensors, "Calculating training error distribution")

    with open(args.results_string("offline"), "wb") as loss_file:
        pkl.dump(loss_over_time, loss_file)

    th.save(model.state_dict(), args.model_saving_string)

    return model, train_losses


def calculate_test_losses(model, args):

    with open(f"{args.data_folder}final_test_tensors_{args.FEATS}.pkl", "rb") as tensor_pkl:
        test_tensors = pkl.load(tensor_pkl)
        test_tensors = [tensor.to(args.device) for tensor in test_tensors]

    test_losses = predict(model, test_tensors, "Testing on new data")

    losses_over_time = {"test": test_losses, "train": args.train_losses}

    with open(args.results_string("complete"), "wb") as loss_file:
        pkl.dump(losses_over_time, loss_file)

    return model


def load_parameters(arguments):

    FEATS_TO_NUMBER = {"analog_feats": 8, "digital_feats": 8, "all_feats": 16}

    arguments.device = th.device('cuda' if th.cuda.is_available() else 'cpu')

    arguments.FEATS = f"{arguments.FEATS}_feats"
    arguments.NUMBER_FEATURES = FEATS_TO_NUMBER[arguments.FEATS]

    arguments.results_folder = "results/"
    arguments.data_folder = "data/"

    if "tcn" in arguments.MODEL_NAME:
        arguments.model_string = f"{arguments.MODEL_NAME}_{arguments.FEATS}_{arguments.EMBEDDING}_{arguments.tcn_layers}_{arguments.tcn_hidden}_{arguments.tcn_kernel}"
    else:
        arguments.model_string = f"{arguments.MODEL_NAME}_{arguments.FEATS}_{arguments.EMBEDDING}_{arguments.LSTM_LAYERS}"

    print(f"Starting execution of model: {arguments.model_string}")

    arguments.results_string = lambda loop_no: f"{arguments.results_folder}final_{loop_no}_losses_{arguments.model_string}_{arguments.EPOCHS}_{arguments.LR}.pkl"
    arguments.model_saving_string = f"{arguments.results_folder}final_offline_{arguments.model_string}_{arguments.EPOCHS}_{arguments.LR}.pt"

    return arguments


def main(arguments):

    MODELS = {"lstm_ae": LSTM_AE, "lstm_sae": LSTM_SAE, 
              "multi_enc_sae": LSTM_SAE_MultiEncoder, "multi_enc_ae": LSTM_AE_MultiEncoder,
              "diff_comp_sae": LSTM_SAE_MultiComp, "diff_comp_ae": LSTM_AE_MultiComp,
              "tcn_ae": TCN_AE}

    if "tcn" in arguments.MODEL_NAME:
        model = MODELS[arguments.MODEL_NAME](arguments.NUMBER_FEATURES,
                                             arguments.EMBEDDING,
                                             arguments.DROPOUT,
                                             arguments.tcn_layers,
                                             arguments.device,
                                             arguments.tcn_hidden,
                                             arguments.tcn_kernel).to(arguments.device)
    else:
        model = MODELS[arguments.MODEL_NAME](arguments.NUMBER_FEATURES,
                                             arguments.EMBEDDING,
                                             arguments.DROPOUT,
                                             arguments.LSTM_LAYERS,
                                             arguments.device,
                                             arguments.sparsity_weight,
                                             arguments.sparsity_parameter).to(arguments.device)

    if os.path.exists(arguments.model_saving_string) and not arguments.force_training:
        model.load_state_dict(th.load(arguments.model_saving_string))
        arguments.train_losses = calculate_train_losses(model, arguments)
    else:
        model, arguments.train_losses = offline_train(model, arguments)

    calculate_test_losses(model, arguments)


if __name__ == "__main__":
    argument_dict = parse_arguments()
    argument_dict = load_parameters(argument_dict)
    main(argument_dict)
