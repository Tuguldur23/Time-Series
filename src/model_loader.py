import torch
from LSTM import LSTMModel
from RNN import RNNModel
from GRU import GRUModel


def build_model(model_name, config):
    if model_name == 'lstm':
        model = LSTMModel(input_dim=config['model_params']['input_dim'],
                          hidden_dim=config['model_params']['hidden_dim'],
                          layer_dim=config['model_params']['layer_dim'],
                          output_dim=config['model_params']['output_dim'],
                          dropout_prob=config['model_params']['dropout_prob'])
        model.load_state_dict(torch.load(config['models'][model_name]))
        return model
    elif model_name == 'rnn':
        model = RNNModel(input_dim=config['model_params']['input_dim'], hidden_dim=config['model_params']['hidden_dim'],
                         layer_dim=config['model_params']['layer_dim'],
                         output_dim=config['model_params']['output_dim'],
                         dropout_prob=config['model_params']['dropout_prob'])
        model.load_state_dict(torch.load(config['models'][model_name]))
        return model
    elif model_name == 'gru':
        model = GRUModel(input_dim=config['model_params']['input_dim'], hidden_dim=config['model_params']['hidden_dim'],
                         layer_dim=config['model_params']['layer_dim'],
                         output_dim=config['model_params']['output_dim'],
                         dropout_prob=config['model_params']['dropout_prob'])
        model.load_state_dict(torch.load(config['models'][model_name]))
        return model
    else:
        print("Model name is not correct!!!")
