import argparse
import numpy as np
import torch
from model.main import train_model


torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False

init_seed = 42
torch.manual_seed(init_seed)
torch.cuda.manual_seed(init_seed)
np.random.seed(init_seed)


def main(args):

    net = train_model(args)
    net.training()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FRL')

    # BASIC SET UP
    parser.add_argument('--dataset', default='adult', type=str, help='dataset name')
    parser.add_argument('--seed', default=64, type=int, help='initial seed')
    parser.add_argument('--epochs', default=20, type=int, help='epochs for training')
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('--theta_dict', \
                        default = {'theta_0_p':0.4,'theta_0_m':0,'theta_1_p':0,'theta_1_m':0.4},\
                        type=dict, help="noisy rate")
    # MODEL
    parser.add_argument('--z_dim', default=15, type=int, help='dimension of the representation z')
    parser.add_argument('--b_dim', default=1, type=int, help='dimension of the representation b')
    parser.add_argument('--z_enc_dim', default=15, type=int, help='dimension of encoder for z')
    parser.add_argument('--x_dec_dim', default=15, type=int, help='dimension of the decoder for x')
    parser.add_argument('--y_dim', default=1, type=int, help='dimension of the y')
    parser.add_argument('--hidden_layer', default=15, type=int, help='hidden layer of the perceptron')

    # OPTIMIZATION
    parser.add_argument('--lr_VAE', default=5e-3, type=float, help='learning rate of the VAE')
    parser.add_argument('--lr_D', default=5e-4, type=float, help='learning rate of the discriminator')
    parser.add_argument('--dropout', default=0.25, type=float, help='dropout rate')  
    
    # HYPERPARAMETERS
    parser.add_argument('--gamma', default=6.4, type=float, help='gamma hyperparameter')
    parser.add_argument('--alpha', default=0.5, type=float, help='alpha hyperparameter')
    parser.add_argument('--beta', default=0.1, type=float, help='beta hyperparameter')
    parser.add_argument('--xi', default=0.1, type=float, help='xi hyperparameter')
    parser.add_argument('--temprature', default=0.9, type=float, help='temprature hyperparameter')

    args = parser.parse_args()

    main(args)