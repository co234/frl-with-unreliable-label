import numpy as np
from model.mlp_vae import *
import pickle
from model.vae import VAE, PredNet
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.nn import Softmax
import torch.optim as optim
from model.discriminator import Discriminator
import torch.nn.functional as F
from model.utils import *


class train_model(object):
    def __init__(self, args):
        use_cuda = torch.cuda.is_available()
        self.device = 'cuda' if use_cuda else 'cpu'
        self.seed = args.seed
        self.epochs = args.epochs

        # Data
        self.dataset = args.dataset
        self.batch_size = args.batch_size
        self.theta_dict = args.theta_dict
        self.input_data = pickle.load(open('dataset/{}/train.pkl'.format(self.dataset),'rb'))
        self.target_data = pickle.load(open('dataset/{}/test.pkl'.format(self.dataset),'rb'))

        # Networks & Optimizers
        self.z_dim = args.z_dim
        self.b_dim = args.b_dim
        self.s_dim = self.input_data['s'].shape[1]
        self.x_dim = self.input_data['x'].shape[1]
        self.y_dim = args.y_dim
        self.z_enc_dim = args.z_enc_dim # encoder hidden layer
        self.x_dec_dim = args.x_dec_dim # decoder hidden layer
        if self.y_dim == 1:
            self.y_out_dim = 2
        else:
            self.y_out_dim = self.y_dim
        self.dropout_rate = args.dropout

        self.hidden_layer = args.hidden_layer

        self.VAE = VAE(self.x_dim,self.s_dim,self.y_dim,self.z_enc_dim,self.x_dec_dim,self.z_dim,self.b_dim,self.dropout_rate)
        self.pred_net = PredNet(self.y_dim,self.b_dim,self.hidden_layer,self.dropout_rate)
        self.D = Discriminator(self.z_dim+self.b_dim).to(self.device)

        self.lr_VAE = args.lr_VAE
        self.lr_D = args.lr_D

        self.optim_VAE = optim.Adam(self.VAE.parameters(), lr=self.lr_VAE)
        self.optim_pred = optim.Adam(self.pred_net.parameters(),lr = self.lr_VAE)
        self.optim_D = optim.Adam(self.D.parameters(), lr=self.lr_D)

        self.nets = [self.VAE, self.pred_net,self.D]

        self.bce = BCEWithLogitsLoss()
        self.ce_logits = CrossEntropyLoss(reduction='none')
        self.mse = MSELoss()
        self.gamma = args.gamma
        self.alpha = args.alpha
        self.beta = args.beta
        self.xi = args.xi

        self.softmax = Softmax(dim = 1)
        self.sigmoid = nn.Sigmoid()
        self.temprature = args.temprature


    def net_mode(self, train):
        for net in self.nets:
            if train:
                net.train()
            else:
                net.eval()

    def training(self):
        self.train_iter, self.test_iter = read_data(self.input_data,self.target_data,self.theta_dict,self.batch_size,self.seed)

        dp_clean = compute_dp(self.input_data['s'].squeeze().long(),self.input_data['y'])
        dp_noise = compute_dp(self.input_data['s'].squeeze().long(),self.input_data['y_noise'])
        print("dp on clean data: {:.2f} >> dp on noisy data: {:.2f}".format(dp_clean,dp_noise))

        scheduler1 = optim.lr_scheduler.ExponentialLR(self.optim_VAE, gamma=0.9)
        scheduler2 = optim.lr_scheduler.MultiStepLR(self.optim_VAE, milestones=[30,80], gamma=0.1)

        for epoch in range(0, self.epochs+1):
            self.train_epoch_progress(epoch)
            scheduler1.step()
            scheduler2.step()

        torch.save(self.VAE, 'model.pkl')

        acc, dp = self.evaluate()


    def train_epoch_progress(self, epoch):

        self.net_mode(train=True)

        train_loss = 0
        truth_res = []
        pred_res = []

        for batch in self.train_iter:
            feature, label, sensitive_feature = batch[0], batch[1], batch[2]

            outputs = self.VAE(feature,sensitive_feature)
            truth_res += list(label.cpu().data.numpy())
            z_representation = outputs['z_encoded']
            biased_representation = outputs['b_encoded']

            n_samples = len(z_representation)

            # TP ESTIMATE KL[q(z,b)||q(z)q(b)]
            b = torch.distributions.Uniform(low=0,high=1).sample((n_samples,self.b_dim))
            z = torch.distributions.Normal(0,1).sample((n_samples,self.z_dim))
            latent = torch.cat([z, b], dim=1).detach()

            output_fake = self.D(latent)
            zeros = torch.zeros(n_samples, dtype=torch.long)
            ones = torch.ones(n_samples,dtype=torch.long)


            output_real = self.D(torch.cat([z_representation,biased_representation],dim=1))
            D_tc_loss = 0.5*(F.cross_entropy(output_real, zeros) + F.cross_entropy(output_fake, ones))

            self.optim_D.zero_grad()
            D_tc_loss.backward(retain_graph=True)
            self.optim_D.step()

            pred = outputs['pred']
            pred = self.softmax(pred)

            reconst_out = self.pred_net(biased_representation,pred)
            pred_b, pred_ym = self.softmax(reconst_out['by']/self.temprature), self.softmax(reconst_out['yy'])

            supervision_loss = self.ce_logits(pred,label)

            tyb_entropy = self.ce_logits(pred_b,label)
            tyy_entropy = self.ce_logits(pred_ym,label)

            constrain_loss = self.xi*(tyy_entropy).mean()+ self.beta*tyb_entropy.mean()
            s = sensitive_feature.unsqueeze(-1).float()

            vae_recon_loss = self.mse(outputs['x_decoded'],feature) + self.alpha*self.bce(outputs['s_decoded'],s)
            vae_kld = kl_divergence(outputs['z_enc_mu'], outputs['z_enc_logvar'])
            log_x1, log_x2 = output_real[:, :1].clone().detach() , output_real[:, 1:].clone().detach()
            vae_tc_loss = (log_x1-log_x2).mean()

            vae_loss = vae_recon_loss+vae_kld + self.gamma*vae_tc_loss
            loss = vae_loss + constrain_loss + supervision_loss.mean()


            train_loss += loss.item()
            train_loss /= len(label)

            self.optim_VAE.zero_grad()
            self.optim_pred.zero_grad()
            loss.backward()
            self.optim_VAE.step()
            self.optim_pred.step()


            pred = pred.cpu()
            pred_label = pred.data.max(1)[1].numpy()
            pred_res += [x for x in pred_label]


    def evaluate(self):
        self.net_mode(train=False)

        truth_res = []
        pred_res = []
        sensitive_membership = []

        for batch in self.test_iter:
            feature, label, sensitive_feature = batch[0], batch[1], batch[2]
            outputs = self.VAE(feature,sensitive_feature)

            truth_res += list(label.cpu().data.numpy())
            pred = outputs['pred']
            pred = self.softmax(pred)
            pred = pred.cpu()
            pred_label = pred.data.max(1)[1].numpy()
            pred_res += [int(x) for x in pred_label]
            sensitive_membership += list(sensitive_feature.cpu().data.numpy())
            label = label.cpu()
        sensitive_membership = np.array(sensitive_membership)

        acc = get_accuracy(truth_res, pred_res)
        dp = compute_dp(sensitive_membership,np.array(pred_res))

        pred_check = np.array(pred_res).sum()/len(pred_res)

        print("sanity check for prediction: {:.2f}".format(pred_check*100))
        print('TEST' + ': ACC %.2f DP %.2f' % (acc*100, dp))
        return acc, dp
