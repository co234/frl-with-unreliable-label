from model.mlp_vae import *
import numpy as np
from torch.utils import data



def get_accuracy(truth, pred):
    assert len(truth) == len(pred)
    right = 0
    for i in range(len(truth)):
        if truth[i] == pred[i]:
            right += 1.0
    return right / len(truth)


class Dataset():
    def __init__(self, x, labels, sensitive_attribute):
        self.x = x
        self.labels = labels
        self.sensitive_attribute = sensitive_attribute
        
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, index):
        return self.x[index], int(self.labels[index]), int(self.sensitive_attribute[index])


def add_label_bias(yclean,rho,theta_dict,seed):
    """
    theta_0_p: P(Y=+1|Z=-1,A=0)
    theta_0_m: P(Y=-1|Z=+1,A=0)
    theta_1_p: P(Y=+1|Z=-1,A=1)
    theta_1_m: P(Y=-1|Z=+1,A=1)
    """
    n = len(yclean)
    np.random.seed(seed)

    t_0_p, t_0_m, t_1_p,t_1_m = theta_dict['theta_0_p'],theta_dict['theta_0_m'],theta_dict['theta_1_p'],theta_dict['theta_1_m']


    def locate_group(label,sensitive_attr,a,y):
        return np.intersect1d(np.where(sensitive_attr==a)[0],np.where(label==y)[0])

    g_01, g_00 = locate_group(yclean,rho,0,1),locate_group(yclean,rho,0,0)
    g_11, g_10 = locate_group(yclean,rho,1,1),locate_group(yclean,rho,1,0)

    group = [g_01,g_00,g_11,g_10]
    theta = [t_0_m,t_0_p,t_1_m,t_1_p]
    tilde_y = [0,1,0,1]

    t = yclean.copy()

    for i in range(len(group)):
        for j in range(len(group[i])):
            p = np.random.uniform(0,1)
            if p < theta[i]:
                t[group[i][j]] = tilde_y[i]
            else:
                t[group[i][j]] = yclean[group[i][j]]

    return t


def compute_di(rho,y):
    p_idx, u_idx = np.where(rho==1)[0], np.where(rho==0)[0]
    f_ratio, m_ratio = (y[p_idx]==1).sum()/len(p_idx), (y[u_idx]==1).sum()/len(u_idx)
    di = np.min([(f_ratio/(m_ratio+1e-15)),(m_ratio/(f_ratio+1e-15))])

    return di

def compute_dp(rho,y):
    p_idx, u_idx = np.where(rho==1)[0], np.where(rho==0)[0]

    f_ratio, m_ratio = (y[p_idx]==0).sum()/len(p_idx), (y[u_idx]==0).sum()/len(u_idx)
    dp = np.abs(f_ratio-m_ratio)
    return dp


def read_data(input_data,target_data,noise_dict,batch_size,seed):

    y = input_data['y'].squeeze().long().cpu().detach().numpy()
    s = input_data['s'].cpu().detach().numpy()

    y_new = add_label_bias(y,s,noise_dict,seed)
    y_tilder = torch.as_tensor(y_new, dtype=torch.float32).unsqueeze(-1)
    input_data['y_noise'] = y_tilder

    #===============compute original di=======================
    training_set = Dataset(input_data['x'], input_data['y_noise'], input_data['s'])
    training_generator = data.DataLoader(training_set, batch_size=batch_size,shuffle=True)


    testing_set = Dataset(target_data['x'], target_data['y'], target_data['s'])
    testing_generator = data.DataLoader(testing_set, batch_size=batch_size,shuffle=True)

    return training_generator, testing_generator


def compute_deo(label,pred,s):

    label = np.array(label)

    y_pos = np.where(label==1)[0]
    f_idx,m_idx = np.where(s==0)[0], np.where(s==1)[0]
  
    pos_f,pos_m = np.intersect1d(y_pos,f_idx), np.intersect1d(y_pos,m_idx)
    f_eo,m_eo = (pred[pos_f]==1).sum()/(len(pos_f)+1e-15), (pred[pos_m]==1).sum()/(len(pos_m)+1e-15)

    return np.abs(f_eo-m_eo)



def kl_divergence(mu, logvar):
    kld = -0.5*(1+logvar-mu**2-logvar.exp()).sum(1).mean()
    return kld


