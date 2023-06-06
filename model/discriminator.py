import torch.nn as nn
import torch.nn.init as init

# SAMME DISCRIMINATOR STRUCTURE AS FACTORVAE
# Adapted from https://github.com/1Konny/FactorVAE

class Discriminator(nn.Module):
    def __init__(self, z_dim):
        super(Discriminator, self).__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(
            nn.Linear(z_dim, 50),
            nn.LeakyReLU(0.1, True),
            nn.Linear(50, 50),
            nn.LeakyReLU(0.1, True),
            nn.Linear(50, 50),
            nn.LeakyReLU(0.1, True),
            nn.Linear(50, 2)
        )
        self.weight_init()

    def weight_init(self, mode='normal'):
        if mode == 'kaiming':
            initializer = kaiming_init
        elif mode == 'normal':
            initializer = normal_init
        for block in self._modules:
            for m in self._modules[block]:
                initializer(m)
                
    def forward(self, z):
        return self.net(z).squeeze()



def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


def normal_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.normal_(m.weight, 0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)
