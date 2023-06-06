import torch
import torch.nn as nn
from torch.nn import Module, Linear


class EncoderNet(Module):
    def __init__(self, in_features, hidden_dim, z_dim, activation):
        super().__init__()
        self.encoder = Linear(in_features, hidden_dim)
        self.activation = activation
        self.logvar_encoder = Linear(hidden_dim, z_dim)
        self.mu_encoder = Linear(hidden_dim, z_dim)

    def forward(self, inputs):
        x = self.encoder(inputs)
        logvar = (0.5 * self.logvar_encoder(x)).exp()
        mu = self.mu_encoder(x)
        epsilon = torch.randn_like(mu)
        z = epsilon * mu + logvar
        return z, logvar, mu


class DecoderNet(Module):
    def __init__(self, in_features, hidden_dim, latent_dim, activation):
        super().__init__()
        self.lin_encoder = Linear(in_features, hidden_dim)
        self.activation = activation
        self.lin_out = Linear(hidden_dim, latent_dim)

    def forward(self, inputs):
        x = self.activation(self.lin_encoder(inputs))
        return self.lin_out(x)


class FC(nn.Module):
    def __init__(self, input_sizze,hidden_size, num_classes):
        super(FC, self).__init__()
        self.relu = nn.ReLU()
        self.activation = nn.Sigmoid()
        self.fc2 = nn.Linear(input_sizze, hidden_size)
        self.fc2_drop = nn.Dropout(0.0)
        self.fc3 = nn.Linear(hidden_size, 2)  
    
    def forward(self, x):
        out = self.fc2(x)
        out = self.relu(out)
        out = self.fc2_drop(out)
        logit = self.fc3(out)
        return logit



