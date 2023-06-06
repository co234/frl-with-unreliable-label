import torch
from torch.nn import Module, Dropout, ReLU
from model.mlp_vae import EncoderNet, DecoderNet, FC


class VAE(Module):
    """
    Input:
    x_dim: int, non-sensitive attribute x dimension
    s_dim: int, sensitive attribute a dimension
    y_dim: int, label dimension
    z_enc_dim: encoder dimension for z
    x_dec_dim: decoder dimension for x
    z_dim: fair representation dimension
    b_dim: biased representation dimension

    This VAE structure is adapted from VFAE: https://github.com/yolomeus/DMLab2020VFAE
    """

    def __init__(self,
                 x_dim,
                 s_dim,
                 y_dim,
                 z_enc_dim,
                 x_dec_dim,
                 z_dim,
                 b_dim,
                 dropout_rate,
                 activation=ReLU()):
        super().__init__()
        y_out_dim = 2 if y_dim == 1 else y_dim
       
        # ENCODER
        self.encoder_z = EncoderNet(x_dim, z_enc_dim, z_dim+b_dim, activation)
        # DECODER
        self.decoder_x = DecoderNet(z_dim+b_dim , x_dec_dim, x_dim , activation)
        self.decoder_s = DecoderNet(b_dim , z_dim, s_dim , activation)
        self.decoder_y = DecoderNet(z_dim, x_dec_dim, y_out_dim,activation)

        self.b_dim = b_dim
        self.dropout = Dropout(dropout_rate)

    def forward(self, x,s):
        
        z_encoded, z_enc_logvar, z_enc_mu = self.encoder_z(x)
        f_encoded = z_encoded[:,:-self.b_dim]
        if self.b_dim == 1:
            b_encoded = torch.unsqueeze(z_encoded[:,-self.b_dim], -1)
        else:
            b_encoded = z_encoded[:,-self.b_dim:]

        x_decoded = self.decoder_x(z_encoded)
        s_decoded = self.decoder_s(b_encoded)
        y_latent = self.decoder_y(f_encoded)


        outputs = {
            # ENCODER OUTPUTS
            'z_encoded': f_encoded,
            'b_encoded':b_encoded,
            'z_enc_logvar': z_enc_logvar,
            'z_enc_mu': z_enc_mu,

            # DECODER OUTPUTS
            'x_decoded': x_decoded,
            's_decoded':s_decoded,
            'pred': y_latent
        }

        return outputs




class PredNet(Module):

    def __init__(self,
                 y_dim,
                 b_dim,
                 hidden_layer,
                 dropout_rate):
        super().__init__()
        y_out_dim = 2 if y_dim == 1 else y_dim
       
        self.pred_by = FC(b_dim,hidden_layer,y_out_dim)
        self.pred_ym = FC(y_out_dim,hidden_layer,y_out_dim)

        self.dropout = Dropout(dropout_rate)

    def forward(self, bias_term,ym):

        b_pred = self.pred_by(bias_term)
        ym_pred = self.pred_ym(ym)


        outputs = {
            'by': b_pred,
            'yy': ym_pred,
        }

        return outputs












    


