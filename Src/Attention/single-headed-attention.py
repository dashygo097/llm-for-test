import torch
import math

import torch.nn.functional as F

from torch import nn



# masked_softmax
def masked_softmax(x , valid_lens):

    return


def scaled_dot_product_attention(Q,K,V):

    d = Q.shape[-1]
    results = torch.bmm(Q , K.transpose(-1,-2))/ math.sqrt(d)
    weights = F.softmax(results , dim=-1)
    outputs = torch.bmm(weights , V)

    return outputs,weights

class DotProductAttetion(nn.Module):

    def __init__(self ,embed_size ,q_size ,k_size , v_size,dropout, **kwargs):
        super().__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)


    def forward(self ,Q,K,V):

        outputs,weights = scaled_dot_product_attention(Q,K,V)

        return outputs,weights

def init_weights(layer):
    if layer == nn.Linear:
        nn.init.normal_(layer.weight , std=1.)
