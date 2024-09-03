import torch
import math

import torch.nn.functional as F

from torch import nn



# masked_softmax
def masked_softmax(x , masked = None):

    if masked is None:
        Y = F.softmax(x , dim=-1)
    elif masked == "^":
        for i in range(x.shape[-2]):
            x[...,i,i+1::] = -1e7
        x = x.transpose(-1,-2)
        Y = F.softmax(x , dim=-1)

    return Y


def scaled_dot_product_attention(Q,K,V,masked=None):

    d = Q.shape[-1]
    results = (Q @ K.transpose(-1,-2))/ math.sqrt(d)
    weights = masked_softmax(results , masked=masked)
    outputs = weights @ V

    return outputs,weights

class MultiHeadedAttetion(nn.Module):

    def __init__(self ,num_heads,d_model,q_size ,k_size , v_size,dropout,masked=None,bias = False, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.masked = masked
        self.W_q = nn.Linear(q_size , d_model,bias=bias)
        self.W_k = nn.Linear(k_size , d_model,bias=bias)
        self.W_v = nn.Linear(v_size , d_model,bias=bias)
        self.W_o = nn.Linear(d_model , d_model,bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self ,Q,K,V):
        batch_size, context, embed_size = Q.shape
        if embed_size % self.num_heads != 0:
            print("error!")
            return
        Q = self.W_q(Q)
        K = self.W_k(K)
        V = self.W_v(V)
        Q = Q.view(batch_size,context,self.num_heads,embed_size).transpose(1,2)
        K = K.view(batch_size,context,self.num_heads,embed_size).transpose(1,2)
        V = V.view(batch_size,context,self.num_heads,embed_size).transpose(1,2)

        outputs,weights = scaled_dot_product_attention(Q,K,V,masked=self.masked)
        outputs = (outputs.transpose(1,2)).reshape(
            batch_size , context ,self.d_model)
        outputs = self.W_o(outputs)

        return outputs,weights

class DotProductAttetion(nn.Module):

    def __init__(self,dropout,masked = None, **kwargs):
        super().__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.masked = masked


    def forward(self ,Q,K,V):

        d = Q.shape[-1]
        results = torch.bmm(Q, K.transpose(-1, -2)) / math.sqrt(d)
        weights = masked_softmax(results, masked=self.masked)
        outputs = torch.bmm(weights, V)

        return outputs,weights


def init_weights(layer):
    if layer == nn.Linear:
        nn.init.normal_(layer.weight , std=1.)

'''
attention = MultiHeadedAttetion(8,128,16,16,16,0.5)
# batch_size , context , token_length
Q = torch.normal(0,10,size=(10, 20 , 16))
K = torch.normal(0,10,size=(10, 20 , 16))
V = torch.normal(0,10,size=(10, 20 , 16))

attention.apply(init_weights)
outputs,weights = attention(Q,K,V)
print(outputs.shape , weights.shape)
print(outputs[0][0][:10])
print(weights[0][0][0][:20])
'''
