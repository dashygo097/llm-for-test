import torch
import math

import torch.nn.functional as F

from torch import nn



# masked_softmax
def masked_softmax(x , valid_lens):

    return


def scaled_dot_product_attention(Q,K,V):

    d = Q.shape[-1]
    results = (Q @ K.transpose(-1,-2))/ math.sqrt(d)
    weights = F.softmax(results , dim=-1)
    outputs = weights @ V

    return outputs,weights

class MultiHeadedAttetion(nn.Module):

    def __init__(self ,num_heads,num_hidden,q_size ,k_size , v_size,dropout, **kwargs):
        super().__init__(**kwargs)
        self.num_hidden = num_hidden
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.W_q = nn.Linear(q_size , num_hidden * num_heads)
        self.W_k = nn.Linear(k_size , num_hidden * num_heads)
        self.W_v = nn.Linear(v_size , num_hidden * num_heads)
        self.W_o = nn.Linear(num_hidden*num_heads , num_hidden * num_heads)

    def forward(self ,Q,K,V):
        batch_size, context, embed_size = Q.shape
        if embed_size % self.num_heads != 0:
            print("error!")
            return
        Q = Q.view(batch_size , context , self.num_heads , embed_size//self.num_heads).transpose(2,1)
        K = K.view(batch_size, context, self.num_heads, embed_size // self.num_heads).transpose(2, 1)
        V = V.view(batch_size, context, self.num_heads, embed_size // self.num_heads).transpose(2, 1)
        Q = self.W_q(Q)
        K = self.W_k(K)
        V = self.W_v(V)

        outputs,weights = scaled_dot_product_attention(Q,K,V)

        return outputs,weights

def init_weights(layer):
    if layer == nn.Linear:
        nn.init.normal_(layer.weight , std=1.)

attention = MultiHeadedAttetion(8,16,16,16,16,0.5)
# batch_size , context , token_length
Q = torch.normal(0,10,size=(10, 20 , 8*16))
K = torch.normal(0,10,size=(10, 20 , 8*16))
V = torch.normal(0,10,size=(10, 20 , 8*16))

attention.apply(init_weights)
outputs,weights = attention(Q,K,V)
print(outputs.shape , weights.shape)
print(outputs[0][0][:10])
print(weights[0][0][0][:20])
