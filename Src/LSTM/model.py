import torch

import torch.nn.functional as F

from torch import nn


def normal(shape,device):
    return torch.randn(size=shape, device=device,) *0.01


def three(num_inputs,num_hiddens,device):
    return (nn.Parameter(normal((num_inputs, num_hiddens),device)),
           nn.Parameter(normal((num_hiddens, num_hiddens),device)),
           nn.Parameter(torch.zeros(num_hiddens, device=device)))

def init_lstm_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device),
            torch.zeros((batch_size, num_hiddens), device=device))

def lstm(inputs, state, params):
    [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c,
     W_hq, b_q , dense] = params
    (H, C) = state
    outputs = []
    for X in inputs:
        I = torch.sigmoid((X @ W_xi) + (H @ W_hi) + b_i)
        F = torch.sigmoid((X @ W_xf) + (H @ W_hf) + b_f)
        O = torch.sigmoid((X @ W_xo) + (H @ W_ho) + b_o)
        C_tilda = torch.tanh((X @ W_xc) + (H @ W_hc) + b_c)
        C = F * C + I * C_tilda
        H = O * torch.tanh(C)
        Y = dense((H @ W_hq) + b_q)
        outputs.append(Y)

    return torch.cat(outputs, dim=0), (H, C)

def get_lstm_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    W_xi, W_hi, b_i = three(num_inputs,num_hiddens,device)  # 输入门参数
    W_xf, W_hf, b_f = three(num_inputs,num_hiddens,device)  # 遗忘门参数
    W_xo, W_ho, b_o = three(num_inputs,num_hiddens,device)  # 输出门参数
    W_xc, W_hc, b_c = three(num_inputs,num_hiddens,device)  # 候选记忆元参数
    # 输出层参数
    W_hq = normal((num_hiddens, num_outputs),device)
    b_q = torch.zeros(num_outputs, device=device)
    dense = nn.Linear(num_outputs, 1).to(device)
    nn.init.normal_(dense.weight,std=0.01)
    # 附加梯度
    params = [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc,
              b_c, W_hq, b_q ]
    for param in params:
        param.requires_grad_(True)
    params.append(dense)
    return params


class Basic097(nn.Module):

    def __init__(self,vocab_size,num_hiddens,device,embed_size = 128,max_length = 128,
        get_params=get_lstm_params,init_state=init_lstm_state,forward_fn=lstm):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size,embedding_dim=embed_size,device=device)
        self.embed_size = embed_size
        self.num_hiddens = num_hiddens
        self.max_length = max_length
        self.params = get_params(embed_size,num_hiddens,device)
        self.init_state = init_state
        self.forward_fn = forward_fn
        self.output_head = nn.Linear(max_length , 1).to(device)

    def forward(self,inputs , state):
        inputs = self.embedding(inputs).transpose(0,1)
        return self.forward_fn(inputs,state,self.params)

    def begin_state(self, batch_size, device):
        return self.init_state(batch_size, self.num_hiddens, device)

    def predict(self,outputs,device):
        outputs = self.output_head(outputs.reshape(-1,self.max_length)).to(device)
        return outputs
def encoder(tokenizer,inputs,max_length=512,padding=True):
    outputs = []
    if padding:
        for i in inputs:
            encoded_i = tokenizer.encode(i,max_length=max_length,truncation=True)
            if len(encoded_i) > max_length:
                return "error: OUT of index"
            else:
                while len(encoded_i) < max_length:
                    encoded_i.append(0)
            outputs.append(encoded_i)
    else :
        for i in inputs:
            encoded_i = tokenizer.encode(i,max_length=max_length,truncation=True)
            outputs.append(encoded_i)
    return torch.tensor(outputs,dtype=torch.long)
'''


from datasets import load_dataset
from transformers import AutoTokenizer

dataset = load_dataset("./stanford-imdb/plain_text")
tokenizer = AutoTokenizer.from_pretrained("./bert-tokenizer")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X = ["I love you!","I do not like early eight."]

# y = "[PAD]"
# print(tokenizer(y))

num_hiddens = 32
max_length = 128
X = encoder(tokenizer,X,max_length=max_length)
print(X.shape)
net = Basic097(tokenizer.vocab_size, num_hiddens, device)
state = net.begin_state(X.shape[0], device)
Y, new_state = net(X.to(device), state)
Z = net.predict(Y)
print(Y.shape, len(new_state), new_state[0].shape)
print(Z.shape)

'''