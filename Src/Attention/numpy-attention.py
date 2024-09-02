import numpy as np

np.random.seed(114514)


def softmax(X):

    X_exp = np.exp(X)
    exp_sum = np.sum(X_exp, axis=-1, keepdims=True)
    return X_exp / exp_sum


def scaled_dot_product_attention(Q, K, V, mask=None):

    # 1. 需要完成调整 K 的转置来匹配 Q 的最后一个维度，

    K_T = K.transpose(0, 1, 3, 2)

    # 2. 计算attn_score并缩放，
    # 3. softmax 应用于最后一个轴计算attn_weight，

    rec_dim = np.sqrt(K.shape[-1])

    attention_weights = softmax(Q @ K_T / rec_dim)
    # 4. 应用attn_weights输出output

    output = attention_weights @ V

    # 5. 带掩码mask的的注意力可以不用实现,但请记住encoder和decoder的transformer块是不一样的，很大一部分都在就在mask上

    return output, attention_weights


def multi_head_attention(embed_size, num_heads, input, mask=None):

    # 1. embed_size 确保可以等分 num_heads 份， 否则输出错误

    if embed_size % num_heads != 0:
        print("error!")
        return

    # 2. 随机初始化Wq,Wk,Wv,Wo矩阵，并对input做线性变换

    batch_size, n_tokens, dim = input.shape
    Wq = np.random.randn(num_heads, dim, dim // num_heads)
    Wk = np.random.randn(num_heads, dim, dim // num_heads)
    Wv = np.random.randn(num_heads, dim, dim // num_heads)
    Wo = np.random.randn(dim // num_heads, dim // num_heads)

    Q = np.dot(input, Wq)
    K = np.dot(input, Wk)
    V = np.dot(input, Wv)
    Q = Q.transpose(0, 2, 1, 3)
    K = K.transpose(0, 2, 1, 3)
    V = V.transpose(0, 2, 1, 3)

    # 3. 利用scaled_dot_product_attention()输出的attn_output计算O

    attn_output, attn_weights = scaled_dot_product_attention(Q, K, V)

    # 4. 返回output, attN_weights

    attn_output = attn_output.transpose(0, 2, 1, 3)
    output = np.dot(attn_output, Wo).reshape(batch_size, n_tokens, dim)
    weights = attn_weights

    return output, weights

    # test e.g.


embed_size = 128
num_heads = 8
input = np.random.randn(10, 20, embed_size)
output, weights = multi_head_attention(embed_size, num_heads, input)
print(output.shape, weights.shape)
print(output[0][0][:10], weights[0][0][0][:10])
