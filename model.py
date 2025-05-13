import torch
import torch.nn as nn
import math

"""
Implementation of transformer from the paper "Attention is all you need"
"""

class InputEmbeddings(nn.Module):
    """
    Class to convert tokens into embedding vectors of size 512
    """

    def _init_(self, d_model, vocab_size):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):

        return self.embedding(x) * math.sqrt(self.d_model)
    
class PositionalEncoding(nn.Module):

    """
    Class to convert the tokens into positional encodings to preserve context and position of tokens

    Inputs:
        d_model: embedding size
        seq_len: maximum len of a sentence
        dropout: ratio of dropout layers
    """

    def __init__(self, d_model, seq_len, dropout):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # matrix shape of (seq_len, d_model)
        pos_en = torch.zeros(seq_len, d_model)
        # vector of shape (seq_len)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Applying sin and cosine to even and odd positions respectibely
        pos_en[:, 0::2] = torch.sin(position * div_term)
        pos_en[:, 1::2] = torch.cos(position * div_term)

        pos_en = pos_en.unsqueeze(0)    # (1, seq_len, d_model)

        self.register_buffer('pos_en', pos_en)

    def forward(self, x):

        x = x + (self.pos_en[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)

class LayerNormalization(nn.Module):

    def __init__(self, eps = 10**-6):
        super().__init__()  
        self.eps = eps
        self.aplha = nn.Parameter(torch.ones(1))    # Multiplied
        self.bisa = nn.Parameter(torch.zeros(1))    # Added

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim=True)
        std = x.std(dim = -1, keepdim = True)
        return self.aplha * (x - mean) / (std + self.eps) + self.bias

class FeedForwardBlock(nn.Module):

    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.layer_1 = nn.Linear(d_model, d_ff)     # W1 and b1
        self.dropout = nn.Dropout(dropout)
        self.layer_2 = nn.Linear(d_ff, d_model)     # W2 and b2

    def forward(self, x):
        # (Batch, seq_len, d_model) --> (Batch, seq_len, d_ff) --> (Batch, seq_len, d_model)
        x = torch.relu(self.layer_1(x))
        x = self.dropout(x)
        return self.layer_2(x)

class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model, num_heads, dropout):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0, "d_model not divisble by num_heads"
        
        self.d_k = d_model // num_heads
        self.w_q == nn.Linear(d_model, d_model)     # W_q
        self.w_k == nn.Linear(d_model, d_model)     # W_k
        self.w_v == nn.Linear(d_model, d_model)     # W_v
        
        self.w_o = nn.Linear(d_model, d_model)      # Wo
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask):
        """
        Inputs:
            q: queries
            k: keys
            v: values
            mask: mask the tokens ahead of the current token
        """
        query = self.w_q(q)         # (Batch, seq_len, d_model) --> (Batch, seq_len, d_model)
        key = self.w_k(k)           # (Batch, seq_len, d_model) --> (Batch, seq_len, d_model)
        value = self.w_v(v)         # (Batch, seq_len, d_model) --> (Batch, seq_len, d_model)

        #  # (Batch, seq_len, d_model) --> (Batch, seq_len, num_heads, d_k) --> (Batch, num_heads, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.num_heads, self.d_k).transpose(1, 2)
        key = query.view(key.shape[0], key.shape[1], self.num_heads, self.d_k).transpose(1, 2)
        value = query.view(value.shape[0], value.shape[1], self.num_heads, self.d_k).transpose(1, 2)


