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
    
    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]

        # (Batch, num_head, seq_len, d_k) --> (Batch, num_head, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2,-1))/math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
        
        attention_scores = attention_scores.softmax(dim = -1)       #(Batch, h, seq_len, seq_len)
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (attention_scores @ value), attention_scores


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

        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        # (Batch, num_head, seq_len, d_k) --> (Batch, seq_len, num_head, d_k) --> (Batch, seq_len, d_model)
        x = x.transpose(1, 2).contigous().view(x.shape[0], -1, self.num_heads * self.d_k)

        return self.w_o(x)
    
class ResidualConnection(nn.Module):

    def __init__(self, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
    
class EncoderBlock(nn.Module):

    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout):
        super().__init__()
        self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.resdiual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])
        
    def forward(self, x, src_mask):
        x = self.resdiual_connection[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.resdiual_connectionp[1](x, self.feed_forward_block)
        return x
    
class Encoder(nn.Module):

    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
