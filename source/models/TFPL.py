from source.models.base_model import *
from source.modules.encoders.encoder import *
from source.modules.encoders.transformer_encoder import *
from source.modules.multi_head_attention import *
from source.modules.ffn.PSffn import *

import torch.nn as nn
import torch


class TFPL(BaseModel):

    def __init__(self,
                 embedding_dim,
                 h=6,
                 ff_dim=300,
                 dropout=0.1
                 ):
        super(TFPL, self).__init__()

        attn = MultiHeadedAttention(h, embedding_dim)
        ff = PositionwiseFeedForward(embedding_dim, ff_dim, dropout)
        encoder_layer = EncoderLayer(embedding_dim, attn, ff, dropout)

        self.EmbedsEncoder = TransformerEncoder(encoder_layer, N=3)

        self.PositionEncoder = PositionalEncoding(embedding_dim, dropout)


    def forward(self, x, length):
        """
        forward.
        """
        result = []

        for i, j in enumerate(length):
            out = x[i, 0:j, :].unsqueeze(0)
            out = self.PositionEncoder(out)
            out = self.EmbedsEncoder(out, mask=None)
            result.append(out.sum(1)/j)

        return torch.cat(result).unsqueeze(1)

