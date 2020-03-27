import torch
import torch.nn as nn

from source.modules.attention import *
from source.modules.multi_head_attention import *
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from source.modules.LayerNorm import *
from source.modules.mlp.mlp import *

class RnnEncoder(nn.Module):
    def __init__(self,
                 embedding_dim=300,
                 model="GRU",
                 dropout=0,
                 hidden_size=None,
                 encoder_layers=2,
                 query=False,
                 query_size=None,
                 bidirectional=False):
        super(RnnEncoder, self).__init__()

        if hidden_size is None:
            hidden_size = embedding_dim

        self.dropout = dropout
        self.types = ['GRU', 'LSTM']

        assert model in self.types

        if model == 'GRU':
            self.model = nn.GRU(
                embedding_dim,
                hidden_size,
                num_layers=encoder_layers,
                batch_first=True,
                dropout=dropout,
                bidirectional=bidirectional)

        elif model == 'LSTM':
            self.model = nn.LSTM(
                embedding_dim,
                hidden_size,
                num_layers=encoder_layers,
                batch_first=True,
                dropout=dropout,
                bidirectional=bidirectional)

        self.model_2 = nn.GRU(
                hidden_size,
                hidden_size,
                num_layers=encoder_layers,
                batch_first=True,
                dropout=dropout,
                bidirectional=bidirectional)

        self.max_pool = nn.MaxPool1d(2, stride=2)

        if query:
            self.query = query
            self.attention = Attention(
                query_size=query_size,
                memory_size=hidden_size,
                return_attn_only=True,
                mode='mlp')

        self.norm = LayerNorm(hidden_size)

        self.dropout = nn.Dropout(0.5)


    def forward(self, batch, lengths, query=None, h_0=None, c_0=None):
        """
        forward
        """
        # Tensor(batch_size, max_sentence_length, word_embeds_size)

        packed_batch = pack_padded_sequence(batch, lengths=lengths, batch_first=True, enforce_sorted=False)

        output, _ = self.model(packed_batch)

        padded_output, lengths = pad_packed_sequence(output, batch_first=True)

        # packed_batch = pack_padded_sequence(padded_output, lengths=lengths, batch_first=True, enforce_sorted=False)
        #
        # output, _ = self.model_2(packed_batch)
        #
        # padded_output, lengths = pad_packed_sequence(output, batch_first=True)

        # padded_output = self.norm(padded_output)
        # padded_output = self.dropout(padded_output)

        if query is not None:
            weights = self.attention(query, padded_output)
            re_output = torch.bmm(weights, padded_output)
        else:
            # (batch_size, hidden_size)
            re_output = torch.cat([torch.div(padded_output.sum(1)[i], lengths[i]).unsqueeze(0)
                                   for i in range(lengths.size(-1))], 0)
            # (batch_size, 1, hidden_size)
            re_output = re_output.unsqueeze(1)

        result = self.max_pool(re_output)

        return result

