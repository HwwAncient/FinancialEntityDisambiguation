from source.models.base_model import *
from source.modules.encoders.encoder import *
from source.modules.LayerNorm import *
from source.models.TFPL import *
from source.modules.mlp.mlp import *

import torch.nn as nn
import torch

def get(size, p):
    mix = torch.rand(size)
    mix[mix < p] = 0
    mix[mix > p] = 1

    out = torch.zeros(size, size)

    for i in range(size):
        out[i, i] = mix[i]
    return out



class NeuPL(BaseModel):

    def __init__(self,
                 embedding_dim,
                 entity_hidden_size,
                 context_hidden_size,
                 mlp_hidden_size,
                 mlp_bias=True,
                 encoder_model='GRU',
                 encoder_layers=1,
                 dropout=0
                 ):
        super(NeuPL, self).__init__()
        self.mlp_bias = mlp_bias
        # self.entity_vec_dim = int(hidden_size/2 + embedding_dim)
        self.entity_vec_dim = int(entity_hidden_size/2)
        self.context_vec_dim = context_hidden_size

        self.EntityEncoder = RnnEncoder(
            embedding_dim=embedding_dim,
            hidden_size=entity_hidden_size,
            encoder_layers=encoder_layers,
            model=encoder_model)

        self.ContextEncoder = RnnEncoder(
            embedding_dim=embedding_dim,
            hidden_size=context_hidden_size,
            query=True,
            query_size=self.entity_vec_dim,
            encoder_layers=encoder_layers,
            model=encoder_model)


        # self.mlp = MLP(
        #     layers_dim=self.mlp_layers,
        #     bias=self.mlp_bias)

        self.out_liner_in = nn.Linear(
            in_features=self.entity_vec_dim+self.context_vec_dim,
            out_features=mlp_hidden_size)

        self.out_liner_out = nn.Linear(
            in_features=mlp_hidden_size,
            out_features=1)

        self.context_mlp = MLP(
            layers_dim=[context_hidden_size, int(context_hidden_size*3/2), context_hidden_size],
            bias=True,
            function=torch.relu)

        self.entity_describe_mlp = MLP(
            layers_dim=[self.entity_vec_dim, int(self.entity_vec_dim*3/2), self.entity_vec_dim],
            bias=True,
            function=torch.relu)

        self.entity_mlp = MLP(
            layers_dim=[embedding_dim, int((embedding_dim+entity_hidden_size)*3/4), int(entity_hidden_size/2)],
            bias=True,
            function=torch.relu)

        self.dropout = nn.Dropout(0.4)

        self.multi_head_attention = MultiHeadedAttention(h=6, d_model=embedding_dim)

        # self.LeftContextEmbedsEncoder = TFPL(embedding_dim=embedding_dim)
        # self.RightContextEmbedsEncoder = TFPL(embedding_dim=embedding_dim)
        # self.EntityEmbedsEncoder = TFPL(embedding_dim=embedding_dim)

        self.max_pool = nn.MaxPool1d(2, stride=2)

        self.norm1 = LayerNorm(int(entity_hidden_size / 2))

        self.norm2 = LayerNorm(entity_hidden_size)

    def forward(self,
                left_context, left_context_lengths,
                right_context, right_context_lengths,
                entity_description, entity_description_lengths,
                entity):
        """
        forward.
        """
        # encoder_left_content = self.LeftContextEmbedsEncoder(left_context, left_context_lengths)
        # encoder_right_content = self.RightContextEmbedsEncoder(right_context, right_context_lengths)
        # encoder_entity_describe = self.EntityEmbedsEncoder(entity_description, entity_description_lengths)

        # encoder_left_content = self.multi_head_attention(left_context, left_context, left_context)
        # encoder_right_content = self.multi_head_attention(right_context, right_context, right_context)
        # encoder_entity_describe = self.multi_head_attention(entity_description, entity_description, entity_description)

        encoder_left_content = self.random_drop(left_context)
        encoder_right_content = self.random_drop(right_context)
        encoder_entity_describe = entity_description

        # 计算实体描述
        entity_description_vec = self.EntityEncoder(encoder_entity_describe, entity_description_lengths)
        entity_description_vec = self.norm1(entity_description_vec)

        # 实体连接向量
        # concat_entity_vec = torch.cat([entity_description_vec, entity.unsqueeze(1)], dim=2)

        # encoder_entity = self.entity_mlp(entity.unsqueeze(1))
        # encoder_entity = self.norm1(encoder_entity)
        # encoder_entity = self.max_pool(encoder_entity)
        # concat_entity_vec = torch.cat([entity_description_vec, encoder_entity], dim=2)
        # concat_entity_vec = self.entity_describe_mlp(concat_entity_vec)
        # concat_entity_vec = self.norm2(concat_entity_vec)

        concat_entity_vec = entity_description_vec

        # mention 上下文向量
        left_context_vec = self.ContextEncoder(encoder_left_content, left_context_lengths, query=concat_entity_vec)
        right_context_vec = self.ContextEncoder(encoder_right_content, right_context_lengths, query=concat_entity_vec)
        left_context_vec = self.norm1(left_context_vec)
        right_context_vec = self.norm1(right_context_vec)

        # left_context_vec = self.ContextEncoder(encoder_left_content, left_context_lengths)
        # right_context_vec = self.ContextEncoder(encoder_right_content, right_context_lengths)

        # left_context_vec = self.max_pool(encoder_left_content)
        # right_context_vec = self.max_pool(encoder_right_content)
        # concat_entity_vec = self.max_pool(encoder_entity_describe)

        # 上下文连接向量
        concat_context_vec = torch.cat([left_context_vec, right_context_vec], dim=2)
        # concat_context_vec = self.context_mlp(concat_context_vec)
        # concat_context_vec = self.norm2(concat_context_vec)
        # concat_context_vec = self.dropout(concat_context_vec)


        # 总连接向量
        concat_entity_context_vec = torch.cat([concat_context_vec, concat_entity_vec], dim=2)
        # concat_entity_context_vec = self.dropout(concat_entity_context_vec)



        # mlp
        out = self.out_liner_in(concat_entity_context_vec)
        out = torch.tanh(out)
        out = self.out_liner_out(out)
        result = torch.sigmoid(out)

        return result

    def random_drop(self, tensor: torch.Tensor, p1=0.2, p2=0.08):

        mix = torch.rand(tensor.size(0))
        eye = torch.eye(tensor.size(1))

        eye = torch.cat([eye.unsqueeze(0) for _ in range(tensor.size(0))], 0)

        for i in range(tensor.size(0)):
            if mix[i] < p1:
                eye[i] = get(tensor.size(1), p2)

        return torch.matmul(eye.cuda(), tensor)


