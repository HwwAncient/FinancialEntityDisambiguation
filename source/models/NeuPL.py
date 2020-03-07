from source.models.base_model import *
from source.modules.encoders.encoder import *
import torch.nn as nn
import torch


class NeuPL(BaseModel):

    def __init__(self,
                 embedding_dim,
                 hidden_size,
                 mlp_layer_num,
                 mlp_hidden_size,
                 mlp_bias=True,
                 encoder_model='GRU',
                 dropout=0
                 ):
        super(NeuPL, self).__init__()
        self.mlp_bias = mlp_bias
        # self.entity_vec_dim = int(hidden_size/2 + embedding_dim)
        self.entity_vec_dim = int(hidden_size / 2)
        self.context_vec_dim = hidden_size
        self.mlp_layers = [self.entity_vec_dim+self.context_vec_dim] + \
                          [mlp_hidden_size for _ in range(mlp_layer_num)] + [1]

        self.EntityEncoder = RnnEncoder(
            embedding_dim=embedding_dim,
            hidden_size=hidden_size,
            model=encoder_model)

        self.ContextEncoder = RnnEncoder(
            embedding_dim=embedding_dim,
            hidden_size=hidden_size,
            query=True,
            query_size=self.entity_vec_dim,
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

        self.dropout = nn.Dropout(dropout)


    def forward(self,
                left_context, left_context_lengths,
                right_context, right_context_lengths,
                entity_description, entity_description_lengths,
                entity):
        """
        forward.
        """
        # 计算实体描述
        entity_description_vec = self.EntityEncoder(entity_description, entity_description_lengths)

        # 实体连接向量
        # concat_entity_vec = torch.cat([entity_description_vec, entity.unsqueeze(1)], dim=2)
        concat_entity_vec = entity_description_vec

        # mention 上下文向量
        left_context_vec = self.ContextEncoder(left_context, left_context_lengths, query=concat_entity_vec)
        right_context_vec = self.ContextEncoder(right_context, right_context_lengths, query=concat_entity_vec)

        # 上下文连接向量
        concat_context_vec = torch.cat([left_context_vec, right_context_vec], dim=2)

        # 总连接向量
        concat_entity_context_vec = torch.cat([concat_context_vec, concat_entity_vec], dim=2)

        # mlp
        out = self.out_liner_in(concat_entity_context_vec)
        out = self.dropout(out)
        out = torch.tanh(out)
        out = self.out_liner_out(out)

        result = torch.sigmoid(out)

        return result

