import torch
from torch import nn
import numpy as np
import copy
import torch.nn.functional as F



embedding_dim = 300
need_pretrain_embedding = True
pad_size = 32
dropout = 0.5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_head = 6
hidden_dim = 512
pad_size = 20
num_classes = 64
num_encoder = 6



class Transformer(nn.Module):
    def __init__(self, pre_train_embed_path):
        super(Transformer, self).__init__()
        self.output_dim = num_classes

        PRE_TRAIN_WORD_EMBEDDING = torch.tensor(np.load(pre_train_embed_path)["embeddings"].astype('float32'))
        if need_pretrain_embedding:
            self.embedding = nn.Embedding.from_pretrained(PRE_TRAIN_WORD_EMBEDDING, freeze=False)
        else:
            pass
            # TODO: init embedding

        self.postion_embedding = Positional_Encoding(embedding_dim, pad_size, dropout, device)
        self.encoder = Encoder(embedding_dim, hidden_dim, dropout)
        self.encoders = nn.ModuleList([copy.deepcopy(self.encoder) for _ in range(num_encoder)])

        self.fc1 = nn.Linear(pad_size * embedding_dim, num_classes)

    def forward(self, x):
        out = self.embedding(x[0])
        out = self.postion_embedding(out)
        for encoder in self.encoders:
            out = encoder(out)
        out = out.view(out.size(0), -1)
        # out = torch.mean(out, 1)
        out = self.fc1(out)
        return out


class Encoder(nn.Module):
    def __init__(self, dim_model, hidden_dim, dropout):
        super(Encoder, self).__init__()
        self.attention = Multi_Head_Attention(dim_model, dropout)
        self.feed_forward = Position_wise_Feed_Forward(dim_model, hidden_dim, dropout)

    def forward(self, x):
        out = self.attention(x)
        out = self.feed_forward(out)
        return out


class Positional_Encoding(nn.Module):
    def __init__(self, embed_size, pad_size, dropout, device):
        super(Positional_Encoding, self).__init__()
        self.device = device
        self.pe = torch.tensor([[pos / (10000.0 ** (i // 2 * 2.0 / embed_size)) for i in range(embed_size)] for pos in range(pad_size)])
        self.pe[:, 0::2] = np.sin(self.pe[:, 0::2])
        self.pe[:, 1::2] = np.cos(self.pe[:, 1::2])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = x + nn.Parameter(self.pe, requires_grad=False).to(self.device)
        out = self.dropout(out)
        return out


class Scaled_Dot_Product_Attention(nn.Module):
    def __init__(self):
        super(Scaled_Dot_Product_Attention, self).__init__()

    def forward(self, Q, K, V, scale=None):
        '''
        Args:
            Q: [batch_size, len_Q, dim_Q]
            K: [batch_size, len_K, dim_K]
            V: [batch_size, len_V, dim_V]
            scale: 缩放因子 论文为根号dim_K
        Return:
            self-attention后的张量，以及attention张量
        '''
        attention = torch.matmul(Q, K.permute(0, 2, 1))
        if scale:
            attention = attention * scale
        # if mask:  # TODO change this
        #     attention = attention.masked_fill_(mask == 0, -1e9)
        attention = F.softmax(attention, dim=-1)
        context = torch.matmul(attention, V)
        return context


class Multi_Head_Attention(nn.Module):
    def __init__(self, dim_model, dropout=0.0):
        super(Multi_Head_Attention, self).__init__()
        assert dim_model % num_head == 0
        self.dim_head = dim_model // num_head
        self.fc_Q = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_K = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_V = nn.Linear(dim_model, num_head * self.dim_head)
        self.attention = Scaled_Dot_Product_Attention()
        self.fc = nn.Linear(num_head * self.dim_head, dim_model)
        self.dropout = nn.Dropout(dropout)

        self.layer_norm = nn.LayerNorm(dim_model)
        # self.layer_norm = nn.LayerNorm([dim_model, 1])

    def forward(self, x):
        batch_size = x.size(0)
        Q = self.fc_Q(x)
        K = self.fc_K(x)
        V = self.fc_V(x)

        # sentence len 不变
        Q = Q.view(batch_size * num_head, -1, self.dim_head)
        K = K.view(batch_size * num_head, -1, self.dim_head)
        V = V.view(batch_size * num_head, -1, self.dim_head)

        # https://blog.csdn.net/qq_40210472/article/details/88826821
        # if mask:  # TODO
        #     mask = mask.repeat(self.num_head, 1, 1)  # TODO change this
        scale = K.size(-1) ** -0.5  # 缩放因子
        context = self.attention(Q, K, V, scale)

        context = context.view(batch_size, -1, self.dim_head * num_head)
        out = self.fc(context)
        out = self.dropout(out)
        out = out + x  # 残差连接
        # out = self.layer_norm(out)
        return out


class Position_wise_Feed_Forward(nn.Module):
    def __init__(self, dim_model, hidden_dim, dropout=0.0):
        super(Position_wise_Feed_Forward, self).__init__()
        self.fc1 = nn.Linear(dim_model, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = out + x  # 残差连接
        out = self.layer_norm(out)
        return out
