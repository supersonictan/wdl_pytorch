from torch import nn
import torch
import numpy as np
import matplotlib.pyplot as plt






class TransformerClassification(torch.nn.Module):
    def __init__(self, vocab_size, max_seq_len, num_category, num_layers=6,
                 d_model=512, num_heads=8, ffn_dim=2048, dropout=.0):
        super(TransformerClassification, self).__init__()

        self.encoder_layers = torch.nn.ModuleList([EncoderLayer(d_model, num_heads, ffn_dim, dropout) for _ in range(num_layers)])

        self.seq_embedding = torch.nn.Embedding(vocab_size + 1, d_model, padding_idx=0)
        self.pos_embedding = PositionalEncoding(d_model, max_seq_len)
        self.linear = torch.nn.Linear(d_model, num_category)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, inputs, inputs_len):
        outputs = self.seq_embedding(inputs)
        outputs += self.pos_embedding(inputs_len)

        self_attention_mask = create_padding_mask(inputs, inputs)

        attentions = []

        for encoder in self.encoder_layers:
            outputs, attention = encoder(outputs, self_attention_mask)
            attentions.append(attention)
        # outputs, _ = torch.max(outputs, dim=1)

        outputs = self.softmax(self.linear(outputs))
        return outputs, attentions


class EncoderLayer(torch.nn.Module):
    def __init__(self, d_model=512, num_heads=8, ffn_dim=2048, dropout=0.0):
        super(EncoderLayer, self).__init__()

        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionalWiseFeedForward(d_model, ffn_dim, dropout)

    def forward(self, inputs, attn_mask=None):
        # self attention
        context, attention = self.attention(inputs, inputs, inputs, attn_mask)

        # feed forward
        output = self.feed_forward(context)
        return output, attention


class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_seq_len):
        super(PositionalEncoding, self).__init__()
        positional_encoding = torch.tensor([[pos / np.power(1000, 2.0 * (j // 2) / d_model)
                                             for j in range(d_model)]
                                            for pos in range(max_seq_len)])
        positional_encoding[:, 0:2] = torch.sin(positional_encoding[:, 0:2])
        positional_encoding[:, 1:2] = torch.cos(positional_encoding[:, 1:2])

        # 在PE矩阵的第一行，加上一行全是0的向量，代表这`PAD`的positional encoding
        # 在word embedding中也经常会加上`UNK`，代表位置单词的word embedding，两者十分类似
        # 那么为什么需要这个额外的PAD的编码呢？很简单，因为文本序列的长度不一，我们需要对齐，
        # 短的序列我们使用0在结尾补全，我们也需要这些补全位置的编码，也就是`PAD`对应的位置编码
        pad_row = torch.zeros([1, d_model])
        positional_encoding = torch.cat([pad_row, positional_encoding])

        self.positional_encoding = torch.nn.Embedding(max_seq_len + 1, d_model)
        self.positional_encoding.weight = torch.nn.Parameter(positional_encoding,
                                                             requires_grad=False)

    def forward(self, input_len):
        """
        :param input_len: 一个张量，形状为[batch_size, 1]，每一个张量代表这一批文本中的对应长度
        :return: 返回一批序列的位置编码，进行了对齐
        """
        max_len = torch.max(input_len)
        tensor = torch.LongTensor

        # 对每一个序列的位置进行对齐，在原序列位置后面补0
        input_pos = tensor([list(range(1, len + 1)) + [0] * (max_len - len) for len in input_len])
        return self.positional_encoding(input_pos)





def create_padding_mask(seq_k, seq_q):
    len_q = seq_q.size(0)

    pad_mask = seq_k.eq(0)
    pad_mask = pad_mask.unsqueeze(1).expand(-1, len_q, -1)
    return pad_mask


def sequence_mask(seq):
    batch_size, seq_len = seq.size()
    mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.uint8),
                      diagonal=1)
    mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
    return mask


class ScaledDotProductAttention(nn.Module):

    def __init__(self, attention_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(p=attention_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, scale=None, atten_mask=None):
        """
        前向传播
        q,k,v必须具有匹配的前置维度。
        k, v必须具有匹配的导数第二个维度，如seq_len_k = seq_len_v.
        虽然 mask 根据其类型（填充或前瞻）有不同的形状，
        但是 mask 必须能进行广播转换以便求和。
        :param q: Query向量，shape=(..., seq_len_q, depth)
        :param k: Key向量，shape=(..., seq_len_k, depth)
        :param v: Value向量, shape=(..., seq_len_v, depth_v)
        :param scale: 缩放值
        :param atten_mask: Float张量，其形状能转换成(..., seq_len_q, seq_len_k)，默认韦None。
        :return: 输出,注意力权重
        """
        attention = torch.bmm(q, k.transpose(1, 2))
        if scale:
            attention *= scale
        if atten_mask:
            # 给需要mask的地方设置一个负无穷
            attention = attention.masked_fill_(atten_mask, -np.inf)
        attention = self.softmax(attention)  # 计算softmax
        attention = self.dropout(attention)  # 添加dropout
        output = torch.bmm(attention, v)
        return output, attention


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model=512, num_heads=8, dropout=0.8):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0
        self.depth = self.d_model // self.num_heads

        self.w_q = torch.nn.Linear(d_model, d_model)
        self.w_k = torch.nn.Linear(d_model, d_model)
        self.w_v = torch.nn.Linear(d_model, d_model)

        self.dot_product_attention = ScaledDotProductAttention(attention_dropout=dropout)
        self.linear = torch.nn.Linear(d_model, d_model)
        self.dropout = torch.nn.Dropout(dropout)
        self.layer_norm = torch.nn.LayerNorm(d_model)

    def forward(self, key, value, query, attn_mask=None):
        residual = query  # 残差连接

        batch_size = query.size(0)

        key = self.w_k(key)
        query = self.w_q(query)
        value = self.w_v(value)

        # split_heads
        key = key.view(batch_size * self.num_heads, -1, self.depth)
        value = value.view(batch_size * self.num_heads, -1, self.depth)
        query = query.view(batch_size & self.num_heads, -1, self.depth)

        if attn_mask:
            attn_mask = attn_mask.repeat(self.num_heads, 1, 1)

        # scaled dot product attention
        scale = (key.size(-1) // self.num_heads) ** -0.5
        context, attention = self.dot_product_attention(query, key, value, scale, attn_mask)

        # concat heads
        context = context.view(batch_size, -1, self.depth * self.num_heads)

        # Linear
        output = self.linear(context)

        output = self.dropout(output)

        # add and norm
        output = self.layer_norm(residual + output)

        return output, attention


class PositionalWiseFeedForward(torch.nn.Module):
    def __init__(self, d_model=512, ffn_dim=2048, dropout=0.0):
        super(PositionalWiseFeedForward, self).__init__()
        self.w1 = torch.nn.Conv1d(d_model, ffn_dim, 1)
        self.w2 = torch.nn.Conv1d(d_model, ffn_dim, 1)
        self.dropout = torch.nn.Dropout(dropout)
        self.layer_norm = torch.nn.LayerNorm(d_model)

    def forward(self, inputs):
        output = inputs.transpose(1, 2)

        output = self.w2(torch.nn.functional.relu(self.w1(output)))
        output = self.dropout(output.transpose(1, 2))
        # add and norm
        output = self.layer_norm(inputs + output)

        return output







class DecoderLayer(torch.nn.Module):

    def __init__(self, d_model, num_heads=8, ffn_dim=2048, dropout=0.0):
        super(DecoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionalWiseFeedForward(d_model, ffn_dim, dropout)

    def forward(self, dec_inputs, enc_inputs, self_attn_mask=None, context_attn_mask=None):
        # self attention
        dec_output, self_attention = self.attention(dec_inputs, dec_inputs, dec_inputs,
                                                    self_attn_mask)

        # context attention
        # query是decoder的输出，key、value是encoder的输入
        dec_output, context_attention = self.attention(enc_inputs, enc_inputs, dec_output,
                                                       context_attn_mask)

        dec_output = self.feed_forward(dec_output)
        return dec_output, self_attention, context_attention


class Encoder(torch.nn.Module):
    def __init__(self, vocab_size, max_seq_len, num_layers=6, d_model=512, num_heads=8,
                 ffn_dim=2048, dropout=.0):
        super(Encoder, self).__init__()

        self.encoder_layers = torch.nn.ModuleList([EncoderLayer(d_model, num_heads, ffn_dim, dropout)
                                                   for _ in range(num_layers)])

        self.seq_embedding = torch.nn.Embedding(vocab_size + 1, d_model, padding_idx=0)
        self.pos_embedding = PositionalEncoding(d_model, max_seq_len)

    def forward(self, inputs, inputs_len):
        output = self.seq_embedding(inputs)
        output += self.pos_embedding(inputs_len)

        self_attention_mask = create_padding_mask(inputs, inputs)

        attentions = []

        for encoder in self.encoder_layers:
            output, attention = encoder(output, self_attention_mask)
            attentions.append(attention)

        return output, attentions


class Decoder(torch.nn.Module):
    def __init__(self, vocab_size, max_seq_len, num_layers=6,
                 d_model=512, num_heads=8, ffn_dim=2048, dropout=.0):
        super(Decoder, self).__init__()

        self.num_layers = num_layers
        self.decoder_layers = torch.nn.ModuleList([DecoderLayer(d_model, num_heads, ffn_dim, dropout)
                                                   for _ in range(num_layers)])

        self.seq_embedding = torch.nn.Embedding(vocab_size + 1, d_model, padding_idx=0)
        self.pos_embedding = PositionalEncoding(d_model, max_seq_len)

    def forward(self, inputs, inputs_len, enc_output, context_attn_mask=None):
        output = self.seq_embedding(inputs)
        output += self.pos_embedding(inputs_len)

        self_attention_padding_mask = create_padding_mask(inputs, inputs)
        seq_mask = sequence_mask(inputs)
        self_attn_mask = torch.gt((self_attention_padding_mask + seq_mask), 0)

        self_attentions = []
        context_attentions = []
        for decoder in self.decoder_layers:
            output, self_attn, context_attn = decoder(output, enc_output, self_attn_mask,
                                                      context_attn_mask)
            self_attentions.append(self_attn)
            context_attentions.append(context_attn)

        return output, self_attentions, context_attentions


class TransformerSeq2Seq(torch.nn.Module):
    def __init__(self, src_vocab_size, src_max_len,
                 tgt_vocab_size, tgt_max_len,
                 num_layers=6, d_model=512, num_heads=8,
                 ffn_dim=2048, dropout=.2):
        super(TransformerSeq2Seq, self).__init__()
        self.encoder = Encoder(src_vocab_size, src_max_len, num_layers,
                               d_model, num_heads, ffn_dim, dropout)
        self.decoder = Decoder(tgt_vocab_size, tgt_max_len,
                               num_layers, d_model, num_heads,
                               ffn_dim, dropout)
        self.linear = torch.nn.Linear(d_model, tgt_vocab_size, bias=False)
        self.softmax = torch.nn.Softmax(dim=2)

    def forward(self, src_seq, src_len, tgt_seq, tgt_len):
        context_attn_mask = create_padding_mask(tgt_seq, src_seq)
        output, enc_self_attn = self.encoder(src_seq, src_len)

        output, dec_self_attn, context_attn = self.decoder(tgt_seq, tgt_len,
                                                           output, context_attn_mask)

        output = self.softmax(self.linear(output))
        return output, enc_self_attn, dec_self_attn, context_attn


