"""
Copyright (c) College of Mechatronics and Control Engineering, Shenzhen University.
All rights reserved.

Description :


Author：Team Li
"""

from agents.DQN_agents.base_conv_net.mobilenet_v2 import MobileNetV2
from torch import nn
import torch
import os
import numpy as np

# --------- total network -------------- #
class TimeSeqQNetwork(nn.Module):
    def __init__(self, n_action):
        super(TimeSeqQNetwork, self).__init__()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        dim_model = 512
        dim_key = dim_value = 64
        n_heads = 8
        dim_ff = 2048

        self.conv_encoder = conv_encoder(last_dim=512, pretrain=True)
        self.transformer_encoder = Encoder(n_layers=6, dim_model=dim_model, dim_key=dim_key, dim_value=dim_value, n_heads=n_heads, dim_ff=dim_ff)
        self.transformer_decoder = Decoder(n_layers=6, dim_model=dim_model, dim_key=dim_key, dim_value=dim_value, n_heads=n_heads, dim_ff=dim_ff)

        self.conv_encoder = self.conv_encoder.to(device)
        self.transformer_encoder = self.transformer_encoder.to(device)
        self.transformer_decoder = self.transformer_decoder.to(device)

        self.q_layer = nn.Linear(dim_model, n_action).to(device)


    def forward(self, input):
        """ ..
        Args:
            input: a time sep of imgs, shape is [bs, sep_num, h, w, c]
        """
        conv_embedding = self.conv_encoder(input)
        encoder_output, attentions = self.transformer_encoder(conv_embedding)

        query = encoder_output[:, -1, :].unsqueeze(dim=1)   ## 将最后一张图所代表的embedding作为查询，查询对应的决策
        out, _, _ = self.transformer_decoder(query, encoder_output)
        q_value = self.q_layer(out).squeeze()
        return q_value



# --------- total network -------------- #


# -------------------- transformer encoder ------------------------- #
class Encoder(nn.Module):
    def __init__(self, n_layers, dim_model, dim_key, dim_value, n_heads, dim_ff):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(dim_model, dim_key, dim_value, n_heads, dim_ff) for _ in range(n_layers)])

    def forward(self, enc_inputs): # enc_inputs : [batch_size x source_len]
        enc_outputs = enc_inputs
        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns


class EncoderLayer(nn.Module):
    def __init__(self, dim_model, dim_key, dim_value, n_heads, dim_ff):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(dim_model, dim_key, dim_value, n_heads)
        self.pos_ffn = PoswiseFeedForwardNet(dim_model, dim_ff)

    def forward(self, enc_inputs):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs) # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [batch_size x len_q x d_model]
        return enc_outputs, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, dim_model, dim_key, dim_value, n_heads):
        """ init
        Args:
            dim_model: is the input dim
        """
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(dim_model, dim_key * n_heads)
        self.W_K = nn.Linear(dim_model, dim_key * n_heads)
        self.W_V = nn.Linear(dim_model, dim_value * n_heads)
        self.linear = nn.Linear(n_heads * dim_value, dim_model)
        self.layer_norm = nn.LayerNorm(dim_model)

        self.dim_key = dim_key
        self.dim_value = dim_value
        self.dim_model = dim_model
        self.n_heads = n_heads

        self.ScaledDotProductAttention = ScaledDotProductAttention(dim_key)

    def forward(self, Q, K, V):
        # q: [batch_size x len_q x d_model], k: [batch_size x len_k x d_model], v: [batch_size x len_k x d_model]
        residual, batch_size = Q, Q.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q_s = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.dim_key).transpose(1,2)  # q_s: [batch_size x n_heads x len_q x d_k]
        k_s = self.W_K(K).view(batch_size, -1, self.n_heads, self.dim_key).transpose(1,2)  # k_s: [batch_size x n_heads x len_k x d_k]
        v_s = self.W_V(V).view(batch_size, -1, self.n_heads, self.dim_value).transpose(1,2)  # v_s: [batch_size x n_heads x len_k x d_v]

        # attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1) # attn_mask : [batch_size x n_heads x len_q x len_k]

        # context: [batch_size x n_heads x len_q x d_v], attn: [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        context, attn = self.ScaledDotProductAttention(q_s, k_s, v_s)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.dim_value) # context: [batch_size x len_q x n_heads * d_v]
        output = self.linear(context)
        return self.layer_norm(output + residual), attn # output: [batch_size x len_q x d_model]


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dim_key):
        super(ScaledDotProductAttention, self).__init__()
        self.dim_key = dim_key

    def forward(self, Q, K, V):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.dim_key) # scores : [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        # scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is one.
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, dim_model, dim_ff):
        super(PoswiseFeedForwardNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=dim_model, out_channels=dim_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=dim_ff, out_channels=dim_model, kernel_size=1)
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, inputs):
        residual = inputs # inputs : [batch_size, len_q, d_model]
        output = nn.ReLU()(self.conv1(inputs.transpose(1, 2)))
        output = self.conv2(output).transpose(1, 2)
        return self.layer_norm(output + residual)
# -------------------- transformer encoder ------------------------- #


# ---------------- transformer decoder -------------------------- #
class Decoder(nn.Module):
    def __init__(self, n_layers, dim_model, dim_key, dim_value, n_heads, dim_ff):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([DecoderLayer(dim_model, dim_key, dim_value, n_heads, dim_ff) for _ in range(n_layers)])

    def forward(self, dec_inputs, enc_outputs): # dec_inputs : [batch_size, target_len, dim_model]
        """"""
        dec_self_attns, dec_enc_attns = [], []
        dec_outputs = dec_inputs
        for layer in self.layers:
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        return dec_outputs, dec_self_attns, dec_enc_attns


class DecoderLayer(nn.Module):
    def __init__(self, dim_model, dim_key, dim_value, n_heads, dim_ff):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention(dim_model, dim_key, dim_value, n_heads)
        self.dec_enc_attn = MultiHeadAttention(dim_model, dim_key, dim_value,  n_heads)
        self.pos_ffn = PoswiseFeedForwardNet(dim_model, dim_ff)

    def forward(self, dec_inputs, enc_outputs):
        """ decoder layer
        Args:
            dec_inputs: decoder的input, 现在当成是最后一张图像的conv embedding, 视作query
            enc_outputs: encoder的输出
        """
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs)
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs)
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs, dec_self_attn, dec_enc_attn
# ---------------- transformer decoder -------------------------- #


# -------------------------- conv encoder --------------------------- #
class conv_encoder(nn.Module):
    """卷积的encoder, 负责处理时间序列图像"""
    def __init__(self, last_dim, pretrain=True, pretrain_file=os.path.join(os.path.dirname(__file__),
                                                                           '../base_conv_net/pretrain/mobilenetv2_0.35-b2e15951.pth')):
        super(conv_encoder, self).__init__()
        self.backbone = MobileNetV2(width_mult=0.35, n_dim=last_dim)
        if pretrain:
            self.load_pretrain(pretrain_file)
            print('load pretrain success...')

    def forward(self, input):
        """forward
        Args:
            input: time seq img input, shape is [bs, seq_num, h, w, c]
        Return:
            a tensor with shape [bs, seq_num, n_dim]
        """
        bs, seq_num, h, w, c = input.size()
        input = input.view(-1, h, w, c).transpose(1, 3)
        out = self.backbone(input).view(bs, seq_num, -1)
        return out

    def load_pretrain(self, pretrain_model_path):
        """load pretrain model"""
        net_dict = self.backbone.state_dict()
        if not torch.cuda.is_available():
            pretrain_dict = torch.load(pretrain_model_path, map_location='cpu')
        else:
            pretrain_dict = torch.load(pretrain_model_path)
        # print(net_dict.keys())
        # print(pretrain_dict.keys())

        load_dict = {(k): v for k, v in pretrain_dict.items() if
                     (k) in net_dict}
        # print(load_dict.keys())

        # last_conv_layer_net_dict_name = ['features.17.0.weight', 'features.17.1.weight', 'features.17.1.bias',
        #                                  'features.17.1.running_mean', 'features.17.1.running_var', 'features.17.1.num_batches_tracked']
        # last_conv_layer_pretrain_dict_name = ['conv.0.weight', 'conv.1.weight', 'conv.1.bias', 'conv.1.running_mean',
        #                                       'conv.1.running_var', 'conv.1.num_batches_tracked']
        #
        # for net_dict_key, pretrain_dict_key in zip(last_conv_layer_net_dict_name, last_conv_layer_pretrain_dict_name):
        #     if net_dict[net_dict_key].size() == pretrain_dict[pretrain_dict_key].size():
        #         load_dict[net_dict_key] = pretrain_dict[pretrain_dict_key]
        #         print(net_dict_key, '  ok')
        #     else:
        #         print(net_dict[net_dict_key].size(), pretrain_dict[pretrain_dict_key].size())

        net_dict.update(load_dict)
        self.backbone.load_state_dict(net_dict, strict=True)
        print(f'load keys:{load_dict.keys()}')
        # self.logger.info(f'load keys:{load_dict.keys()}')
# ------------------- conv encoder ---------------------- #


if __name__ == '__main__':
    import time

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # conv_encoder = conv_encoder(last_dim=512, pretrain=True)
    # conv_encoder = conv_encoder.to(device)
    #
    # transformer_encoder = Encoder(n_layers=6, dim_model=512, dim_key=64, dim_value=64, n_heads=8, dim_ff=2048)
    # transformer_encoder = transformer_encoder.to(device)
    #
    # transformer_decoder = Decoder(n_layers=6, dim_model=512, dim_key=64, dim_value=64, n_heads=8, dim_ff=2048)
    # transformer_decoder = transformer_decoder.to(device)
    #
    # input = torch.rand(size=(64, 5, 224, 224, 3)).to(device)
    #
    # # for i in range(100):
    # #     time0 = time.time()
    # #     out = conv_encoder(input)
    # #     print('%f'%(time.time() - time0))
    # # pass
    #
    # for i in range(100):
    #     time0 = time.time()
    #     conv_embedding = conv_encoder(input)
    #     encoder_output, attentions = transformer_encoder(conv_embedding)
    #     a, b, c = transformer_decoder(encoder_output[:, -1, :].unsqueeze(dim=1), encoder_output)
    #     print(time.time() - time0)


    q_network = TimeSeqQNetwork(n_action=5)
    input = torch.rand(size=(64, 5, 224, 224, 3)).to(device)
    q_value = q_network(input)
    pass



