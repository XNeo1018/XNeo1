"""
attention机制
"""

import torch.nn as nn
import torch
import torch.nn.functional as F
import config

class Attention(nn.Module):
    def __init__(self, method="general"):
        super(Attention, self).__init__()
        self.method = method
        assert self.method in ["general", "dot", "concat"], "moedl error! should in 'general', 'dot', 'concat'"

        if self.method == "general":
            # 注意bias为False的时候才能表示矩阵乘法变换
            self.Wa = nn.Linear(config.decoder_hidden_size,config.encoder_hidden_size, bias=False)
        elif self.method == "concat":
            self.Wa = nn.Linear(config.encoder_hidden_size+config.decoder_hidden_size,config.decoder_hidden_size, bias=False)
            self.Va = nn.Linear(config.decoder_hidden_size,1, bias=False)

    def forward(self, encoder_outputs, decoder_hidden):
        """

        :param encoder_outputs: size [batch_size,seq_len,encoder_hidden_size]
        :param decoder_hidden: size [num_layer*bio,batch_size,decoder_hidden_size]
        :return:
        """
        # 根据指定的model计算match，如果是dot方法：
        if self.method == "dot":
            assert encoder_outputs.size(-1) == decoder_hidden.size(-1), "method 'dot' decoder_hidden_size not same as encoder"
            # decoder_hidden = decoder_hidden[-1,:,:].permute(1,2,0)  # [batch_size,decoder_hidden_size,1]
            # attention_weights = encoder_outputs.bmm(decoder_hidden).squeeze(-1)  # [batch_size,seq_len]
            # attention_weights = F.log_softmax(attention_weights,dim=-1)  # [batch_size,seq_len]
            batch_size, seq_len, encoder_hidden_size = encoder_outputs.size()
            attention_weights = torch.zeros([batch_size,seq_len])
            decoder_hidden = decoder_hidden[-1]  # [batch_size,decoder_hidden_size]
            for i in range(batch_size):
                for j in range(seq_len):
                    attention_weights[i,j] = decoder_hidden[i].dot(encoder_hidden_size[i,j])
            attention_weights = F.log_softmax(attention_weights,dim=-1)

        # 如果是general方法：
        elif self.method == "general":
            # print(encoder_outputs)
            # encoder_outputs = self.Wa(encoder_outputs)  # [batch_size,seq_len,decoder_hidden_size]
            # decoder_hidden = decoder_hidden[-1].unsqueeze(-1)  # [batch_size,decoder_hidden_size,1]
            # attention_weights = encoder_outputs.bmm(decoder_hidden).squeeze(-1)  # [batch_size,seq_len]
            # attention_weights = F.log_softmax(attention_weights, dim=-1)  # [batch_size,seq_len]

            # 对decoder_hidden进行线性变换
            decoder_hidden = decoder_hidden[-1]  # [batch_size,decoder_hidden_size]
            # encoder_max_len = encoder_outputs.size(-1)
            # decoder_hidden.repeat(1,1,encoder_max_len) # [batch_size,decoder_hidden_size,max_len]
            decoder_hidden = self.Wa(decoder_hidden).unsqueeze(-1)  # [batch_size,encoder_hidden_size, 1]
            attention_weights = torch.bmm(encoder_outputs,decoder_hidden).squeeze(-1)  # [batch_size,seq_len]
            attention_weights = F.log_softmax(attention_weights, dim=-1)  # [batch_size,seq_len]

        elif self.method == "concat":
            decoder_hidden = decoder_hidden[-1]  # [batch_size,decoder_hidden_size]
            seq_len = encoder_outputs.size(1)
            decoder_hidden = decoder_hidden.repeat(1,seq_len,1)  # [batch_size,seq_len, decoder_hidden_size]
            concated = torch.concat([encoder_outputs, decoder_hidden],dim=-1)  # [batch_size,seq_len, decoder_hidden_size+encoder_hidden_size]
            concated = concated.view(-1,config.decoder_hidden_size+config.encoder_hidden_size)  # [batch_size*seq_len, decoder_hidden_size+encoder_hidden_size]
            attention_weights = self.Va(torch.tanh(self.Wa(concated)))  # [batch_size*seq_len,1]
            attention_weights = attention_weights.squeeze(-1).view(-1,seq_len)  # [batch_size, seq_len]

            attention_weights = F.log_softmax(attention_weights, dim=-1)  # [batch_size,seq_len]

        return attention_weights

