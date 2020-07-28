import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained_bert.modeling import BertModel, BertPreTrainedModel
from torch.nn import CrossEntropyLoss


class BertForQuestionAnswering(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForQuestionAnswering, self).__init__(config)
        self.bert = BertModel(config)
        # TODO check with Google if it's normal there is no dropout on the token classifier of SQuAD in the TF version
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.activation = nn.ReLU()
        self.head_num = 12

        self.mix_lambda = nn.Parameter(torch.tensor(0.5))

        self.linear_q = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.linear_k = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.linear_v = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.linear_o = nn.Linear(config.hidden_size, config.hidden_size, bias=True)

        self.linear_q2 = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.linear_k2 = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.linear_v2 = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.linear_o2 = nn.Linear(config.hidden_size, config.hidden_size, bias=True)

        self.linear_q3 = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.linear_k3 = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.linear_v3 = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.linear_o3 = nn.Linear(config.hidden_size, config.hidden_size, bias=True)

        self.ensemble_linear = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.ensemble_activation = nn.Tanh()

        self.qa_outputs = nn.Linear(config.hidden_size, 2)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, word_length_1, word_length_2, word_length_3, word_slice_1, word_slice_2, word_slice_3,
                token_type_ids=None, attention_mask=None, start_positions=None, end_positions=None):

        encoded_layers, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)

        seg_attention_out = self.MultiHeadSegATT(encoded_layers, encoded_layers, encoded_layers, self.linear_q,
                                                 self.linear_k, self.linear_v, word_length_1, word_slice_1)

        seg_attention_out2 = self.MultiHeadSegATT(encoded_layers, encoded_layers, encoded_layers, self.linear_q2,
                                                  self.linear_k2, self.linear_v2, word_length_2, word_slice_2)

        seg_attention_out3 = self.MultiHeadSegATT(encoded_layers, encoded_layers, encoded_layers, self.linear_q3,
                                                  self.linear_k3, self.linear_v3, word_length_3, word_slice_3)

        # tricky way to ensemble by character position.
        batch, seqlen, hidden = input_ids.size(0), input_ids.size(1), self.config.hidden_size
        sequence_output = torch.autograd.Variable(torch.zeros([batch, seqlen, hidden])).to(input_ids.device)
        for i in range(seqlen):
            att1 = self.ensemble_activation(self.ensemble_linear(seg_attention_out[:, i, :]))
            att2 = self.ensemble_activation(self.ensemble_linear(seg_attention_out3[:, i, :]))
            att3 = self.ensemble_activation(self.ensemble_linear(seg_attention_out2[:, i, :]))
            att4 = self.ensemble_activation(self.ensemble_linear(encoded_layers[:, i, :]))
            sequence_output[:, i, :] = att1 + att2 + att3 + att4

        # sequence_output = seg_attention_out2 # For ablation experiment

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            return total_loss
        else:
            return start_logits, end_logits

    def MultiHeadSegATT(self, q, k, v, Q, K, V, word_lengths, word_slice_indexs):
        q, k, v = Q(q), K(k), V(v)

        q = self.activation(q)
        k = self.activation(k)
        v = self.activation(v)

        q = self._reshape_to_batches(q)
        k = self._reshape_to_batches(k)
        v = self._reshape_to_batches(v)  # batch_size, head_num, seq_len, sub_dim

        dk = q.size()[-1]
        scores = q @ (k.transpose(-2, -1)) / math.sqrt(dk)  # batch_size, head_num, seq_len_row, seq_len_col

        attention = F.softmax(scores, dim=-1)

        scalar = self.calculate_scale(attention.detach(), word_slice_indexs, word_lengths)

        y = attention * scalar @ v  # applying aligned attention

        y = self._reshape_from_batches(y)
        y = self.linear_o(y)
        y = self.activation(y)
        return y

    def _reshape_to_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        sub_dim = in_feature // self.head_num
        return x.reshape(batch_size, seq_len, self.head_num, sub_dim) \
            .permute(0, 2, 1, 3)  # batch_size, head_num, seq_len, sub_dim

    def _reshape_from_batches(self, x):
        batch_size, head_num, seq_len, sub_dim = x.size()
        out_dim = head_num * sub_dim
        return x.permute(0, 2, 1, 3).reshape(batch_size, seq_len, out_dim)

    def extra_repr(self):
        return 'in_features={}, head_num={}, bias={}, activation={}'.format(
            self.in_features, self.head_num, self.bias, self.activation,
        )

    def calculate_scale(self, att_weights, seg_slice, seg_length):
        batch_size, head_num, seq_len_row, seq_len_col = att_weights.size()
        batch_size = int(batch_size)
        mask = torch.zeros(att_weights.size()).to(seg_length.device)
        # iterate till encounter padding tag, for early stopping and accelerate.
        stop_condition = (seg_length != 0).sum(dim=1)
        for batch_idx in range(batch_size):
            if att_weights[batch_idx].nelement() == 0:
                continue
            for s in range(int(stop_condition[batch_idx])):
                token_pos = seg_slice[batch_idx][s]
                token_length = seg_length[batch_idx][s]
                if token_pos > stop_condition[batch_idx]:
                    break
                if bool(token_length > 1):
                    att = att_weights[batch_idx, :, :, token_pos: token_pos + token_length]
                    if att.nelement() == 0:
                        continue
                    mean = att.mean(-1, keepdim=True)  # .repeat(att.size(0))
                    max = att.max(-1, keepdim=True)[0]
                    # try to make attention more balanced
                    # mean = mean * (att <= mean).float() + att * (att > mean).float()
                    mix = max * self.mix_lambda + mean * (torch.tensor(1).to(seg_length.device) - self.mix_lambda)
                    mask[batch_idx, :, :, token_pos: token_pos + token_length] = mix / att
                else:
                    mask[batch_idx, :, :, token_pos: token_pos + token_length] = \
                        torch.ones([head_num, seq_len_row, token_length])
        return mask
