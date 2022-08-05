import math
import torch
import logging
import traceback
import numpy as np
from torch import nn

import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import MarginRankingLoss
from ..transformers import BertPreTrainedModel, RobertaModel

from itertools import repeat
from torch._six import container_abcs
from typing import List
import geoopt as gt
import torch
from torch import nn
from ..hyperbolic.poincare import PoincareBall
from ..hyperbolic.mobius_linear import MobiusLinear

logger = logging.getLogger()


# -------------------------------------------------------------------------------------------
# Modified CNN
# -------------------------------------------------------------------------------------------

def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


_single = _ntuple(1)


class _ConvNd(nn.Module):
    __constants__ = ['stride', 'padding', 'dilation', 'groups', 'bias',
                     'padding_mode', 'output_padding', 'in_channels',
                     'out_channels', 'kernel_size']

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding,
                 groups, bias, padding_mode):
        super(_ConvNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        self.padding_mode = padding_mode
        if transposed:
            self.weight = Parameter(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size))
        else:
            self.weight = Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        return s.format(**self.__dict__)

    def __setstate__(self, state):
        super(_ConvNd, self).__setstate__(state)
        if not hasattr(self, 'padding_mode'):
            self.padding_mode = 'zeros'


class Conv1d(_ConvNd):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        kernel_size = _single(kernel_size)
        stride = _single(stride)
        padding = _single(padding)
        dilation = _single(dilation)
        super(Conv1d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _single(0), groups, bias, padding_mode)

    def forward(self, input):
        input = input.transpose(1, 2)

        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[0] + 1) // 2, self.padding[0] // 2)
            return F.conv1d(F.pad(input, expanded_padding, mode='circular'),
                            self.weight, self.bias, self.stride,
                            _single(0), self.dilation, self.groups)

        output = F.conv1d(input, self.weight, self.bias, self.stride,
                          self.padding, self.dilation, self.groups)
        output = output.transpose(1, 2)
        return output


# -------------------------------------------------------------------------------------------
# CnnGram Extractor
# -------------------------------------------------------------------------------------------

class NGramers(nn.Module):
    def __init__(self, input_size, hidden_size, max_gram, dropout_rate):
        super().__init__()

        self.cnn_list = nn.ModuleList([nn.Conv1d(in_channels=input_size,
                                                 out_channels=hidden_size,
                                                 kernel_size=n) for n in range(1, max_gram + 1)])
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = x.transpose(1, 2)

        cnn_outpus = []
        for cnn in self.cnn_list:
            y = cnn(x)
            y = self.relu(y)
            y = self.dropout(y)
            cnn_outpus.append(y.transpose(1, 2))
        outputs = torch.cat(cnn_outpus, dim=1)
        return outputs


# -------------------------------------------------------------------------------------------
# Inherit BertPreTrainedModel
# -------------------------------------------------------------------------------------------
class RobertaForCnnGramRanking(BertPreTrainedModel):

    def __init__(self, config):
        super(RobertaForCnnGramRanking, self).__init__(config)

        max_gram = 5
        cnn_output_size = 768
        cnn_dropout_rate = (config.hidden_dropout_prob / 2)
        # RobertaModel
        self.roberta = RobertaModel(config)
        self.cnn2gram = NGramers(input_size=config.hidden_size,
                                 hidden_size=cnn_output_size,
                                 max_gram=max_gram,
                                 dropout_rate=cnn_dropout_rate)

        self.classifier = nn.Linear(cnn_output_size, 1)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.sigmoid = nn.Sigmoid()
        self.init_weights()
        self.scalar_layer = AdaptiveMixingLayer()
        self.rank_value = cnn_output_size
        #self.hyper_rank = nn.Linear(768, cnn_output_size, bias=False)
        self.hyper_rank = nn.Parameter(torch.Tensor(config.hidden_size, cnn_output_size))
        nn.init.kaiming_normal_(self.hyper_rank, mode='fan_in', nonlinearity='relu')
        self.p_ball = gt.PoincareBall()
        self.c = 1
        self.hyper_linear = MobiusLinear(manifold=PoincareBall(), in_features=cnn_output_size, out_features=1, c=self.c)
        self.min_norm = 1e-15
        self.eps = {torch.float32: 4e-3, torch.float64: 1e-5}
        #self.hyper_linear2 = MobiusLinear(manifold=self.p_ball, in_features=256, out_features=1, c=self.c)
# -------------------------------------------------------------------------------------------
# RobertaForTFRanking
# -------------------------------------------------------------------------------------------
class RobertaForTFRanking(RobertaForCnnGramRanking):


    def forward(self, input_ids, attention_mask, valid_ids, active_mask, valid_output, labels=None):
        # --------------------------------------------------------------------------------
        # Embedding
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        outputs_final = torch.stack(outputs[2])[1:13]
        sequence_output = self.scalar_layer(outputs_final)
        batch_size = sequence_output.size(0)
        for i in range(batch_size):
            valid_num = sum(valid_ids[i]).item()
            vectors = sequence_output[i][valid_ids[i] == 1]
            valid_output[i, :valid_num].copy_(vectors)
        sequence_output = self.dropout(valid_output)
        hyper_sequence_output = self.p_ball.expmap0(torch.matmul(sequence_output, self.hyper_rank))
        mean_document = self.einstein_midpoint(hyper_sequence_output).unsqueeze(1)
        ngrams_outputs = self.p_ball.expmap0(self.cnn2gram(sequence_output))
        # --------------------------------------------------------------------------------
        # Importance Scores
        #classifier_scores1 = (-self.p_ball.dist2(ngrams_outputs, mean_document.expand_as(ngrams_outputs)) / math.sqrt(self.rank_value))
        classifier_scores1 = (-self.p_ball.dist(ngrams_outputs, mean_document.expand_as(ngrams_outputs)) / math.sqrt(self.rank_value))
        classifier_scores2 = self.hyper_linear(ngrams_outputs.view(-1, self.rank_value)).view(batch_size, -1)
        classifier_scores = 0.5*classifier_scores1 + 0.5*classifier_scores2
        classifier_scores = classifier_scores.unsqueeze(1).expand(
            active_mask.size())  # shape = (batch_size, max_diff_ngram_num, max_gram_num)
        classifier_scores = classifier_scores.masked_fill(mask=active_mask, value=-float('inf'))
        total_scores, indices = torch.max(classifier_scores, dim=-1)  # shape = (batch_size * max_diff_ngram_num)
        # --------------------------------------------------------------------------------
        # Loss Compute
        if labels is not None:
            Rank_Loss_Fct = MarginRankingLoss(margin=1.0 / math.sqrt(self.rank_value), reduction='mean')
            device = torch.device("cuda", total_scores.get_device())
            flag = torch.FloatTensor([1]).to(device)
            rank_losses = []
            for i in range(batch_size):
                score = total_scores[i]
                label = labels[i]
                true_score = score[label == 1]
                neg_score = score[label == -1]
                rank_losses.append(Rank_Loss_Fct(true_score.unsqueeze(-1), neg_score.unsqueeze(0), flag))
            rank_loss = torch.mean(torch.stack(rank_losses))
            return rank_loss

        else:
            return total_scores  # shape = (batch_size * max_differ_gram_num)

    def proj(self, x, c):
        norm = torch.clamp_min(x.norm(dim=-1, keepdim=True, p=2), self.min_norm)
        maxnorm = (1 - self.eps[x.dtype]) / (c ** 0.5)
        cond = norm > maxnorm
        projected = x / norm * maxnorm
        return torch.where(cond, projected, x)

    def klein_constraint(self, x):
        last_dim_val = x.size(-1)
        norm = torch.reshape(torch.norm(x, dim=-1), [-1, 1])
        maxnorm = (1 - self.eps[x.dtype])
        cond = norm > maxnorm
        x_reshape = torch.reshape(x, [-1, last_dim_val])
        projected = x_reshape / (norm + self.min_norm) * maxnorm
        x_reshape = torch.where(cond, projected, x_reshape)
        x = torch.reshape(x_reshape, list(x.size()))
        return x

    def to_klein(self, x, c=1):
        x_2 = torch.sum(x * x, dim=-1, keepdim=True)
        x_klein = 2 * x / (1.0 + x_2)
        x_klein = self.klein_constraint(x_klein)
        return x_klein

    def klein_to_poincare(self, x, c=1):
        x_poincare = x / (1.0 + torch.sqrt(1.0 - torch.sum(x * x, dim=-1, keepdim=True)))
        x_poincare = self.proj(x_poincare, c)
        return x_poincare

    def lorentz_factors(self, x):
        x_norm = torch.norm(x, dim=-1)
        return 1.0 / (1.0 - x_norm ** 2 + self.min_norm)

    def einstein_midpoint(self, x, c=1):
        x = self.to_klein(x, c)
        x_lorentz = self.lorentz_factors(x)
        x_norm = torch.norm(x, dim=-1)
        # deal with pad value
        x_lorentz = (1.0 - torch._cast_Float(x_norm == 0.0)) * x_lorentz
        x_lorentz_sum = torch.sum(x_lorentz, dim=-1, keepdim=True)
        x_lorentz_expand = torch.unsqueeze(x_lorentz, dim=-1)
        x_midpoint = torch.sum(x_lorentz_expand * x, dim=1) / x_lorentz_sum
        x_midpoint = self.klein_constraint(x_midpoint)
        x_p = self.klein_to_poincare(x_midpoint, c)
        return x_p

class AdaptiveMixingLayer(nn.Module):

    def __init__(self, hidden_size=768):
        super().__init__()

        self.hidden_size = hidden_size
        #self.v_q = nn.Linear(hidden_size, 1, bias=False)
        #self.W_q = nn.Linear(hidden_size, hidden_size, bias=False)

        self.v_q = nn.Parameter(torch.Tensor(hidden_size, 1))
        nn.init.kaiming_normal_(self.v_q, mode='fan_in', nonlinearity='relu')

        self.W_q = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        nn.init.kaiming_normal_(self.W_q, mode='fan_in', nonlinearity='relu')


    def forward(self, layer_representations):

        atten = torch.softmax(torch.matmul(layer_representations, self.v_q).permute(1, 2, 0, 3) / math.sqrt(768.0), 2)
        atten_h = torch.matmul(layer_representations.permute(1, 2, 3, 0), atten).squeeze(-1)
        outputs = torch.matmul(atten_h, self.W_q)

        return outputs