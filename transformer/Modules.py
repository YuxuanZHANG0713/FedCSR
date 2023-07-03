# Copyright (c) 2021, Soohwan Kim. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.init as init
from torch import Tensor
import torch.nn.functional as F

class ResidualConnectionModule(nn.Module):
    """
    Residual Connection Module.
    outputs = (module(inputs) x module_factor + inputs x input_factor)
    """
    def __init__(self, module: nn.Module, module_factor: float = 1.0, input_factor: float = 1.0):
        super(ResidualConnectionModule, self).__init__()
        self.module = module
        self.module_factor = module_factor
        self.input_factor = input_factor

    def forward(self, inputs: Tensor) -> Tensor:
        return (self.module(inputs) * self.module_factor) + (inputs * self.input_factor)


class Linear(nn.Module):
    """
    Wrapper class of torch.nn.Linear
    Weight initialize by xavier initialization and bias initialize to zeros.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        init.xavier_uniform_(self.linear.weight)
        if bias:
            init.zeros_(self.linear.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)


class View(nn.Module):
    """ Wrapper class of torch.view() for Sequential module. """
    def __init__(self, shape: tuple, contiguous: bool = False):
        super(View, self).__init__()
        self.shape = shape
        self.contiguous = contiguous

    def forward(self, x: Tensor) -> Tensor:
        if self.contiguous:
            x = x.contiguous()

        return x.view(*self.shape)


class Transpose(nn.Module):
    """ Wrapper class of torch.transpose() for Sequential module. """
    def __init__(self, shape: tuple):
        super(Transpose, self).__init__()
        self.shape = shape

    def forward(self, x: Tensor) -> Tensor:
        return x.transpose(*self.shape)


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn


class Swish(nn.Module):
    r"""
    In the ContextNet paper, the swish activation function works consistently better than ReLU.
    """
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs * inputs.sigmoid()



class SELayer(nn.Module):
    r"""
    Squeeze-and-excitation module.
    Args:
        dim (int): Dimension to be used for two fully connected (FC) layers
    Inputs: inputs, input_lengths
        - **inputs**: The output of the last convolution layer. `FloatTensor` of size
            ``(batch, dimension, seq_length)``
        - **input_lengths**: The length of input tensor. ``(batch)``
    Returns: output
        - **output**: Output of SELayer `FloatTensor` of size
            ``(batch, dimension, seq_length)``
    """
    def __init__(self, d_model: int) -> None:
        super(SELayer, self).__init__()
        assert d_model % 8 == 0, 'Dimension should be divisible by 8.'

        self.dim = d_model
        self.sequential = nn.Sequential(
            nn.Linear(self.dim, self.dim // 8),
            Swish(),
            nn.Linear(self.dim // 8, self.dim),
        )

        
    def forward(
            self,
            inputs: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        r"""
        Forward propagate a `inputs` for SE Layer.
        Args:
            **inputs** (torch.FloatTensor): The output of the last convolution layer. `FloatTensor` of size
                ``(batch, dimension, seq_length)``
            **input_lengths** (torch.LongTensor): The length of input tensor. ``(batch)``
        Returns:
            **output** (torch.FloatTensor): Output of SELayer `FloatTensor` of size
                ``(batch, dimension, seq_length)``
        """
        # inputs = F.avg_pool2d(inputs, kernel_size=(4,1), padding=0)
        inputs = inputs.transpose(1,2)
        input_length = inputs.shape[2]

        residual = inputs
        inputs = inputs.sum(dim=2) / input_length
        output = self.sequential(inputs)

        output = output.sigmoid().unsqueeze(2)
        output = output.repeat(1, 1, input_length)
        output = output * residual + residual
        output = output.transpose(1,2) # (batch, frame, dim)
        return output

class SubLayer(nn.Module):
    r"""
    Squeeze-and-excitation module.
    Args:
        dim (int): Dimension to be used for two fully connected (FC) layers
    Inputs: inputs, input_lengths
        - **inputs**: The output of the last convolution layer. `FloatTensor` of size
            ``(batch, dimension, seq_length)``
        - **input_lengths**: The length of input tensor. ``(batch)``
    Returns: output
        - **output**: Output of SELayer `FloatTensor` of size
            ``(batch, dimension, seq_length)``
    """
    def __init__(self, d_model: int) -> None:
        super(SubLayer, self).__init__()
        assert d_model % 8 == 0, 'Dimension should be divisible by 8.'

        self.dim = d_model
        self.sequential = nn.Sequential(
            nn.Linear(self.dim, self.dim // 8),
            Swish(),
            nn.Linear(self.dim // 8, self.dim),
        )

        self.project = nn.Linear(self.dim*2, self.dim)
        

    def forward(
            self,
            inputs: Tensor,
            input_lengths: Tensor
    ) -> Tuple[Tensor, Tensor]:
        r"""
        Forward propagate a `inputs` for SE Layer.
        Args:
            **inputs** (torch.FloatTensor): The output of the last convolution layer. `FloatTensor` of size
                ``(batch, dimension, seq_length)``
            **input_lengths** (torch.LongTensor): The length of input tensor. ``(batch)``
        Returns:
            **output** (torch.FloatTensor): Output of SELayer `FloatTensor` of size
                ``(batch, dimension, seq_length)``
        """
        # inputs = F.avg_pool2d(inputs, kernel_size=(4,1), padding=0)
        inputs = inputs.transpose(1,2)
        input_length = inputs.shape[2]

        residual = inputs
        inputs = inputs.sum(dim=2) / input_length
        output = self.sequential(inputs)

        output = output.sigmoid().unsqueeze(2)
        output = output.repeat(1, 1, input_length)
        output = output * residual + residual
        output = output.transpose(1,2) # (batch, frame, dim)

        # print(output.shape)
        if output.shape[1]%2 != 0:
            shape = output.shape
            # shape[1]=output.shape[1]%4
            output= torch.cat((output, torch.zeros((shape[0],2-output.shape[1]%2, shape[2])).to(output)), dim=1)

        # print(output.shape)
        output = output.reshape((output.shape[0], int(output.shape[1]/2), output.shape[2]*2))

        output = self.project(output)

        # output = F.avg_pool2d(output, kernel_size=(4,1), padding=0)

        output_lengths = input_lengths >> 1

        return output, output_lengths




class GateLayer(nn.Module):
    def __init__(self, channel, reduction=1):
        super(GateLayer, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        batch_size, seq_len, expert_num, dimension = x.size() # (batch, seq_length, 3, dimension)

        residual = x
        y = x.sum(dim=-1) / dimension
        y = self.fc(y).view(batch_size, seq_len, expert_num, 1)
        # print(y.size())
        x = y.expand_as(x) * residual + residual
        x = x.transpose(2, 3).sum(-1)
        # x = self.avg_pool(x).view(batch_size, seq_len, -1)
        return x