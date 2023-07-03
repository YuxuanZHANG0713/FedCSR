import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional
import torch.nn.init as init
from transformer.Modules import Linear





class PositionalEncoding(nn.Module):
    """
    Positional Encoding proposed in "Attention Is All You Need".
    Since transformer contains no recurrence and no convolution, in order for the model to make
    use of the order of the sequence, we must add some positional information.

    "Attention Is All You Need" use sine and cosine functions of different frequencies:
        PE_(pos, 2i)    =  sin(pos / power(10000, 2i / d_model))
        PE_(pos, 2i+1)  =  cos(pos / power(10000, 2i / d_model))
    """
    def __init__(self, d_model: int = 512, max_len: int = 10000) -> None:
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model, requires_grad=False)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, length: int) -> Tensor:
        return self.pe[:, :length]



class RelativeMultiHeadAttentionModule(nn.Module):
    """
    Multi-head attention with relative positional encoding.
    This concept was proposed in the "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context"

    Args:
        d_model (int): The dimension of model
        num_heads (int): The number of attention heads.
        dropout_p (float): probability of dropout

    Inputs: query, key, value, pos_embedding, mask
        - **query** (batch, time, dim): Tensor containing query vector
        - **key** (batch, time, dim): Tensor containing key vector
        - **value** (batch, time, dim): Tensor containing value vector
        - **pos_embedding** (batch, time, dim): Positional embedding tensor
        - **mask** (batch, 1, time2) or (batch, time1, time2): Tensor containing indices to be masked

    Returns:
        - **outputs**: Tensor produces by relative multi head attention module.
    """
    def __init__(self, n_head: int, d_model: int, d_k: int, d_v: int, dropout: float = 0.1):
        super(RelativeMultiHeadAttentionModule, self).__init__()
        assert d_model % n_head == 0, "d_model % num_heads should be zero."

        
        self.d_model = d_model
        self.d_head = int(d_model / n_head)
        self.num_heads = n_head
        self.sqrt_dim = math.sqrt(d_model)

        self.query_proj = Linear(d_model, n_head * d_k)
        self.key_proj = Linear(d_model, n_head * d_k)
        self.value_proj = Linear(d_model, n_head * d_v)
        self.pos_proj = Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(p=dropout)
        self.u_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
        self.v_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
        torch.nn.init.xavier_uniform_(self.u_bias)
        torch.nn.init.xavier_uniform_(self.v_bias)

        self.out_proj = Linear(d_model, d_model)

    def forward(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            pos_embedding: Tensor,
            mask: Optional[Tensor] = None,
    ) -> Tensor:
        batch_size = value.size(0)


        query = self.query_proj(query).view(batch_size, -1, self.num_heads, self.d_head)
        key = self.key_proj(key).view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        value = self.value_proj(value).view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        pos_embedding = self.pos_proj(pos_embedding).view(batch_size, -1, self.num_heads, self.d_head)

        content_score = torch.matmul((query + self.u_bias).transpose(1, 2), key.transpose(2, 3))
        pos_score = torch.matmul((query + self.v_bias).transpose(1, 2), pos_embedding.permute(0, 2, 3, 1))
        pos_score = self._relative_shift(pos_score)

        score = (content_score + pos_score) / self.sqrt_dim

        if mask is not None:
            mask = mask.unsqueeze(1)
            score.masked_fill_(mask, -1e9)

        attn = F.softmax(score, -1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, value).transpose(1, 2)
        # print("context", context.shape)

        context = context.contiguous().view(batch_size, -1, self.d_model)

        return self.out_proj(context), attn

    def _relative_shift(self, pos_score: Tensor) -> Tensor:
        batch_size, num_heads, seq_length1, seq_length2 = pos_score.size()
        zeros = pos_score.new_zeros(batch_size, num_heads, seq_length1, 1)
        padded_pos_score = torch.cat([zeros, pos_score], dim=-1)

        padded_pos_score = padded_pos_score.view(batch_size, num_heads, seq_length2 + 1, seq_length1)
        pos_score = padded_pos_score[:, :, 1:].view_as(pos_score)

        return pos_score


class RelativeMultiHeadedSelfAttention(nn.Module):
    """
    Conformer employ multi-headed self-attention (MHSA) while integrating an important technique from Transformer-XL,
    the relative sinusoidal positional encoding scheme. The relative positional encoding allows the self-attention
    module to generalize better on different input length and the resulting encoder is more robust to the variance of
    the utterance length. Conformer use prenorm residual units with dropout which helps training
    and regularizing deeper models.

    Args:
        d_model (int): The dimension of model
        num_heads (int): The number of attention heads.
        dropout_p (float): probability of dropout

    Inputs: inputs, mask
        - **inputs** (batch, time, dim): Tensor containing input vector
        - **mask** (batch, 1, time2) or (batch, time1, time2): Tensor containing indices to be masked

    Returns:
        - **outputs** (batch, time, dim): Tensor produces by relative multi headed self attention module.
    """
   
    def __init__(self, n_head: int, d_model: int, d_k: int, d_v: int, dropout: float = 0.1):
        super(RelativeMultiHeadedSelfAttention, self).__init__()

        self.positional_encoding = PositionalEncoding(d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.attention = RelativeMultiHeadAttentionModule(n_head, d_model, d_k, d_v, dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, inputs: Tensor, mask: Optional[Tensor] = None):
        
        residual = inputs

        batch_size, seq_length, _ = inputs.size()
        pos_embedding = self.positional_encoding(seq_length)
        pos_embedding = pos_embedding.repeat(batch_size, 1, 1)
        inputs = self.layer_norm(inputs)

        
        outputs, attn = self.attention(inputs, inputs, inputs, pos_embedding=pos_embedding, mask=mask)
        outputs = self.dropout(outputs)
        outputs += residual
        
        return outputs, attn


class MMRelativeMultiHeadAttentionModule(nn.Module):
    """
    Multi-head attention with relative positional encoding.
    This concept was proposed in the "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context"

    Args:
        d_model (int): The dimension of model
        num_heads (int): The number of attention heads.
        dropout_p (float): probability of dropout

    Inputs: query, key, value, pos_embedding, mask
        - **query** (batch, time, dim): Tensor containing query vector
        - **key** (batch, time, dim): Tensor containing key vector
        - **value** (batch, time, dim): Tensor containing value vector
        - **pos_embedding** (batch, time, dim): Positional embedding tensor
        - **mask** (batch, 1, time2) or (batch, time1, time2): Tensor containing indices to be masked

    Returns:
        - **outputs**: Tensor produces by relative multi head attention module.
    """
    def __init__(self, n_head: int, d_model: int, d_k: int, d_v: int, dropout: float = 0.1):
        super(MMRelativeMultiHeadAttentionModule, self).__init__()
        assert d_model % n_head == 0, "d_model % num_heads should be zero."

        
        self.d_model = d_model
        self.d_head = int(d_model / n_head)
        self.num_heads = n_head
        self.sqrt_dim = math.sqrt(d_model)

        self.query_proj = Linear(d_model, n_head * d_k)
        self.key_proj = Linear(d_model, n_head * d_k)
        self.value_proj_1 = Linear(d_model, n_head * d_v)
        self.value_proj_2 = Linear(d_model, n_head * d_v)

        self.pos_proj = Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(p=dropout)
        self.u_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
        self.v_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))


        torch.nn.init.xavier_uniform_(self.u_bias)
        torch.nn.init.xavier_uniform_(self.v_bias)

        self.out_proj_1 = Linear(d_model, d_model)
        self.out_proj_2 = Linear(d_model, d_model)

    def forward(
            self,
            inputs_1: Tensor,
            inputs_2: Tensor,
            pos_embedding: Tensor,
            mask: Optional[Tensor] = None,
    ) -> Tensor:
        batch_size = inputs_1.size(0)


        query_1 = self.query_proj(inputs_1).view(batch_size, -1, self.num_heads, self.d_head)
        key_1 = self.key_proj(inputs_1).view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        value_1 = self.value_proj_1(inputs_1).view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)

        query_2 = self.query_proj(inputs_2).view(batch_size, -1, self.num_heads, self.d_head)
        key_2 = self.key_proj(inputs_2).view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        value_2 = self.value_proj_1(inputs_2).view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)

        pos_embedding = self.pos_proj(pos_embedding).view(batch_size, -1, self.num_heads, self.d_head)

        content_score_1 = torch.matmul((query_1 + self.u_bias).transpose(1, 2), key_1.transpose(2, 3))
        pos_score_1 = torch.matmul((query_1 + self.v_bias).transpose(1, 2), pos_embedding.permute(0, 2, 3, 1))
        pos_score_1 = self._relative_shift(pos_score_1)
        score_1 = (content_score_1 + pos_score_1) / self.sqrt_dim

        content_score_2 = torch.matmul((query_2 + self.u_bias).transpose(1, 2), key_2.transpose(2, 3))
        pos_score_2 = torch.matmul((query_2 + self.v_bias).transpose(1, 2), pos_embedding.permute(0, 2, 3, 1))
        pos_score_2 = self._relative_shift(pos_score_2)
        score_2 = (content_score_2 + pos_score_2) / self.sqrt_dim

        if mask is not None:
            mask = mask.unsqueeze(1)
            score_1.masked_fill_(mask, -1e9)
            score_2.masked_fill_(mask, -1e9)

        attn_1 = F.softmax(score_1, -1)
        attn_1 = self.dropout(attn_1)
        context_1 = torch.matmul(attn_1, value_1).transpose(1, 2)
        context_1 = context_1.contiguous().view(batch_size, -1, self.d_model)
        context_1 = self.out_proj_1(context_1)

        attn_2 = F.softmax(score_2, -1)
        attn_2 = self.dropout(attn_2)
        context_2 = torch.matmul(attn_2, value_2).transpose(1, 2)
        context_2 = context_2.contiguous().view(batch_size, -1, self.d_model)
        context_2 = self.out_proj_2(context_2)

        return context_1, attn_1, context_2, attn_2


    def _relative_shift(self, pos_score: Tensor) -> Tensor:
        batch_size, num_heads, seq_length1, seq_length2 = pos_score.size()
        zeros = pos_score.new_zeros(batch_size, num_heads, seq_length1, 1)
        padded_pos_score = torch.cat([zeros, pos_score], dim=-1)

        padded_pos_score = padded_pos_score.view(batch_size, num_heads, seq_length2 + 1, seq_length1)
        pos_score = padded_pos_score[:, :, 1:].view_as(pos_score)

        return pos_score


class MMRelativeMultiHeadedSelfAttention(nn.Module):
    """
    Conformer employ multi-headed self-attention (MHSA) while integrating an important technique from Transformer-XL,
    the relative sinusoidal positional encoding scheme. The relative positional encoding allows the self-attention
    module to generalize better on different input length and the resulting encoder is more robust to the variance of
    the utterance length. Conformer use prenorm residual units with dropout which helps training
    and regularizing deeper models.

    Args:
        d_model (int): The dimension of model
        num_heads (int): The number of attention heads.
        dropout_p (float): probability of dropout

    Inputs: inputs, mask
        - **inputs** (batch, time, dim): Tensor containing input vector
        - **mask** (batch, 1, time2) or (batch, time1, time2): Tensor containing indices to be masked

    Returns:
        - **outputs** (batch, time, dim): Tensor produces by relative multi headed self attention module.
    """
   
    def __init__(self, n_head: int, d_model: int, d_k: int, d_v: int, dropout: float = 0.1):
        super(MMRelativeMultiHeadedSelfAttention, self).__init__()

        self.positional_encoding = PositionalEncoding(d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.attention = MMRelativeMultiHeadAttentionModule(n_head, d_model, d_k, d_v, dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, inputs_1: Tensor, inputs_2: Tensor, mask: Optional[Tensor] = None):
        
        residual_1 = inputs_1
        residual_2 = inputs_2

        batch_size, seq_length, _ = inputs_1.size()
        pos_embedding = self.positional_encoding(seq_length)

        pos_embedding = pos_embedding.repeat(batch_size, 1, 1)

        inputs_1 = self.layer_norm(inputs_1)
        inputs_2 = self.layer_norm(inputs_2)

        outputs_1, attn_1, outputs_2, attn_2 = self.attention(inputs_1, inputs_2, pos_embedding=pos_embedding, mask=mask)
        outputs_1 = self.dropout(outputs_1)
        outputs_1 += residual_1
        outputs_2 = self.dropout(outputs_2)
        outputs_2 += residual_2
        
        return outputs_1, attn_1, outputs_2, attn_2

class VanillaMultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.temperature=d_k ** 0.5

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

       
        residual = q

        # print("q", q.shape)

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.


        # q, attn = self.attention(q, k, v, mask=mask)

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            # print("mask", mask.shape)
            # print("attn", attn.shape)
            attn = attn.masked_fill(mask == 0, -1e9)

        # print("attn", attn.shape)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        output = output.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        output = self.dropout(self.fc(output))
        output += residual

        output = self.layer_norm(output)

        return output, attn



class MultiHeadAttentionFusion(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.temperature=d_k ** 0.5

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v_q, v_k, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v_q.size(1)

       
        residual = 0.5*(v_q + v_k)

        # print("q", q.shape)

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)

        v_q = self.w_vs(v_q).view(sz_b, len_v, n_head, d_v)
        v_k = self.w_vs(v_k).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v_q, v_k = q.transpose(1, 2), k.transpose(1, 2), v_q.transpose(1, 2), v_k.transpose(1, 2)

        
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            mask = mask.unsqueeze(1)
            attn = attn.masked_fill(mask == 0, -1e9)
            mask = mask.transpose(2 ,3)
            attn = attn.masked_fill(mask == 0, -1e9)

        # print("attn", attn.shape)

        qk_attn = self.dropout(F.softmax(attn, dim=-1))
        kq_attn = self.dropout(F.softmax(attn.transpose(2, 3), dim=-1))

        confidence = qk_attn**2 + kq_attn**2 - 2*qk_attn*kq_attn
        confidence = torch.sum(confidence, dim=-1, keepdim=True)
        confidence = torch.sqrt(confidence+1e-8)
        confidence = 1 - torch.sigmoid(confidence)
        output = torch.matmul(attn, v_q + v_k)

        output = output.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        output = self.dropout(self.fc(output))
        output += residual
        output = self.layer_norm(output)

        return output, attn