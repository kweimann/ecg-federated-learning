import math
from typing import Union, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class BasicBlock(nn.Module):
  def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1):
    super().__init__()
    if stride > 1 or in_channels != out_channels:
      self.shortcut = conv1d(in_channels, out_channels, kernel_size=1, stride=stride)
    else:
      self.shortcut = nn.Identity()
    self.bn1 = norm1d(in_channels)
    self.relu1 = nn.ReLU()
    self.conv1 = conv1d(in_channels, out_channels, kernel_size, stride)
    self.bn2 = norm1d(out_channels)
    self.relu2 = nn.ReLU()
    self.conv2 = conv1d(out_channels, out_channels, kernel_size)

  def forward(self, x: Tensor) -> Tensor:
    shortcut = self.shortcut(x)
    x = self.bn1(x)
    x = self.relu1(x)
    x = self.conv1(x)
    x = self.bn2(x)
    x = self.relu2(x)
    x = self.conv2(x)
    out = x + shortcut
    return out


class ResNet(nn.Module):
  def __init__(self, stages: Sequence[int], num_outputs: int, *, block: nn.Module = BasicBlock,
               in_channels: int = 12, width: Union[int, Sequence[int]] = 64,
               kernel_size: Union[int, Sequence[int]] = 3, stem_kernel_size: int = 7):
    super().__init__()

    if isinstance(kernel_size, int):
      kernel_size = [kernel_size] * len(stages)
    assert len(kernel_size) == len(stages)
    if isinstance(width, int):
      width = [width * 2 ** i for i in range(len(stages))]
    assert len(width) == len(stages)

    self.block = block
    out_channels = width[0]
    self.conv1 = conv1d(in_channels, out_channels, stem_kernel_size, stride=2)
    self.max_pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
    self.stages = []
    in_channels = out_channels
    for i, num_layers in enumerate(stages):
      stride = 1 if i == 0 else 2  # first stage has a stride of 1 due to max pooling in the stem
      out_channels = width[i]
      self.stages.append(
        self._make_stage(num_layers, in_channels, out_channels, kernel_size[i], stride))
      in_channels = out_channels
    self.stages = nn.Sequential(*self.stages)
    self.bn1 = norm1d(in_channels)
    self.relu1 = nn.ReLU()
    self.global_pool = nn.AdaptiveAvgPool1d(output_size=1)
    self.fc = nn.Linear(in_channels, num_outputs)

    for module in self.modules():
      if isinstance(module, nn.Conv1d):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
          nn.init.constant_(module.bias, 0)
      elif isinstance(module, (nn.BatchNorm1d, nn.GroupNorm)):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
      elif isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=.02)
        if module.bias is not None:
          nn.init.constant_(module.bias, 0)

  def forward(self, x: Tensor) -> Tensor:
    x = self.conv1(x)
    x = self.max_pool(x)
    x = self.stages(x)
    x = self.bn1(x)
    x = self.relu1(x)
    x = self.global_pool(x).squeeze(-1)
    out = self.fc(x)
    return out

  def get_layer_info(self):
    """Returns a list of tuples that describe convolutional and pooling layers.
    Each tuple contains (kernel_size, stride, padding)."""
    layers = [
      (self.conv1.kernel_size[0], self.conv1.stride[0], self.conv1.padding[0]),
      (self.max_pool.kernel_size, self.max_pool.stride, self.max_pool.padding)
    ]
    for stage in self.stages:
      for block in stage.children():
        assert isinstance(block, BasicBlock)
        layers.append((block.conv1.kernel_size[0], block.conv1.stride[0], block.conv1.padding[0]))
        layers.append((block.conv2.kernel_size[0], block.conv2.stride[0], block.conv2.padding[0]))
    return layers

  def _make_stage(self, num_layers, in_channels, out_channels, kernel_size, stride):
    layers = [self.block(in_channels, out_channels, kernel_size, stride)]
    for i in range(num_layers - 1):
      layers.append(self.block(out_channels, out_channels, kernel_size))
    return nn.Sequential(*layers)


class AttentiveResNet(ResNet):
  def __init__(self, use_attn_pool: bool = False, sig_len: int = None, **kwargs):
    super().__init__(**kwargs)
    self.use_attn_pool = use_attn_pool
    if use_attn_pool:
      if sig_len is None:
        raise ValueError('Argument `sig_len` is required to compute sequence length in the attention module.')
      pooled_sig_len = sig_len // (2 ** (len(self.stages) + 1))
      self.attn_pool = AttentionPooling(
        d_model=self.fc.in_features,
        num_outputs=self.fc.out_features,
        max_seq_len=pooled_sig_len)
      del self.bn1
      del self.relu1
      del self.global_pool
      del self.fc

  def forward(self, x, target_mask=None):
    x = self.conv1(x)
    x = self.max_pool(x)
    x = self.stages(x)
    if self.use_attn_pool:
      logits, _ = self.attn_pool(x, target_mask)
      logits = logits.squeeze(-1)
    else:
      del target_mask
      x = self.bn1(x)
      x = self.relu1(x)
      x = self.global_pool(x).squeeze(-1)
      logits = self.fc(x)
    return logits


class AttentionPooling(nn.Module):
  def __init__(self, num_outputs, d_model: int = 512, num_heads: int = 1,
               max_seq_len: int = 512, dropout: float = 0.1, learned_pos_emb=False) -> None:
    super().__init__()
    self.d_model = d_model
    self.query = nn.Parameter(torch.zeros(num_outputs, d_model))
    if learned_pos_emb:
      self.positional = LearnedEmbedding(
        d_model=d_model, seq_len=max_seq_len)
    else:
      self.positional = ReversedPositionalEncoding(
        d_model=d_model, seq_len=max_seq_len)
    self.dropout = nn.Dropout(dropout)
    self.latents_norm = nn.LayerNorm(d_model)
    self.query_norm = nn.LayerNorm(d_model)
    self.mha = MultiHeadAttention(
      q_inputs=d_model,
      kv_inputs=d_model,
      qk_outputs=d_model,
      v_outputs=d_model,
      attn_outputs=1,
      num_heads=num_heads)

    nn.init.trunc_normal_(self.query, std=.02)
    for module in self.modules():
      if isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.weight, 1.)
        nn.init.constant_(module.bias, 0.)
      if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=.02)
        if module.bias is not None:
          nn.init.constant_(module.bias, 0.)

  def forward(self, latents: Tensor, query_mask=None):
    B, _, latent_size = latents.size()
    mask = None
    if query_mask is not None:
      latent_mask = torch.ones(B, latent_size, device=latents.device)
      mask = make_cross_attention_mask(query_mask, latent_mask)
    latents = latents.transpose(1, 2)   # swap channel axis with signal axis
    latents = latents * math.sqrt(self.d_model)
    latents = self.positional(latents)
    latents = self.dropout(latents)
    latents = self.latents_norm(latents)
    query = self.query * math.sqrt(self.d_model)
    query = self.query_norm(query)
    query = query.repeat(B, 1, 1)
    logits, attn_weights = self.mha(query, latents, mask=mask)
    return logits, attn_weights


class MultiHeadAttention(nn.Module):
  def __init__(self,
               q_inputs,
               kv_inputs,
               qk_outputs=None,
               v_outputs=None,
               attn_outputs=None,
               num_heads=8,
               dropout=0.):
    super().__init__()
    if qk_outputs is None:
      qk_outputs = q_inputs
    if v_outputs is None:
      v_outputs = qk_outputs
    if attn_outputs is None:
      attn_outputs = v_outputs
    assert qk_outputs % num_heads == 0
    assert v_outputs % num_heads == 0
    self.H = num_heads
    self.dropout = dropout
    self.D_qk = qk_outputs // num_heads
    self.D_v = v_outputs // num_heads
    self.W_q = nn.Linear(q_inputs, qk_outputs)
    self.W_k = nn.Linear(kv_inputs, qk_outputs)
    self.W_v = nn.Linear(kv_inputs, v_outputs)
    self.W_o = nn.Linear(v_outputs, attn_outputs)

  def forward(self, q, kv, *, mask=None):
    # inputs:
    #   q: Tensor [B, N_q, q_inputs]
    #   kv: Tensor [B, N_kv, kv_inputs]
    #   mask: Tensor [B, N_q, N_kv]
    # outputs:
    #   attn: Tensor [B, N_q, attn_outputs]
    B, N_q, _ = q.size()
    _, N_kv, _ = kv.size()
    q = self.W_q(q).reshape(B, N_q, self.H, self.D_qk)
    k = self.W_k(kv).reshape(B, N_kv, self.H, self.D_qk)
    v = self.W_v(kv).reshape(B, N_kv, self.H, self.D_v)
    attn, attn_weights = multi_head_attend(  # [B, N_q, v_outputs], [B, H, N_q, N_kv]
      q, k, v, mask=mask, dropout=self.dropout, training=self.training)
    attn = self.W_o(attn)  # [B, N_q, attn_outputs]
    return attn, attn_weights


class ReversedPositionalEncoding(nn.Module):
  def __init__(self, d_model: int, seq_len: int) -> None:
    super().__init__()
    encoding = positional_encoding(d_model, seq_len).flip(0).unsqueeze(0)
    self.register_buffer('encoding', encoding)

  def forward(self, x: Tensor) -> Tensor:
    N = x.size(1)
    return x + self.encoding[:, -N:]


class LearnedEmbedding(nn.Module):
  def __init__(self, d_model: int, seq_len: int) -> None:
    super().__init__()
    self.embedding = nn.Parameter(torch.empty(1, seq_len, d_model))
    nn.init.trunc_normal_(self.embedding, std=.02)

  def forward(self, x: Tensor) -> Tensor:
    N = x.size(1)
    return x + self.embedding[:, -N:]


def multi_head_attend(q, k, v, mask=None, dropout=0., training=True):
  # inputs:
  #   q: Tensor [B, N_q, H, D_qk]
  #   k: Tensor [B, N_kv, H, D_qk]
  #   v: Tensor [B, N_kv, H, D_v]
  #   mask: Tensor [B, N_q, N_kv]
  # outputs:
  #   attn: Tensor [B, N_q, H * D_v]
  #   attn_weights: Tensor [B, H, N_q, N_kv]
  B, N_q, H, D_qk = q.size()
  _, _, _, D_v = k.size()
  attn_weights = torch.einsum('bqhd,bkhd->bhqk', q, k) / math.sqrt(D_qk)
  if mask is not None:
    attn_weights = attn_weights.masked_fill(mask.unsqueeze(1) == 0, -1e9)
  attn_weights = F.softmax(attn_weights, dim=-1)
  if dropout > 0:
    attn_weights = F.dropout(attn_weights, dropout, training=training)
  attn = torch.einsum('bhqk,bkhd->bqhd', attn_weights, v).reshape(B, N_q, H * D_v)
  if mask is not None:
    wipe_attn = torch.all(mask == 0, dim=2, keepdim=True)  # [B, N_q, 1]
    attn = attn.masked_fill(wipe_attn, 0)
  return attn, attn_weights


def make_cross_attention_mask(q_mask, kv_mask):
  # inputs:
  #   q_mask: Tensor [B, N_q]
  #   kv_mask: Tensor [B, N_kv]
  # outputs:
  #   mask: Tensor [B, N_q, N_kv]
  return torch.einsum('bq,bk->bqk', q_mask, kv_mask)


def positional_encoding(d_model: int, seq_len: int) -> Tensor:
  assert d_model % 2 == 0  # for simplicity
  pos = torch.arange(0, seq_len).unsqueeze(1)
  i = torch.arange(0, d_model, 2)
  angle_rads = pos * torch.exp(i * -(math.log(10000) / d_model))
  pe = torch.zeros(seq_len, d_model)
  pe[:, 0::2] = torch.sin(angle_rads)
  pe[:, 1::2] = torch.cos(angle_rads)
  return pe


def conv1d(in_channels: int, out_channels: int, kernel_size: int, stride: int = 1) -> nn.Conv1d:
  assert kernel_size % 2 == 1  # for simplicity
  padding = (kernel_size - 1) // 2
  return nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=False)


def norm1d(in_channels: int, num_groups: int = None) -> nn.GroupNorm:
  # replace batch normalization with group normalization as per: https://arxiv.org/abs/2003.00295
  if num_groups is None:
    num_groups = in_channels // 16
  return nn.GroupNorm(num_groups, in_channels)


def get_receptive_field(layers):  # in the input layer
  """Source: https://distill.pub/2019/computing-receptive-fields/#solving-receptive-field-size"""
  r_0 = 1
  s_acc = 1
  for k_l, s_l, _ in layers:
    r_0 = r_0 + (k_l - 1) * s_acc
    s_acc = s_acc * s_l
  return r_0
