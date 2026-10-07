import torch

from einops import rearrange
from torch import nn


class CausalSelfAttention(nn.Module):
  def __init__(self, config):
    super().__init__()

    self.num_attention_heads = config.num_attention_heads
    self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
    self.all_head_size = self.num_attention_heads * self.attention_head_size

    # Initialize the linear transformation layers for key, value, query.
    self.query = nn.Linear(config.hidden_size, self.all_head_size)
    self.key = nn.Linear(config.hidden_size, self.all_head_size)
    self.value = nn.Linear(config.hidden_size, self.all_head_size)
    # This dropout is applied to normalized attention scores following the original
    # implementation of transformer. Although it is a bit unusual, we empirically
    # observe that it yields better performance.
    self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

  def transform(self, x, linear_layer):
    # The corresponding linear_layer of k, v, q are used to project the hidden_state (x).
    proj = linear_layer(x)
    # Next, we need to produce multiple heads for the proj. This is done by spliting the
    # hidden state to self.num_attention_heads, each of size self.attention_head_size.
    proj = rearrange(proj, 'b t (h d) -> b t h d', h=self.num_attention_heads)
    # By proper transpose, we have proj of size [bs, num_attention_heads, seq_len, attention_head_size].
    proj = rearrange(proj, 'b t h d -> b h t d')
    return proj

  def attention(self, key, query, value, attention_mask):

    # key/query/value:
    # [bs, num_heads, seq_len, head_dim]

    # 1. 计算 QK^T
    attention_scores = torch.matmul(
        query,
        key.transpose(-1, -2)
    )

    # 2. 缩放，防止数值过大
    attention_scores = attention_scores / (self.attention_head_size ** 0.5)

    # 3. 构造 causal mask，防止看到未来 token
    seq_len = query.size(-2)

    #生成一个下三角矩阵 下三角是全1的 
    causal_mask = torch.tril(
        torch.ones(
            seq_len,
            seq_len,
            device=query.device,
            dtype=torch.bool
        )
    )

    # [seq_len, seq_len]
    # -> [1, 1, seq_len, seq_len]
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

    attention_scores = attention_scores.masked_fill(
        ~causal_mask,
        float('-inf')
    )
    
    # 4. 应用 padding attention mask
    if attention_mask is not None:
        attention_scores = attention_scores + attention_mask
        
    # 5. 转成 attention weights
    attention_probs = torch.softmax(
        attention_scores,
        dim=-1
    )

    # 6. dropout
    attention_probs = self.dropout(attention_probs)

    # 7. attention weights × V
    context = torch.matmul(
        attention_probs,
        value
    )

    # context:
    # [bs, num_heads, seq_len, head_dim]
    # -> [bs, seq_len, num_heads, head_dim]
    context = rearrange(
        context,
        'b h t d -> b t h d'
    )

    # 合并多个 head
    # [bs, seq_len, num_heads, head_dim]
    # -> [bs, seq_len, hidden_size]
    context = rearrange(
        context,
        'b t h d -> b t (h d)'
    )

    return context


  def forward(self, hidden_states, attention_mask):
    """
    hidden_states: [bs, seq_len, hidden_state]
    attention_mask: [bs, 1, 1, seq_len]
    output: [bs, seq_len, hidden_state]
    """
    # First, we have to generate the key, value, query for each token for multi-head attention
    # using self.transform (more details inside the function).
    # Size of *_layer is [bs, num_attention_heads, seq_len, attention_head_size].
    key_layer = self.transform(hidden_states, self.key)
    value_layer = self.transform(hidden_states, self.value)
    query_layer = self.transform(hidden_states, self.query)
    
    # Calculate the multi-head attention.
    attn_value = self.attention(key_layer, query_layer, value_layer, attention_mask)
    return attn_value
