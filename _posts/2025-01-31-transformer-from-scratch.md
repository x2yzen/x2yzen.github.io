---
title: Transformer from Scratch
date: 2025-01-31 08:00:00 +0800
categories: [language_model]
tags: [llm] 
pin: true
math: true
---

尽量少依赖地构建 transformer 原型

## Original design

![](/assets/images/2025-01-31-transformer-from-scratch/SWXWbBautoe3QHx7GTijr8bTpxi.png)
![](/assets/images/2025-01-31-transformer-from-scratch/OITebUeGaoNz7axNMTFju39lpAb.png)

### Demo implementation

```python
import torch
import torch.nn as nn
import torch.optim as optim
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output

    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(start=0, end=max_seq_length, step=1, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(start=0, end=d_model, step=2).float() * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x


class DecoderOnlyLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderOnlyLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, tgt_mask):
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        super(Transformer, self).__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        seq_length = tgt.size(1)
        no_peek_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
        tgt_mask = tgt_mask & no_peek_mask
        return src_mask, tgt_mask

    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        output = self.fc(dec_output)
        return output


class DecoderOnlyTransformer(nn.Module):
    def __init__(self, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        super(DecoderOnlyTransformer, self).__init__()
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        self.decoder_layers = nn.ModuleList(
            [DecoderOnlyLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, tgt):
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        seq_length = tgt.size(1)
        no_peek_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
        tgt_mask = tgt_mask & no_peek_mask
        return tgt_mask

    def forward(self, tgt):
        tgt_mask = self.generate_mask(tgt)
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, tgt_mask)

        output = self.fc(dec_output)
        return output


if __name__ == "__main__":
    tgt_vocab_size = 5000
    d_model = 512
    num_heads = 2
    num_layers = 8
    d_ff = 2048
    max_seq_length = 100
    dropout = 0.1

    transformer = DecoderOnlyTransformer(tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)

    # Generate random sample data
    tgt_data = torch.randint(1, tgt_vocab_size, (64, max_seq_length))  # (batch_size, seq_length)

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

    transformer.train()

    for epoch in range(10):
        optimizer.zero_grad()
        output = transformer(tgt_data[:, :-1])
        loss = criterion(output.contiguous().view(-1, tgt_vocab_size), tgt_data[:, 1:].contiguous().view(-1))
        loss.backward()
        optimizer.step()
        print(f"Epoch: {epoch + 1}, Loss: {loss.item()}")
```

*或者参考这个项目：[https://github.com/jingyaogong/minimind](https://github.com/jingyaogong/minimind)

### Explanation step by step

对照以下笔记和代码运行的 debug 断点，理解每一个代码模块的作用

![](/assets/images/2025-01-31-transformer-from-scratch/E0RWbBIeAoWKtmx7ydwjlocmpCg.jpg)

## Positional embedding

Ref: Sinusoidal and RoPE positional embedding ([youtube](https://www.youtube.com/watch?v=GQPOtyITy54))

### Absolute Positional Embedding

The position of each token is explicitly represented by a unique embedding vector. This vector is either added to or concatenated with the word embeddings before being input into the transformer layers.Sinusoidal encoding is a common implementation

$$
PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d_{model}})\newline
PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d_{model}})\newline
a_{m,n} = q_m^T k_n = [W_q(x_m + PE(m))]^T W_k(x_n + PE(n))
$$

![](/assets/images/2025-01-31-transformer-from-scratch/T4Q8bdxjxoOl2ExCGTWjMQHtpSg.png)

As could be seen in the picture, the change for each position is a bit erratic and difficult to find patterns

### Relative embedding

#### T5 relative bias

add pair-wise distance embedding to the attn matrix, the addition is the same as long as the relative position between the token pair keeps the same (e.g. appending prefix or suffix does not effect the embeding)

![](/assets/images/2025-01-31-transformer-from-scratch/QID5bZloSoCo2IxiufAjeNW9pVf.png)

#### RoPE

- split q and k vectors, 2 dimensions each (tot. d_model/2 slices)
- assign each slice a unique \theta value

  $$
  heta_{i} = b^{-2i/D}
  $$
- apply 2-D rotation on each slice based on the position index *实际往往是通过复数乘法实现的

  $$
  {\Theta, m}^{d} = \begin{pmatrix}
  $$

\cos(m\theta_i) & -\sin(m\theta_i) \\
\sin(m\theta_i) & \cos(m\theta_i)
\end{pmatrix}$$

![](/assets/images/2025-01-31-transformer-from-scratch/JwomblCvcoM3Q8xsdRmjGVlhpef.png)

Wrap up:

$$
a_{m,n} = q_m^T k_n = [R_{\Theta,d}^m (W_q x_m)]^T R_{\Theta,d}^n W_k x_n = (W_q x_m)^T {R_{\Theta,d}^m}^T R_{\Theta,d}^n W_k x_n
$$

The relative (angular) position between tokens is preserved and is more predictable compared to sinusoidal encoding, yields better perplexity beyond training length.

![](/assets/images/2025-01-31-transformer-from-scratch/VLK9bcvTqo8hbVxsnNwjgFmepOh.png)
![](/assets/images/2025-01-31-transformer-from-scratch/LYD9bKopwo4KmwxnGGujJ4P0p2c.png)

### Position Interpolation

讨论基于 RoPE 进行 context window extension 的方法。总的来说各类方法都可以概述为：

$$
f'_{\mathbf{W}}(\mathbf{x}_m, m, \theta_d) = f_{\mathbf{W}}(\mathbf{x}_m, g(m), h(\theta_d))
$$

定义：

$$
s= \frac{L'}{L} > 1
$$

由于 d 维度向量中每个位置实际旋转的角度

$$
m\theta_i = mb^{-2d/D}
$$

为了使得 m 能够取到比 training 时更大的值，同时乘积尽量不超限，最简单的方式是令

$$
g(m) = \frac{m}{s}
$$

从 embedding vector 的角度看，这个方法均匀地“拉伸”所有维度；或者从 positional encoding 的角度看，均匀地降低了所有采样频率（所有给定距离的 token pair，相对旋转角度都缩小了 s 倍），类似的方法还有令

$$
h(\theta_d) = (b*s)^{-2d/{D}}
$$

这种覆盖全采样频率的方法还有优化的空间：引入波长的概念，表征对于 d 维度的 embedding 向量，相距多少个 token 才能够使 RoPE encoding 旋转一整周

$$
\lambda_{d} = \frac{2\pi}{\theta_{d}} = 2\pi b^{\frac{2d}{|D|}}
$$

- 在短波长的维度（比如不到 L/32），positional encoding 负责捕获 token 的局部位置关系，由于 RoPE 具备周期性的，在进行 context windown extension 时，并不需要调整（e.g. embedding 无非就是从 32 整周变成了 64 整周，整体 embedding 数值 scale 没有膨胀，分布仍然是均匀的
- 长波长的维度（比如比 L 更长）则捕获全局的 token 位置关系，在 context windown extension 时会面临更大的问题（e.g. embedding 数值 scale 可能会超过 training 时见过的上限，并显著改变在角位置上的分布），因此需要 interpolation

这就是 [YaRN](https://arxiv.org/abs/2309.00071) 的基本思想，还有少数 trick，比如 dynamic scaling（可能会破坏 kv cache）和 temperature factor，比较细节，需要时查阅原文即可。

![](/assets/images/2025-01-31-transformer-from-scratch/S4RybsiuloIV3exV91CjnSgRptw.png)

## Mixture-of-Experts (MoE)

([ref](https://huggingface.co/blog/moe#what-is-a-mixture-of-experts-moe))

MoEs:

- Are **pretrained much faster** vs. dense models
- Have **faster inference** compared to a model with the same number of parameters
- Require **high VRAM** as all experts are loaded in memory
- Face many **challenges in fine-tuning**, but <u>recent work</u> with MoE instruction-tuning is promising

![](/assets/images/2025-01-31-transformer-from-scratch/R0sSb2RsCo8zgVxgnYMj5UrCpfg.png)

### Motivation

最主要的目的是想扩大模型容量（以参数量计），但在计算量上妥协。i.e. scaling model capacity without proportional compute cost

- Dense Models: Scaling up dense models (e.g., Transformers) requires increasing both model size and compute proportionally.
- Sparse MoE Models: Only a subset of experts is activated per input, allowing you to increase the number of experts (model capacity) without increasing the computation cost significantly.

  - For example, in a model with 100 experts where only 2 are active per input, the computational cost is equivalent to a much smaller dense model.
  - This enables training and inference on much larger models than would be feasible with dense architectures.

### 主要结构

- Sparse MoE layers are used instead of dense feed-forward network (FFN) layers.
- A gate network or router, that determines which tokens are sent to which expert

### 实现细节

- 怎么处理 top-k masking 带来的 indifferentiable：When performing backpropagation in this setup, the top-k hard masking step is effectively ignored during the gradient computation. Instead, the loss propagates gradients back through the original scores (before the masking)
- Load balancing loss：为了避免 expert 之间分到的 token sample 不均衡，会额外加上一个 load balance loss，经常被定义为每个 batch 中每个 expert 被选到的频率与均匀分布的 kl-div，在最后一层累加在一块，再和 target loss 加和反向传播
  ![](/assets/images/2025-01-31-transformer-from-scratch/Pr5ybDr8ooKyAHxRnmkjR0tEpZe.png)

```

# Compute the main task loss

main_loss = nn.MSELoss()(outputs, target)

# Sum all balance losses

total_balance_loss = sum(all_balance_losses)

# Total loss (main task + all balance losses)

total_loss = main_loss + 0.1 * total_balance_loss  # Weight balance loss as needed

```

### fine-tuning MoEs

- more **prone to overfitting***, so we can explore** higher regularization (e.g. dropout)** within the experts themselves (e.g. we can have one dropout rate for the dense layers and another, higher, dropout for the sparse layers). ***reason**: each expert sees a smaller fraction of the overall dataset during training and has less diverse data to learn from. It may overfit to the specific patterns in its assigned subset, leading to poorer generalization; the  sparse model typically has high model capacity per input

- Whether to use **auxiliary loss** (the load balancing loss): some experiments show turning off auxiliary loss and just drop overload tokens does not significantly impact performance, as **token dropping** might be a form of regularization that helps prevent overfitting, while there are other experiments that show the opposite

- Which parameters to **freeze**: while the majority number of parameters lies in MoE layers, freeze them and only tune others reaches similar performance as tuning all the parameters.

- different fine-tuning **hyperparameter setups** - e.g., sparse models tend to benefit more from smaller batch sizes and higher learning rates.

## Useful Resources

- The paper "attention is all you need" ([arxiv](https://arxiv.org/abs/1706.03762))

- Self-attention explained by excel sheets ([link](https://docs.google.com/spreadsheets/d/1QFeC5vASezY-JQKdBfrn5vDnrTQgpehUMHqFuL-sqHI/edit?usp=sharing), [github](https://github.com/ImagineAILab/ai-by-hand-excel/?tab=readme-ov-file))

- The original demo implementation ([link](https://www.datacamp.com/tutorial/building-a-transformer-with-py-torch))

