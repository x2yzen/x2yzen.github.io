---
title: Whitebox Transformer From Scratch
date: 2024-11-25 19:26:00 +0800
categories: [language_model]
tags: [transformer, pytorch]
math: false
---

A tutorial offering an intuitive understanding of the decoder-only Transformer.  
The whole pipeline is sketched, then implemented from scratch in PyTorch.

## The Hand-Drawn Schematic

![Hand-drawn whitebox decoder-only transformer schematic](/assets/images/2024-11-25-whitebox-transformer/whitebox-transformer.png)
_One page covering the decoder-only path: tokenization → embedding → Q/K/V projection → multi-head split → RoPE → scaled dot-product attention with causal mask → residual + LayerNorm → feed-forward (or MoE) → residual + LayerNorm → LM head → cross-entropy. Click to zoom._

What follows is the notebook in its original cell order. The training cell finishes with three epochs of generations starting from "Once upon a time"—a fun reminder of how quickly even a tiny transformer picks up the rhythm of a corpus.

## The MVP Implementation

A Transformer is just a bunch of tensor operations, so we can build one **with and only with** PyTorch.  
To help build intuition, tensor shapes are noted in comments at every step.


```python
import torch
import torch.nn as nn
import torch.optim as optim
import math
```


```python
class Attention(nn.Module):
  def __init__(self,d_model,n_heads):
    super().__init__()

    assert d_model%n_heads == 0, "d_model must be divisible by n_heads"
    self.d_model = d_model
    self.n_heads = n_heads
    self.d_head = d_model//n_heads

    self.W_q = nn.Linear(in_features=d_model,out_features=d_model)
    self.W_k = nn.Linear(in_features=d_model,out_features=d_model)
    self.W_v = nn.Linear(in_features=d_model,out_features=d_model)
    self.W_o = nn.Linear(in_features=d_model,out_features=d_model)

  def apply_rope(self,input):
    """
    Args:
      input: [batch_size, n_heads, seq_len, d_head]
    Returns:
      rotated_input: [batch_size, n_heads, seq_len, d_head]
    """
    device = input.device

    # Compute frequencies for each dimension pair
    i = torch.arange(start=0, end=self.d_head, step=2, dtype=torch.float32, device=device)
    freqs = 1.0 / (10000 ** (i / self.d_head))

    # Compute rotation angles
    positions = torch.arange(start=0, end=input.size(2), step=1, device=input.device)
    angles = torch.outer(positions,freqs) # [seq_len, d_head/2]
    cos = torch.cos(angles).unsqueeze(0).unsqueeze(0) # [1,1,seq_len,d_head/2]
    sin = torch.sin(angles).unsqueeze(0).unsqueeze(0)

    input_1 = input[..., 0::2]  # Even indices [batch_size, seq_len, d_model/2]
    input_2 = input[..., 1::2]  # Odd indices

    rotated_input_1 = input_1 * cos - input_2 * sin # [batch_size,n_heads,seq_len,d_model/2]
    rotated_input_2 = input_1 * sin + input_2 * cos

     # [batch_size, seq_len, d_model/2, 2] -> [batch_size, seq_len, d_model]
    rotated_input = torch.stack([rotated_input_1, rotated_input_2], dim=-1).flatten(start_dim=-2, end_dim=-1)

    return rotated_input


  def forward(self,x):
    """
    Args:
      [batch_size, seq_len, d_model]
    Returns:
      [batch_size, seq_len, d_model]
    """
    # project x to Q,K,V
    Q = self.W_q(x) # [batch_size, seq_len, d_model]
    K = self.W_k(x)
    V = self.W_v(x)

    # split into multi-heads
    batch_size, seq_len, d_model = x.shape
    Q = Q.view(batch_size, seq_len,self.n_heads, self.d_head)
    Q = Q.transpose(1,2) # [batch_size, n_heads, seq_len, d_head]
    K = K.view(batch_size, seq_len,self.n_heads, self.d_head)
    K = K.transpose(1,2)
    V = V.view(batch_size, seq_len,self.n_heads, self.d_head)
    V = V.transpose(1,2)

    # RoPE
    Q = self.apply_rope(Q)# [batch_size, n_heads, seq_len, d_head]
    K = self.apply_rope(K)

    # scaled dot-product attention
    attn_scores = torch.matmul(Q,K.transpose(-2,-1)) / math.sqrt(self.d_head) # [batch_size, n_heads, seq_len, seq_len]

    # causal mask: lower_tri with 1 on diagnal
    causal_mask = torch.ones(seq_len, seq_len, device=x.device).tril().bool() # [1, seq_len, seq_len]
    attn_scores = attn_scores.masked_fill(mask = causal_mask==0, value = -1e9)# [batch_size, n_heads, seq_len, seq_len]
    # softmax to make mean of any [batch_size,seq_len,:] is 1
    attn_scores = torch.softmax(attn_scores,dim=-1)
    # output = attn weighted sum of V
    attn_output = torch.matmul(attn_scores,V) # [batch_size, n_heads, seq_len, seq_len] @ [batch_size, n_heads, seq_len, d_head] = [batch_size, n_heads, seq_len, d_head]

    # merge all heads
    # [batch, n_heads, seq_len, d_head] -> [batch, seq_len, n_heads, d_head] -> [batch, seq_len, d_model]
    attn_output = attn_output.transpose(1,2).reshape(batch_size, seq_len, d_model)

    # output projection: to blend multiple heads
    output = self.W_o(attn_output)

    return output
```


```python
class FeedForward(nn.Module):
  def __init__(self,d_model,d_ff):
    super().__init__()
    self.fc1 = nn.Linear(in_features=d_model,out_features=d_ff)
    self.fc2 = nn.Linear(in_features=d_ff,out_features=d_model)
    self.relu = nn.ReLU()

  def forward(self,x):
    """
    Args:
      x: [batch_size, seq_len, d_model]
    Returns:
      [batch_size, seq_len, d_model]
    """
    return self.fc2(self.relu(self.fc1(x)))
```


```python
class MoEFeedForward(nn.Module):
  def __init__(self,d_model,d_ff,num_experts=8, top_k=2):
    super().__init__()
    self.d_model = d_model
    self.num_experts = num_experts
    self.top_k = top_k

    self.router = nn.Linear(in_features=d_model,out_features=num_experts)
    self.experts = nn.ModuleList([FeedForward(d_model,d_ff) for _ in range(num_experts)])

  def forward(self,x):
    """
    Args:
      x: [batch_size, seq_len, d_model]
    Returns:
      output: [batch_size, seq_len, d_model]
      aux_loss: load_balancing loss
    """
    batch_size, seq_len, d_model = x.shape

    # 1. routing decision
    router_logits = self.router(x) # [batch_size, seq_len, num_experts]
    router_probs = torch.softmax(router_logits,dim=-1) # [batch_size, seq_len, num_experts]

    # 2. pick top-k experts
    # top_k_probs: [batch_size, seq_len, top_k]
    # top_k_indices: [batch_size, seq_len, top_k]
    top_k_probs, top_k_indices = torch.topk(router_probs,self.top_k,dim=-1)
    # redo the normalization on selected
    top_k_probs= top_k_probs / top_k_probs.sum(dim=-1,keepdim=True) # [batch_size, seq_len, top_k]

    # 3. compute expert output
    # flat releavant tensors
    x_flat = x.view(-1,d_model) # [batch_size*seq_len, d_model]
    top_k_indices_flat = top_k_indices.view(-1, self.top_k) # [batch_size*seq_len, top_k]
    top_k_probs_flat = top_k_probs.view(-1, self.top_k)# [batch_size*seq_len, top_k]

    # initialize
    output_flat = torch.zeros_like(x_flat)

    # loop each expert
    for expert_idx in range(self.num_experts):
      expert_mask = (top_k_indices_flat == expert_idx) # [batch_size*seq_len, top_k] bool
      # find token indices that chose this expert
      # [batch_size*seq_len, top_k] -> [batch_size*seq_len]->[num_tokens_for_this_expert]
      token_indices = expert_mask.any(dim=-1).nonzero(as_tuple=True)[0]

      # num of element is 0 (no token picked this expert)
      if token_indices.numel() == 0:
        continue

      # expert FFN output
      expert_input = x_flat[token_indices]  # [num_tokens_for_this_expert, d_model]
      expert_output = self.experts[expert_idx](expert_input)  # [num_tokens_for_this_expert, d_model]

      # expert weight
      expert_weights = top_k_probs_flat[token_indices]  # [num_tokens_for_this_expert, top_k]
      expert_mask_for_selected = expert_mask[token_indices]  # [num_tokens_for_this_expert, top_k] bool
      expert_weights = (expert_weights * expert_mask_for_selected).sum(dim=-1, keepdim=True)  # [num_tokens_for_this_expert, 1]

      # weighted sum to output
      output_flat[token_indices] += expert_weights * expert_output

    # 4. reshape
    output = output_flat.view(batch_size, seq_len, d_model)

    # 5. aux. loss for load balancing
    aux_loss = self._compute_load_balancing_loss(top_k_indices)

    return output, aux_loss

  def _compute_load_balancing_loss(self,top_k_indices):
    """
    Args:
      top_k_indices: [batch_size, seq_len, top_k]
    Returns:
      aux_loss: scalar
    """
    batch_size, seq_len, _ = top_k_indices.shape
    num_tokens = batch_size * seq_len

    # count how many times each expert is chosen
    expert_counts = top_k_indices.view(-1).bincount(minlength=self.num_experts).float() # [num_experts]
    expert_freq = expert_counts / (num_tokens * self.top_k)  # [num_experts]

    ideal_freq = 1.0 / self.num_experts
    aux_loss = ((expert_freq - ideal_freq) ** 2).sum()

    return aux_loss
```


```python
class DecoderLayer(nn.Module):
  def __init__(self,d_model,d_ff,n_heads,use_moe,num_experts,top_k):
    super().__init__()
    self.self_attn = Attention(d_model,n_heads)
    self.norm1 = nn.LayerNorm(d_model)

    self.use_moe = use_moe
    if use_moe:
      self.feed_forward = MoEFeedForward(d_model,d_ff,num_experts,top_k)
    else:
      self.feed_forward = FeedForward(d_model,d_ff)

    self.norm2 = nn.LayerNorm(d_model)

  def forward(self, x):
    """
    Args:
      x: [batch_size, seq_len, d_model]
    Returns:
      x=x->attn->ff: [batch_size, seq_len, d_model]
    """
    # self-attn
    attn_output = self.self_attn(x) # [batch_size, seq_len, d_model]
    # residual connection
    x = x + attn_output # do not use +=
    # normalization through each row (make each token representation of zero-mean and unit variance)
    x =  self.norm1(x)
    # feed forward
    if self.use_moe:
      ff_output,aux_loss = self.feed_forward(x) # [batch_size, seq_len, d_model]
    else:
      ff_output = self.feed_forward(x) # [batch_size, seq_len, d_model]
      aux_loss = None
    # # residual connection + normalization
    x = self.norm2(x + ff_output)
    return x, aux_loss
```


```python
class DecoderOnlyTransformer(nn.Module):
  def __init__(self,vocab_size,d_model,num_layers,d_ff,n_heads,use_moe=False, num_experts=8,top_k=2):
    super().__init__()
    self.embedding = nn.Embedding(vocab_size,d_model)
    self.use_moe = use_moe
    self.decoder_layers = nn.ModuleList([DecoderLayer(d_model,d_ff,n_heads,use_moe,num_experts,top_k) for _ in range(num_layers)])
    self.norm = nn.LayerNorm(d_model)
    self.fc= nn.Linear(d_model,vocab_size)

  def forward(self,input_ids):
    """
    Args:
      input_ids: [batch_size, seq_length]
    Returns:
      logits: [batch_size, seq_length, vocab_size]
    """
    # embedding
    x = self.embedding(input_ids)

    total_aux_loss = 0
    # decoder layer
    for dec_layer in self.decoder_layers:
      x,aux_loss = dec_layer(x)
      if aux_loss:
        total_aux_loss += aux_loss
    # norm
    x = self.norm(x)
    # fc
    logit = self.fc(x) # [[batch_size, seq_length, vocab_size]]

    if self.use_moe:
      return logit, total_aux_loss/len(self.decoder_layers)
    else:
      return logit, None
```


## Train with Autograd


```python
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
```


```python
# Config
SEQ_LEN = 128
BATCH_SIZE = 16
D_MODEL = 64
NUM_LAYERS = 4
NUM_HEADS = 2
D_FF = 128
LR = 3e-4
EPOCHS = 3
MAX_SAMPLES = 5000
USE_MOE = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
```

```text
Device: cuda
```


```python
# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
vocab_size = tokenizer.vocab_size
```

```text
/usr/local/lib/python3.12/dist-packages/huggingface_hub/utils/_auth.py:93: UserWarning: 
The secret `HF_TOKEN` does not exist in your Colab secrets.
To authenticate with the Hugging Face Hub, create a token in your settings tab (https://huggingface.co/settings/tokens), set it as secret in your Google Colab and restart your session.
You will be able to reuse this secret in all of your notebooks.
Please note that authentication is recommended but still optional to access public models or datasets.
  warnings.warn(
```

```text
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
WARNING:huggingface_hub.utils._http:Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
```


```python
# Load dataset
print("Loading dataset...")
raw_data = load_dataset('roneneldan/TinyStories', split=f'train[:{MAX_SAMPLES}]')

# Tokenize all texts
print("Tokenizing...")
all_tokens = []
for item in tqdm(raw_data):
    tokens = tokenizer.encode(item['text'], max_length=SEQ_LEN+1, truncation=True)
    if len(tokens) > SEQ_LEN:  # only keep sequences that are long enough
        all_tokens.append(tokens[:SEQ_LEN+1])

print(f"Total sequences: {len(all_tokens)}")
```

```text
Loading dataset...
```

```text
Tokenizing...
```

```text
Total sequences: 4761
```


```python
# create model
model = DecoderOnlyTransformer(
    vocab_size=vocab_size,
    d_model=D_MODEL,
    num_layers=NUM_LAYERS,
    d_ff=D_FF,
    n_heads=NUM_HEADS,
    use_moe = USE_MOE
).to(device)

num_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {num_params:,}")
```

```text
Parameters: 7,083,377
```


```python
# Optimizer and loss
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

# Training loop
print("\nTraining...")
model.train()

for epoch in range(EPOCHS):
    total_loss = 0
    num_batches = 0

    # Shuffle data each epoch
    import random
    random.shuffle(all_tokens)

    # Process in batches
    for i in tqdm(range(0, len(all_tokens), BATCH_SIZE)):
        # Get batch
        batch = all_tokens[i:i+BATCH_SIZE]
        if len(batch) < BATCH_SIZE:
            continue

        # Convert to tensors: input = tokens[:-1], target = tokens[1:]
        x = torch.tensor([seq[:-1] for seq in batch], dtype=torch.long).to(device)  # [B, SEQ_LEN]
        y = torch.tensor([seq[1:] for seq in batch], dtype=torch.long).to(device)   # [B, SEQ_LEN]

        # Forward pass
        logits,aux_loss = model(x)  # [B, SEQ_LEN, vocab_size]

        # Calculate loss
        loss = criterion(logits.reshape(-1, vocab_size), y.reshape(-1)) # to flatten the batch and sequence dimensions
        if aux_loss is not None:
          loss += 0.01*aux_loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}")

    # Generate sample text
    print("-" * 50)
    model.eval()
    prompt = "Once upon a time"
    input_ids = tokenizer.encode(prompt)
    context = torch.tensor(input_ids).unsqueeze(0).to(device)  # [1, prompt_len]

    # Generate 50 tokens
    for _ in range(50):
        with torch.no_grad():
            logits, aux_loss = model(context)  # [1, seq_len, vocab_size]
            next_logits = logits[0, -1, :]  # [vocab_size]
            next_token = torch.argmax(next_logits).unsqueeze(0).unsqueeze(0)  # [1, 1]
            context = torch.cat([context, next_token], dim=1)

    generated = tokenizer.decode(context[0].tolist())
    print(f"Generated: {generated}")
    print("-" * 50)
    model.train()

print("\nDone!")
```

```text

Training...
```

```text
Epoch 1/3, Loss: 6.9988
--------------------------------------------------
Generated: Once upon a time, there was a little girl. She was a little girl. She was so the little girl.
--------------------------------------------------
```

```text
Epoch 2/3, Loss: 4.7406
--------------------------------------------------
Generated: Once upon a time, there was a little girl named Lily. She was very happy and the park. One day, she was very happy. One day, she was so excited to the park.

The little girl was so excited to the park. She was
--------------------------------------------------
```

```text
Epoch 3/3, Loss: 4.1058
--------------------------------------------------
Generated: Once upon a time, there was a little girl named Lily. She loved to play with her mommy. One day, she was very happy to the park. She was very happy and she had a big, so she was very happy.

One day,
--------------------------------------------------

Done!
```


```python
# to save model
torch.save(model.state_dict(), 'model.pt')
```


