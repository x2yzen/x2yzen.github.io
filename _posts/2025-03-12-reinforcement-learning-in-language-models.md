---
title: Reinforcement Learning in Language Models
date: 2025-03-12 17:00:00 +0800
categories: [language_model]
tags: [llm,rl]
pin: false
math: true
---

A comprehensive overview of reinforcement learning techniques used in language models

## Overview

![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/wb1.png)

1. PPO and GRPO are RL algorithm variants that have evolved through many iterations. They're difficult to understand because they're quite far from the original forms taught in RL tutorials (grid-walking + Bellman equations), but those original forms indeed provide the best intuition.
2. Working backwards from the results, we need to bridge at least two gaps to connect the storyline:

   - LM RL → deep RL: Language models are typically viewed as policies in deep RL, so both PPO and GRPO belong to the policy-gradient family of methods in deep RL
   - deep RL → classic RL: Deep RL uses parameterized policy functions with optimality expressed as scalar functions, driven by gradient optimization algorithms to search for optimal policies; while the easily understood classic RL uses tabular policies with optimality expressed as Bellman optimal conditions, driven by iterative methods to explore optimal policies. They connect through value function estimation methods needed for policy gradient calculations (TD-learning → Monte Carlo learning → iterative methods → BOE)

## Proximal Policy Optimization (PPO)

Starting from _advantage actor-critic (A2C)_, adding KL divergence as a penalty term or CLIP to control that the objective function after iteration doesn't differ too much from the original function, we get Proximal Policy Optimization (PPO).

### Theory

Previous actor-critic algorithms focused solely on maximizing reward model outputs, which causes problems in the language model context. Reward models are trained on far fewer samples than pretraining, so their knowledge can be sparse and irregular, making them easy to hack. For example, meaningless sequences (thethethethe... or lots of emojis) often mysteriously gain favor from reward models. Optimizing language models toward reward models without constraints eventually leads to nonsensical outputs. This gave birth to PPO and TRPO (trust-region policy optimization), algorithms that essentially require the post-iteration policy to not differ too much from the pre-iteration policy. The most obvious penalty would be using KL divergence as a penalty term, which can indeed be done. Let's first introduce a more commonly used and simpler implementation: PPO-clip (reportedly favored by OpenAI).

$$
\theta_{k+1} = \arg\max_{\theta} \mathbb{E}_{s, a \sim \pi_{\theta_k}} \left[ \mathbb{E}_{\theta_k} \left[ L\left(s, a, \theta_k, \theta\right) \right] \right]
$$

where

$$
L\left(s, a, \theta_{k}, \theta\right) = \min\left(\frac{\pi_{\theta}(a \mid s)}{\pi_{\theta_{k}}(a \mid s)} A^{\pi_{\theta_{k}}}(s, a), \quad g\left(\epsilon, A^{\pi_{\theta_{k}}}(s, a)\right)\right)
$$

$$
g(\epsilon, A) = \left\{
\begin{array}{ll}
(1 + \epsilon) A & \text{if } A \geq 0, \\
(1 - \epsilon) A & \text{if } A < 0.
\end{array}
\right.
$$

This looks complex, so let's break it down:

If the current (s,a) advantage > 0, the equation becomes:

$$
L\left(s, a, \theta_{k}, \theta\right) = \min\left(\frac{\pi_{\theta}(a \mid s)}{\pi_{\theta_{k}}(a \mid s)}, (1+\epsilon)\right) A^{\pi_{\theta_{k}}}(s, a)
$$

Comparing to equation (10.7), it's almost identical (after taking gradients, the denominator can be absorbed to become log). The main difference is the additional min($\epsilon$) term → this is the "clip" meaning, using this hyperparameter to cap the gains and avoid taking steps that are too large.

### Implementation

![](/assets/images/2025-03-12-reinforcement-learning-in-language-models/SO1HbDlfmosLJJx46ZMjVJSkprf.png)

Referring to the specific implementation of trl-ppo_trainer ([https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L117](https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L117)):

1. Starting from the step function at line 655, the inputs are:
   - queries (batch_size, q_seq_len)
   - responses (batch_size, resp_seq_len)
   - scores (batch_size,)

2. Line 747: [AutoModelForCausalLMWithValueHead](https://github.com/huggingface/trl/blob/v0.11.2/trl/models/modeling_value_head.py#L61) performs forward pass, computing conditional probability _all_logprobs_ (batch_size, q_seq_len+resp_seq_len) and _values_ (batch_size, q_seq_len+resp_seq_len) for each position of each sample (q+resp). Of course, since the q part doesn't count, these values will be masked out in subsequent calculations.

	```python
with torch.no_grad():
all_logprobs, logits_or_none, values, masks = self.batched_forward_pass(
self.model,
queries,
responses,
model_inputs,
response_masks=response_masks,
return_logits=full_kl_penalty,)
	```

3. Line 777's compute_reward function calculates rewards for each position in each sample (batch_size, q_seq_len+resp_seq_len):
	```python
	rewards, non_score_reward, kls = self.compute_rewards(scores, all_logprobs, ref_logprobs, masks)
	```

	Specifically, for each sample, non_score_reward (q_seq_len+resp_seq_len,) is derived from the logprob difference between the current model and reference model (model without RL). The simplest approach is element-wise subtraction. Line 11 adds the reward model score at the last non-zero mask position (end of response) to get the final reward. This is intuitive: since current reward models only score complete sequences, before the final state, rewards only contain penalty terms (the greater the difference from original distribution, the more negative), until the trajectory reaches completion, where this state gets an additional score from the reward model.

	```python
	for score, logprob, ref_logprob, mask in zip(scores, logprobs, ref_logprobs, masks):
	# compute KL penalty (from difference in logprobs)
	kl = self._kl_penalty(logprob, ref_logprob)
	kls.append(kl)
	non_score_reward = -self.kl_ctl.value * kl
	non_score_rewards.append(non_score_reward)
	reward = non_score_reward.clone()
	last_non_masked_index = mask.nonzero()[-1]
	
	# reward is preference model score + KL penalty
	reward[last_non_masked_index] += score
	rewards.append(reward)
	```

4. Back to the step function, line 781's compute_advantages function calculates the crucial advantages:
	```python
values, advantages, returns = self.compute_advantages(values, rewards, masks)
	```

	Specifically: the loop starts from the last state of each sample. Since there's no next state, line 2 sets nextvalues to 0. Following the previously discussed GAE, this state's delta is directly estimated as reward minus current state value (line 4). In line 5, since lastgaelam's initial value is 0, this state's advantage is directly estimated using delta:
	$$\hat{A}_{t}^{(1)} := \delta_{t}^{V} = r_t + \gamma V(s_{t+1}) - V(s_t)$$
	
	Moving to the next loop calculating the second-to-last state, its nextvalues equals the final state's value, delta is estimated according to the above formula, while L5 reflects the GAE estimator
	$$\sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}^V$$
	with upper limit l=1, meaning it uses weighted combination of $$\delta_{t+1}^V$$ and $$\delta_{t}^V$$ for bias-variance tradeoff. Subsequent loops follow similar logic, yielding advantages for each position of each sample (batch_size, q_seq_len+resp_seq_len).

	```python
	lastgaelam = 0
	for t in reversed(range(gen_len)):
	nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
	delta = rewards[:, t] + self.config.gamma * nextvalues - values[:, t]
	lastgaelam = delta + self.config.gamma * self.config.lam * lastgaelam
	advantages_reversed.append(lastgaelam)
	```

5. Line 832 begins actual training:
	```python
	train_stats = self.train_minibatch(...)
	```

	Focus on loss calculation, which has two components:
	- First is value function loss, measuring how accurately the value function predicts returns-to-go for each position of each sample, with clipping applied
	- Second is policy gradient loss, computing for each position of each sample:
	$$\mathbb{E}_{\tau \sim \pi_{\theta}}\left[\sum_{t=0}^{T}\frac{\pi_{\theta}(a \mid s)}{\pi_{\theta_{k}}(a \mid s)}A^{\pi_{\theta}}\left(s_{t}, a_{t}\right)\right]$$
	with clipping applied
	- The masked_mean function converts these losses to scalars by averaging over all non-zero mask positions

	```python
	vf_losses1 = (vpreds - returns) ** 2
	vf_losses2 = (vpredclipped - returns) ** 2
	vf_loss = 0.5 * masked_mean(torch.max(vf_losses1, vf_losses2), mask)
	vf_clipfrac = masked_mean(torch.gt(vf_losses2, vf_losses1).float(), mask)
	
	ratio = torch.exp(logprobs - old_logprobs)
	
	pg_losses = -advantages * ratio
	pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - self.config.cliprange, 1.0 + self.config.cliprange)
	
	pg_loss = masked_mean(torch.max(pg_losses, pg_losses2), mask)
	pg_clipfrac = masked_mean(torch.gt(pg_losses2, pg_losses).float(), mask)
	
	loss = pg_loss + self.config.vf_coef * vf_loss
	
	def masked_mean(values: torch.Tensor, mask: torch.Tensor, axis: Optional[bool] = None) -> torch.Tensor:
	"""Compute mean of tensor with a masked values."""
	if axis is not None:
	return (values * mask).sum(axis=axis) / mask.sum(axis=axis)
	else:
	return (values * mask).sum() / mask.sum()
	```

### Demo

Using the [yelp_review_full](https://huggingface.co/datasets/Yelp/yelp_review_full) corpus, truncating the beginning and having the model continue writing, with the goal of using RLHF to force positive reviews (even if the original was 0 stars, it has to be spun positively =.=). For proof of concept convenience, I chose the relatively new small model [Qwen2.5-1.5B-base](https://huggingface.co/Qwen/Qwen2.5-1.5B), running on Tesla L4 24G *1.

![](/assets/images/2025-03-12-reinforcement-learning-in-language-models/OrXmbRuvgoo9jrxFJPHjlnUppYf.png)

Before RL, I did an SFT stage using the same dataset, fine-tuning ~1% of Qwen2.5-1.5B-base parameters with r=128 LoRA to familiarize the model with Yelp review style. Experience shows this makes continuations more natural, and metrics show some effectiveness ([script](https://code.byted.org/renxinyuyang/llm-trials/blob/dev/sft-lora.py), [run log](https://wandb.ai/x2yzen-freelance/huggingface/runs/ymr65k8x?nw=nwuserx2yzen)).

![](/assets/images/2025-03-12-reinforcement-learning-in-language-models/MYI7bNvaooMmnPxkTSrjysOOpHf.png)

In the RL stage, I used an existing text sentiment classification model ([distilbert-base-multilingual-cased-sentiments-student](https://huggingface.co/lxyuan/distilbert-base-multilingual-cased-sentiments-student)) as RM, using the POSITIVE label logit as reward. I used [TRL](https://huggingface.co/docs/trl/index) (Transformer Reinforcement Learning)'s [PPO trainer](https://huggingface.co/docs/trl/ppo_trainer) to complete one RLHF round, aligning the fine-tuned Qwen2.5-1.5B from the previous step ([script](https://code.byted.org/renxinyuyang/llm-trials/blob/dev/sentiment-rl.py), [run log](https://wandb.ai/x2yzen-freelance/trl/runs/g82qp1tv?nw=nwuserx2yzen)).

As training progresses, the model's output reward scores gradually increase and stabilize, indicating the model more consistently outputs positive responses.

![](/assets/images/2025-03-12-reinforcement-learning-in-language-models/M2GEb5jbxoejONx3ntljxTj8pFe.png)

The difference from the original model also rises then remains stable.

![](/assets/images/2025-03-12-reinforcement-learning-in-language-models/MKusbh29GocdW9xhfqEj3D8upsh.png)

The metrics overall meet expectations. Let's examine some typical cases to see how the model's responses to the same input changed before and after training:

![](/assets/images/2025-03-12-reinforcement-learning-in-language-models/EzqpbHpaUoszHSxJn4xjfWV3pmc.png)

![](/assets/images/2025-03-12-reinforcement-learning-in-language-models/Andxb6lhGo4qqoxya1ejCim3pne.png)

![](/assets/images/2025-03-12-reinforcement-learning-in-language-models/SUHpblFUcoCuaUx4WhVjbmfBpsc.png)

### Discussion

Unlike SFT's question→answer training paradigm, RLHF is a comparison-based training stage. From one perspective, this training mode's benefit is mainly reducing sample acquisition costs, even making training standards that are difficult to explicitly describe possible. In many scenarios, having human annotators "generate" preference-compliant training samples is expensive, or even nearly impossible due to annotator limitations and subtle standards that can't be completely expressed in language. But if provided with several samples and having trained annotators judge which they prefer more, the cost is much lower (imagine training a generative model to draw handsome men and beautiful women - having annotators draw handsome sketches one by one as training samples versus just judging which of several model-generated faces is more handsome - the cost difference is obvious).

In current LLM pipelines, the general sequence is pretrain→SFT→RL. The mainstream view is that most model knowledge and capabilities are acquired during pretraining, SFT only teaches the model specific formats to answer specific questions, while RL doesn't create new capabilities or formats but ensures effect stability. In other words, whether a model can answer a question is largely determined at the base model stage - if the intelligence isn't sufficient, subsequent training won't help much; if intelligence is adequate, after SFT the model gains response patterns more suitable for specific scenarios, but the probabilistic model nature means these capabilities and patterns won't be 100% stable in every response. Furthermore, the RL stage uses preference alignment to suppress non-preferred sample probabilities and amplify effect stability, ultimately becoming a checkpoint ready for real-world deployment.

## Group Relative Policy Optimization (GRPO)

### Theory

This new concept became famous through deepseek-R1 (https://arxiv.org/abs/2501.12948), but the actual changes are minimal. Observe the following formula:

![](/assets/images/2025-03-12-reinforcement-learning-in-language-models/IuZobftE6oXxNox62QDjozd3pGg.png)

Compared to CLIP-PPO, there are only two differences:

1. Besides CLIP, it adds KL divergence as a penalty term to further constrain the difference between target and reference models

2. It removes the value network needed for advantage estimation (complex, expensive, and easily hacked), replacing it with normalized reward means from multiple response samples to calculate advantage. This returns to the more primitive Monte Carlo approach, where advantage is estimated by the difference between this sample's reward and the average reward across multiple samples, yielding _GRPO_.

### Implementation 

TRL already includes [GRPOTrainer](https://huggingface.co/docs/trl/v0.16.0/grpo_trainer#quick-start) implementation, divided into four clear stages:

![](/assets/images/2025-03-12-reinforcement-learning-in-language-models/Z20cbBDPGoqmZJxAuj6jJB3ppfe.png)

1. **generating completions**: sample a batch of prompts and generate a set of _G_ (default 8) completions for each prompt (denoted as _oi_).

2. **computing the advantage**: 
	- for each of the _G_ sequences, compute the reward using a reward model.
	- calculated relative comparisons and do normalization
		![](/assets/images/2025-03-12-reinforcement-learning-in-language-models/U7OobR0PAoAkGkxLLOqjoAm4p2b.png)

3. **estimating the KL divergence**: KL divergence is estimated using the approximator introduced by <u>Schulman et al. (2020)</u>. 
	![](/assets/images/2025-03-12-reinforcement-learning-in-language-models/FOydbG8gLoEuILxi3IpjDVapp6g.png)

4. **computing the loss**
	![](/assets/images/2025-03-12-reinforcement-learning-in-language-models/LcERbtbfSo32zmxzjEKjIZzqpKh.png)

Usage is relatively simple, though actual training requires consideration of integration with distributed frameworks (deepspeed) and inference frameworks (vllm) for efficiency:

```python
# train_grpo.py

from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer

dataset = load_dataset("trl-lib/tldr", split="train")

# Define the reward function, which rewards completions that are close to 20 characters

def reward_len(completions, **kwargs):
return [-abs(20 - len(completion)) for completion in completions]

training_args = GRPOConfig(output_dir="Qwen2-0.5B-GRPO", logging_steps=10)
trainer = GRPOTrainer(
model="Qwen/Qwen2-0.5B-Instruct",
reward_funcs=reward_len,
args=training_args,
train_dataset=dataset,
)
trainer.train()
```  

### Demo

#### Dataset

Using a [countdown game dataset](https://huggingface.co/datasets/Jiayi-Pan/Countdown-Tasks-3to4), with rules similar to the 24-point game - use 3 numbers with basic arithmetic operations to create a target number.

![](/assets/images/2025-03-12-reinforcement-learning-in-language-models/XHGObV0asofIhEx87A0jAheGp5f.png)

Using the dataset to construct training prompts in the following format:

```
## ROLE

You are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer to the question.

## QUESTION

Using the numbers [79, 17, 60], create an equation that equals 36.

# REQUIREMENTS

1. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once.
2. First show your thinking process between <think> </think> tags. And return the final answer equation between <answer> </answer> tags
3. Be concise and clear, your output length is limited to ~300 words.
   output example: '<think>your thinking process</think> thus the answer is <answer> (1 + 2) / 3 </answer>'
```

#### Reward 

Based on regex completion of model output strings, simply put:

1. **Format score**: Judge whether output contains necessary '<answer> </answer>' format - if yes, get 0.1 points, otherwise 0 points directly
2. **Correctness score**: Extract the equation within '<answer> </answer>' tags and eval it - if it meets requirements and result is correct, get 0.9 points

Combined effect: completely correct gets 1 point, correct format but wrong equation gets 0.1 points, everything else gets 0 points.

#### Training

Training overview:

- model: [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
- epoch=1
- batch_size = 16 * 8
- learning_rate = 2e-6
- Resource usage: H20*8*50hrs
- log: [wandb](https://wandb.ai/anUsualTeamName/o1-replica/runs/fz4w9xpj/workspace?nw=nwuserx2yzen)

Observing training metrics, they basically meet expectations:

![](/assets/images/2025-03-12-reinforcement-learning-in-language-models/Ghkib7DOhoL4mXxGdykjQGb1p1c.png)

Case analysis also shows that after 1 epoch of training, the model has better handling of both the form and content of this task:

![](/assets/images/2025-03-12-reinforcement-learning-in-language-models/QZ9jbPq3foXatzxLEfxjLSIJpZd.png)

![](/assets/images/2025-03-12-reinforcement-learning-in-language-models/Wy1hbPux4opfMFxkrvdjE3KCpgc.png)