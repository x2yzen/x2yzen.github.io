---
title: Reinforcement Learning in Language Models
date: 2025-03-12 17:00:00 +0800
categories: [language_models]
tags: [llm,rl]
pin: false
math: true
---

整理在语言模型中使用的强化学习技术

## Overview

![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/wb1.png)

1. PPO 和 GRPO 都是演进了非常多步的 RL 算法变种，不好理解的原因是它们与各种 RL 教程的原始形态（走格子 +Bellman equation）相距太远，但确实只有原始形态的算法才具备最好的 intuition
2. 从结果出发向前推，需要弥补至少两个 gap 才能将故事线串起来

   - LM RL -> deep RL：LM 通常被视为 deep RL 里的 policy，因此 PPO 和 GRPO 都属于 deep RL 中 policy-gradient 家族的方法；
   - deep RL -> classic RL：deep RL 的策略函数是参数化的，optimality 表示为某个标量函数，通过梯度优化算法驱动对最优策略的搜索；而容易理解的经典 RL 策略是表格化的，optimality 表示为 Bellman optimal condition，通过迭代方法驱动最优策略的探索；他们之间通过 policy gradient 梯度计算时需要的对价值函数的估计方法(TD-learning->Monte carlo learning -> iterative methods -> BOE)连接起来

## Proximal Policy Optimization (PPO)

从_advantage actor-critic (A2C)_开始，加上 KL-Div 作为惩罚项或者 CLIP，控制迭代后的目标函数和原函数差别不要太大，就得到了 Proximal Policy Optimization (PPO)  

### Theory

之前的 actor-critic 算法将优化目标单一地定为追求 reward model 的最大化，在语言模型的语境下会导致一些问题，reward model 训练样本量远少于 pretrain，它的知识可能是稀疏并且无规律的，比较容易被 hack，比如经常可以发现一段无意义的语句（thethethethethe...或者一堆 emoji）会莫名其妙地获得 reward model 的青睐，而将语言模型无限制地向 reward model 优化，最终也会导致胡言乱语，因此诞生了 PPO 和 TRPO (trust-region policy optimization) 这一类算法，本质上都是要求 RL 迭代后的策略和迭代前差距不要过大。最容易想到的 penalty 当然是用 KL divergence 作为惩罚项，事实上也确实可以这么做，先介绍一个更常用也更简单的实现：PPO-clip（据说 openai 比较惯用这种形式）。

$$
\theta_{k+1} = \arg\max_{\theta} \mathbb{E}_{s, a \sim \pi_{\theta_k}} \left[ \mathbb{E}_{\theta_k} \left[ L\left(s, a, \theta_k, \theta\right) \right] \right]
$$

其中

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

看起来挺复杂，拆解一下来理解：

如果当前(s,a)的 advantage>0，上面的式子变成：

$$
L\left(s, a, \theta_{k}, \theta\right) = \min\left(\frac{\pi_{\theta}(a \mid s)}{\pi_{\theta_{k}}(a \mid s)}, (1+\epsilon)\right) A^{\pi_{\theta_{k}}}(s, a)
$$

对比一下式(10.7)，几乎完全一样（求梯度后分母可以收进去变成 log），主要的不同其实就是多了一个 min($\epsilon
$)项 -> 这个就是 clip 的意思，用这个超参给收益加上一个 cap，避免步子太大。

### Implementation

![](/assets/images/2025-03-12-reinforcement-learning-in-language-models/SO1HbDlfmosLJJx46ZMjVJSkprf.png)

参考 trl-ppo_trainer（ [https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L117](https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L117)）的具体实现：

1. 从 655 行的 step 函数开始看起，输入是：

   - queries (batch_size, q_seq_len)
   - responses  (batch_size, resp_seq_len)
   - scores (batch_size,)
2. 747 行 [AutoModelForCausalLMWithValueHead](https://github.com/huggingface/trl/blob/v0.11.2/trl/models/modeling_value_head.py#L61) 进行 forward，计算出了每一个样本（q+resp）每一位的 conditional probability _all_logprobs _(batch_size, q_seq_len+resp_seq_len)和_values _(batch_size, q_seq_len+resp_seq_len)，当然，由于 q 部分的数字都不做数，在后续计算的时候它们会被_mask_掩盖掉


```python
with torch.no_grad():
all_logprobs, logits_or_none, values, masks = self.batched_forward_pass(
self.model,
queries,
responses,
model_inputs,
response_masks=response_masks,
return_logits=full_kl_penalty,
)

```

3. 777行的compute_reward函数计算出了每个样本中，每一个位置的rewards (batch_size, q_seq_len+resp_seq_len)

```python
rewards, non_score_reward, kls = self.compute_rewards(scores, all_logprobs, ref_logprobs, masks)
```

具体来看，对于每个样本，non_score_reward (q_seq_len+resp_seq_len,) 由当前模型与参考模型（没有经过RL的模型）下该样本的logprob差异得到（最简单的做法就是按位减法），第11行在最后一个mask非0位（也就是resp的结尾）加上了reward model提供的score，成为了最终的reward——容易理解，由于目前的rewards model只对一个完成的序列进行打分，因此在最后一个状态之前，reward都只有惩罚项（与原始分布的区别越大，负的越多），直到轨迹达到完成状态，这个状态额外加上一个reward model给出的评分

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

4. 回到step函数，781行compute_advantages函数计算了关键的advantages
	```
values, advantages, returns = self.compute_advantages(values, rewards, masks)
```

具体来看：loop从每个样本的最后一个状态开始，由于已经没有下一个state，因此第2行nextvalues为0，按照前一节讨论过的GAE，该状态的delta直接估计为回报减去本状态的价值（第4行），而第5行中由于lastgaelam初始值为0，该状态的advantage直接用delta估计
$$\hat{A}_{t}^{(1)} := \delta_{t}^{V} = r_t + \gamma V(s_{t+1}) - V(s_t)$$
进入下一个循环，计算倒数第二个状态，它的nextvalues等于最终状态的value，delta也按照上式进行估计，而L5则体现了GAE估计量（
$\sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}^V$）l上限取1的形式，也就是说一定程度上使用$\delta_{t+1}^V$和$\delta_{t}^V$加权的方式进行了bias和variance的tradeoff。后续循环逻辑类似，这样就得到了每个样本每一位的advantages (batch_size, q_seq_len+resp_seq_len)

```python
lastgaelam = 0
for t in reversed(range(gen_len)):
nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
delta = rewards[:, t] + self.config.gamma * nextvalues - values[:, t]
lastgaelam = delta + self.config.gamma * self.config.lam * lastgaelam
advantages_reversed.append(lastgaelam)

```

5. 832行开始实际的训练
```python
train_stats = self.train_minibatch(...)
```

主要关注一下loss的计算的方式，分为两项：
- 第一项是value function loss，衡量每个样本每一位上value function的对rtg预测的准确程度，并且进行了clip；
- 另一项是policy gradient loss，计算的是每个样本每一位的$\mathbb{E}_{\tau \sim \pi_{\theta}}\left[\sum_{t=0}^{T}\frac{\pi_{\theta}(a \mid s)}{\pi_{\theta_{k}}(a \mid s)}A^{\pi_{\theta}}\left(s_{t}, a_{t}\right)\right]$，并且进行了clip；
- masked_mean函数以所有mask非0位取平均的方式，将上述loss转化为标量

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

使用[yelp_review_full](https://huggingface.co/datasets/Yelp/yelp_review_full)语料库，截取开头让模型续写，目标是通过RLHF来强行输出好评（即使原来是0分也得圆回来=。=）。概念验证方便起见，模型选择了比较新的小尺寸模型[Qwen2.5-1.5B-base](https://huggingface.co/Qwen/Qwen2.5-1.5B)，硬件是Tesla L4 24G *1

![](/assets/images/2025-03-12-reinforcement-learning-in-language-models/OrXmbRuvgoo9jrxFJPHjlnUppYf.png)

在RL之前先做了一个SFT阶段，用同样的数据集，使用r=128 Lora微调了Qwen2.5-1.5B-base大约~1%的参数，让模型熟悉yelp review的画风，经验表明这样续写出来会更自然一些，从指标看也有一定的效果（[script](https://code.byted.org/renxinyuyang/llm-trials/blob/dev/sft-lora.py), [run log](https://wandb.ai/x2yzen-freelance/huggingface/runs/ymr65k8x?nw=nwuserx2yzen)）

![](/assets/images/2025-03-12-reinforcement-learning-in-language-models/MYI7bNvaooMmnPxkTSrjysOOpHf.png)

在RL阶段，使用一个现成的文本情感分类模型（[distilbert-base-multilingual-cased-sentiments-student](https://huggingface.co/lxyuan/distilbert-base-multilingual-cased-sentiments-student)）作为RM，以POSITIVE label的logit作为reward，使用上文读过的 [TRL](https://huggingface.co/docs/trl/index) (Transformer Reinforcement Learning) 提供的[PPO trainer](https://huggingface.co/docs/trl/ppo_trainer)来完成一次RLHF，让上一步中微调过的Qwen2.5-1.5B进行对齐（[script](https://code.byted.org/renxinyuyang/llm-trials/blob/dev/sentiment-rl.py), [run log](https://wandb.ai/x2yzen-freelance/trl/runs/g82qp1tv?nw=nwuserx2yzen)）

可以看到随着训练的进行，模型输出的reward分数逐渐增加并趋于稳定，表明模型更稳定地输出了积极的回复

![](/assets/images/2025-03-12-reinforcement-learning-in-language-models/M2GEb5jbxoejONx3ntljxTj8pFe.png)

而与原始模型的差异也经过一段上升后保持稳定

![](/assets/images/2025-03-12-reinforcement-learning-in-language-models/MKusbh29GocdW9xhfqEj3D8upsh.png)

从指标上看整体是符合预期的，来抽一些典型的case，看看训练前和训练后模型对同一个输入的反馈是如何变化的：

![](/assets/images/2025-03-12-reinforcement-learning-in-language-models/EzqpbHpaUoszHSxJn4xjfWV3pmc.png)

![](/assets/images/2025-03-12-reinforcement-learning-in-language-models/Andxb6lhGo4qqoxya1ejCim3pne.png)

![](/assets/images/2025-03-12-reinforcement-learning-in-language-models/SUHpblFUcoCuaUx4WhVjbmfBpsc.png)

### Discussion

不同于SFT的问题->答案训练范式，RLHF是一个基于对比的训练阶段，从某种角度上说，这种训练模式的收益主要是降低了样本的获取成本，甚至将一些难以明确描述的训练标准变为可能。在很多场景里，人类标注者去“生成”一个符合偏好的训练样本是昂贵的，甚至受限于标注者的能力，难以用语言完备表述的细微标准等，是接近不可能的，但如果被提供了几个样本，由受过训练的标注者判断更偏好哪个，成本就低很多了（试想训练一个生图模型来画帅哥美女，标注者自己一张一张画帅哥速写做训练样本和只是评判下这几个模型生成的人脸哪个更帅，成本不言而喻）。

在当前的LLM流水线里，一般遵从pretrain->SFT->RL的顺序，主流观点认为绝大多数模型知识和能力是在pretrain阶段获得的，SFT阶段模型只是学习特定格式（format）来回答特定问题，而RL阶段不产生新的能力也不产生新的格式，而是保证效果的稳定性。换句话说，一个问题模型有没有能力回答，很大程度上在基础模型阶段就确定了，如果智商不够不会答，后续训练希望也不大；如果智商到了，经过SFT，模型获得了更符合特定场景的回答模式，但概率模型的本质决定了这些能力和模式在每次回答时不会100%稳定；进一步地，RL阶段则通过偏好对齐，打压不符合偏好的样本出现的概率，放大了效果的稳定性，最终变成一个能够上线实际使用的ckpt。

## Group Relative Policy Optimization (GRPO)

### Theory

由于deepseek-R1（https://arxiv.org/abs/2501.12948）而闻名于世的新概念，实际上的改动也非常小，观察下式：

![](/assets/images/2025-03-12-reinforcement-learning-in-language-models/IuZobftE6oXxNox62QDjozd3pGg.png)

实际上对比CLIP-PPO就只有两点区别：

1. 除了CLIP，还加上了一个KL-Div作为惩罚项，进一步约束目标模型和参考模型差距不要过大

2. 去掉了为了估计advantage所需要的value net（复杂，昂贵，且容易被hack），用多个回答采样的reward归一化均值代替来计算advantage，而是回归了比较原始的monte carlo思想，advantage改由该样本回报相对多个样本平均回报的差值来估计，就得到了_GRPO_

### Implementation

WIP

