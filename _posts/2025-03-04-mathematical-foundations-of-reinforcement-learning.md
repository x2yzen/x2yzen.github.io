---
title: Mathematical Foundations of Reinforcement Learning
date: 2025-03-04 17:00:00 +0800
categories: [reinforcement_learning]
tags: [rl]
pin: false
math: true
---

## Overview

Mathematical Foundations of Reinforcement Learning([Youtube series](https://www.youtube.com/watch?v=6M-hpwj6Kb8&list=PLEhdbSEZZbDYwsXT1NeBZbmPCbIIqlgLS&index=13) and[ the book](https://github.com/MathFoundationRL/Book-Mathematical-Foundation-of-Reinforcement-Learning))的笔记。  

这门课一个比较好的地方是把强化学习的很多概念比较有效地串起来了，总结来说：

![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/wb1.png)

1. 强化学习的核心问题是找到最优策略（或者对偶地，最优价值函数），也就是求解_Bellman optimal equation (BOE）_；下面都是求解方法
2. BOE 的等号两边涉及策略和价值函数两个相互 dependent 的变量，采用迭代直至收敛的方法来求解 <- _value/policy iteration_
3. 环境模型（奖励和状态转移函数）通常是未知的，在 value/policy iteration 中引入蒙特卡洛采样来估计 action value <- _MC learning_
4. MC 依赖完整的 episode 采样，通常是不高效的；其实可以在每个 step（而不是等待一整个链）都进行更新（incrementally），整体收敛性仍可得到保证 <- _TD-learning_，作用于 action value BOE 又称_Q-learning_

---Deep RL 分界线---

用参数化函数（主流选择 DNN）替代离散表格来表征价值和策略，可简化存储并获得外推性，在状态空间很大时很有用；此时 optimality 不便由 BOE 来表达（不再显式地直接操作和评估每一个 entry），而需要标量指标来评估并优化函数

1. 将（action）价值函数用 DNN 替代，标量指标是其与真实价值（估计）的 MSE，产生了_deep Q-learning_
2. 将策略函数用 DNN 替代，标量指标是 average state value/average reward，产生了_policy gradient__, 叠加上降方差的手段，_变成了_advantage policy gradient_，或者叫做_advantage actor-critic (A2C)_

## L2 Bellman Equation

### State value

<u>Def</u>: Expected discounted total return (denote as Gt) starting from the state (denote as s) and following a policy (denote as pi) from now on

![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/DJzUbumeSoP8CHxwsAAjVMrkpvf.png)

### Bellman equation

根据 MDP 的性质，Gt 可以拆解为 immediate rewards 和 future rewards，由此描述了任意状态 s 的价值与其它状态价值之间的关系，称为 Bellman equ

![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/JmxJbv0tcogEZxxubmMjAZRmpef.png)

pi 和 p 分别代表策略和环境模型，是已知的，v 是需要求解的，因为这个 equation 对每一个状态 s 都有效，所以本质上一定可以联立形成一个可解的线性方程组，举例：

![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/ZMxzblv8AoEK97xHvSqjNj6cpzg.png)
![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/ZBZPb5VS9ouyZnxs9Yojy2TxpZd.png)

这个方程组具备解析解，但状态过多时，经常使用迭代的方式来获取近似解，只需要把随机的 v 初始值不断带入方程右端，收敛后即可求得给定 pi 策略下的 state value function

![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/O473bX9vNoqSjfxeMBrj4z86pnh.png)
![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/VxnVbAFL1oTorDx6gk3jLdfCpBc.png)

### Action value

<u>Def.</u> Action value 定义在一对(s,a) tuple 上，是 expected discounted total return starting from the state s and taking actiona a, thereafter follow a policy.

有了 state value 以后，就可以计算 action value，action value 可以指导我们 which action to take

![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/G3EAbdu0HoDUrXx6YOzjgoHap2c.png)

## L3 Optimal Policy and Bellman Optimality Equation

### Optimal Policy

<u>Def.</u> 最优策略是指存在一个策略，其下每个状态的 value（预期 return）都不小于其它任何策略下该状态的 value（否则在该状态下就可以调整策略）

![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/AhFTbAx8xo4fDmxQtdqjvZsbpbb.png)

带入 Bellman equation，得到最优策略下不同 state 的 value function 应当满足的关系，称为 Bellman optimality equation （BOE)

![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/EfqqbImIwo1M81x510PjEGThpJh.png)

这个式子比较 tricky，因为 pi 和 v 互相 dependent，但可以证明（_contraction mapping theorem_）仍然可以通过迭代的方法来求解（称为_value iteration_）。具体方法是，从一个给定的初始值 v 出发，此时对于状态 s，最大化 value 的 policy pi 必定为选择最大的 q(s,a)对应的 action a，从而计算出新的 v，如此往复直至收敛，就得到了 optimal value 以及 optimal policy (deterministic greedy)

![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/MzrObhYhKooc6xxHYawjYyQQpze.png)
![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/YyTvb8ezZo0ryWxwc6bjp4Gmp8U.png)

## L4 Value and Policy Iteration

### value iteration

value iteration 从一个给定的初始 value function v0 出发，计算出对应的 q(s,a)和 deterministic greedy policy pi，然后再用这个 pi 带入 bellman equation 迭代一步得到 v1 <- 需要注意，此时的 v1 **并不是** value function，因为其满足的方程是 v1=f(v0)而不是 bellman equation 的标准形式 v1=f(v1)，而随着迭代逐渐收敛，vk+1 和 vk 的差异可以忽略不计时，我们才得到了 optimal value function 和对应的 policy

伪代码实现

![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/ZNpObgN8Bosv1sx5rk7jtjVjp2f.png)

### policy iteration

类似的，由于 v 和 pi 是一一对应的，根据对称性，也可以从一个给定的初始策略 pi0 出发，计算出对应的 value function v_pi0（<-这意味着这一步需要**嵌套一个迭代收敛的过程**，得到满足 bellman equation 的，准确的 value function），然后用 v_pi0 计算 q_pi0(s,a)来迭代 pi1 并重复迭代，最终也可以使(pi, v) pair 达到各自收敛并对应。这个方法被称为 polity iteration，相比于 value iteration，policy iteration 在每一步外层循环中都内嵌了一个额外的循环，要求 value function 达到收敛

伪代码实现：

![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/INFTbxppqociBAxqP3Cjzl0HpHf.png)

### Truncated policy iteration

显然可以在 policy iteration 的 value function 求解过程中，不再要求达到收敛，而是规定一定的步数，来获得边际效益递减条件下有益的 tradeoff

![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/IYNbbSO9loRCo4x850KjRSoXp9B.png)
![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/Xl2RbpXRooiXWLxj0Bdjs5nSpsc.png)

3 种方法的收敛关系：

<table>
<tr>
<td><br/></td><td>Value iteration<br/></td><td>Policy iteration<br/></td><td>Truncated policy iteration<br/></td></tr>
<tr>
<td>内层循环迭代次数<br/></td><td>最少<br/></td><td>最多<br/></td><td>中等<br/></td></tr>
<tr>
<td>外层循环迭代次数<br/></td><td>最多<br/></td><td>最少<br/></td><td>中等<br/></td></tr>
</table>

![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/EgR1bzNbxoNvDaxeKTKjsJDepGd.png)

## L5 Monte Carlo Learning

While value and policy iteration is model-based, MC methods are **model-free**

### Basic version

核心思想：policy iteration 的关键是计算 action value function（因为 optimal policy is simply deterministic greedy on action value）

![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/WuvxbMU11oD7mzxpgzojPI1Ypjg.png)

而在没有 model（p）的场景中可以使用 data（a.k.a. experience）来完成 q(s,a)的计算

![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/IhWabaWsXogUNxxf8vPjZ39GpTc.png)

*The episode length should be **sufficiently long** but **does not have to be infinitely long. **

### Exploring starts

Generalized policy iteration, to use data more **efficiently**

- every-visit method

![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/V5Nhb50e8okW14x9WBSjGGtqpHf.png)

- Policy improvement timing: Improve the policy episode-by-episode, i.e. use the return of a single episode to approximate the action value.

![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/LwanbvYB9o4CKQxJnUojgw8CpKc.png)

<u>Requirement:</u> we need to generate sufficiently many episodes  starting from **every** state-action pair <- is **difficult** **to achieve** in practice.

### Soft policies

<u>Def.</u> A policy is called soft if the probability of taking any action is positive. e.g. epsilon-greedy

![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/J5aRbWwbRoI6HRxeBOpj7vATpaf.png)

Simply insert into MC algorithm, we get:

![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/BXOkbAVtSo9rpuxADL1jAVYApih.png)

<u>Discussion </u>

Q: If greedy policies are replaced by ε-greedy policies in the policy improvement step, can we **still guarantee to obtain optimal policies**?

The answer is both yes and no. By yes, we mean that, when given sufficient samples, the algorithm can** **converge to an ε-greedy policy that is **optimal in the set Πε**. By no, we mean that the policy is merely optimal in Πε but **may not be optimal in Π.** However, if ε is sufficiently small, the optimal policies in Πε are** close** to those in Π.

Q: pros n' cons?

- The advantage of ε-greedy policies is that they have **stronger  exploration** ability so that exploring starts condition is not required.
- The disadvantage is that ε-greedy policies are **not optimal** in general (we can only show that there always exist greedy policies that are optimal). With **bigger ε, **the optimal ε-greedy policy will even** lose consistency** with the greedy optimal policy (which means the action with the largest probability in converged soft policy may be different from the deterministic greedy policy, making it impossible to restore deterministic version based on soft alternatives.

## L6 Stochastic approximation and stochastic gradient descent

### Incremental estimation of sample mean

如果样本是逐渐收集的，不用等到收集完成所有的样本再一次计算均值，而是可以在收集的过程中不断逼近：

![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/U0inbQRw9oEGXPxZr2ljfudLpFh.png)

更一般地，可以将系数 1/k 替换为 alpha_k>0，可以证明在满足少数宽泛假设的前提下，w 仍将收敛到期望

![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/DsElb5vIzojb94xQ3I8jbeO8pDd.png)

### Robbins-Monro

<u>Def. </u>Stochastic approximation (SA):

- SA refers to a broad class of **stochastic iterative** algorithms solving **root finding** or **optimization** problems.
- Compared to many other root-finding algorithms such as gradient-based methods, SA is powerful in the sense that it does **not require** knowing the expression of the **objective function** **nor its derivative.**

<u>Def.</u> Robbins-Monro:

Suppose that we would like to find the root of the equation

![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/QlPLbbrGWoD0i2xfGfSjy0TZpyb.png)

And possibly, we can only obtain a noisy observation of g(w):

![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/RcaAb9T9yo4EbBxNQT2j1wpTpke.png)

![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/GfwAbudNxoUWDmxTWT4jVu35p3f.png)

The RM algorithm that can solve g(w) = 0 is

![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/OFqgbOgdTotrLExsi6bjKrlvpy2.png)

If the following conditions are met

![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/IgJ4bGbJYoRukUxBc4kjaqWspyb.png)

<u>Discussion</u>

(a): In the first condition, 0 < c1 ≤ ∇wg(w) indicates that g(w) is a **monotonically increasing** function. This condition ensures that the root of g(w) = 0 exists and is unique. If g(w) is monotonically decreasing, we can simply treat −g(w) as a new function that is monotonically increasing.

As an application, we can formulate an **optimization problem** in which the objective function is J(w) as a root-finding problem: g(w) =. ∇wJ(w) = 0. In this case, the condition that **g(w) is monotonically increasing** indicates that **J(w) is convex**, which is a commonly adopted assumption in optimization problems.

The inequality ∇wg(w) ≤ c2 indicates that **the gradient of g(w) is bounded** from above. For example, g(w) = tanh(w − 1) satisfies this condition, but g(w) = w**3 − 5 does not.

(b): The second condition of {ak} requires that ak should **converge to zero** as k → ∞ but  **not too fast**. One typical choice is** ak = 1/k**

(c): The third condition of noise is **mild**.

总结：RM 算法本质上是说**单调函数的根可以通过简单的函数观测值（即使是有噪音的）迭代得到**，而不需要知道表达式等信息。

<u>example</u>

![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/KTIPbkJajoBseMxEMsejEktTpmf.png)
![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/ZnOsbpErGoYDbKx01ZijeAJEpng.png)

Also, it could be derived that the_ incremental estimation of sample mean_ is a special case of RM algorithm:

![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/RpG8bHOKCog64PxqdvVj7Ohspxe.png)

### Stochastic gradient descent

simply speaking, SGD = batchGD@1mailto:batchGD@1

![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/GG8LbM3G6oka8SxpMm9jLIS1pnh.png)

![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/WKWtbQ46aoISkyxWxGbjlG5Apse.png)
![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/Jpvyb2izDoEKSfx3B7Wj86b8pUc.png)

<u>Discussion</u>

(a): **convergence guarantee**: it could be derived that the_ SGD_ is a special case of RM algorithm, thus its convergence is guaranteed under the same RM conditions

![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/BfeybrVIMopOtMx0jtnjnCiMpVc.png)

(b) **convergence behaviour: **The relative error between the stochastic and true gradients δk is **inversely proportional to |wk − w∗|**. As a result, when |wk − w∗| is large, δk is small. In this case, the SGD algorithm behaves like the gradient descent algorithm and hence wk quickly converges to w∗. When wk is close to w∗, the relative error δk may be large, and the convergence exhibits more randomness. See analysis:

![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/JdxibDm1YoIYoPxPhM2jGrJapmc.png)

(c) BGD, MBGD and SGD

![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/PGNSbx7PGobLIyxwMw4jO7LKpRd.png)

![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/X6txbUlkFocXmdx6fEljM5JJpeh.png)

## L7 Temporal-Difference Learning

### TD learning of state value

First, we show besides mean estimation, the RM algorithm also applies to the following format:

![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/CZ91bF11eo3TfNxQVORjw0JXpye.png)

This is basically the form of the Bellman equation

![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/Xy37bons3o4TEJxwr6Cji1Dppwf.png)

![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/FE6wbceNtozkzFxzcPQjNfTgpLf.png)

So we get the TD estimation of state value function as

![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/N94Bb8OE8oQRkNxLkxcjkIEFpuC.png)

TD vs. MC

![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/ZCIwbKgLtoHXG2xQFUBjP3Hvpnc.png)

### TD learning of action value

**SARSA**

利用 TD 求解 action value function：

![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/GRKNbyI78ofkxMxuoqljrm69pMc.png)

<u>Derivation</u>

稍有一点复杂性，主要分为两部分：(a)：获得 Bellman equation 的 action value 形式；(b)：使用 RM 方法估计根

![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/QP9zbqo4OorNbLx9LXvjLVOMpuc.jpg)

利用 SARSA 可以进行 policy improvement，从而迭代到 optimal policy

![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/EAKrb23AGoeFb9xkXZqjv1mfp7g.png)

**expected-SARSA**

本质上是在 SARSA 中，用 E[q_t(s_t+1,A)]代替 q_t(s_t+1,a_t+1)作为 action value 的估计，显然是 more computation vs. less variance 的 tradeoff

![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/MXK2b2fyKohkFTxrpdjjyiGnpOe.png)

**N-step SARSA**

本质上是在 SARSA 中，对 action value 用实际的 reward 展开更多步后，再用 action value function 收起，随着步数 n 增大，性能上 bias 下降，variance 上升，最终变为 MC

![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/GCdlbKtuCo1GawxXm3ojm0s2pjh.png)
![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/KPzabx2lAolQ99xVYkEjQ7yrpdh.png)

### TD learning of optimal action values: Q-learning

- Sarsa can estimate the action values of a **given policy**. It must be combined with a policy improvement step to find optimal policies.
- Q-learning can directly estimate **optimal** action values and hence optimal policies.

![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/WKRDbaD7BoN17xxyht7jf8pGpnc.png)

根据 Robbins-Monro，这个迭代算法实际上是在求

![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/MWb3bodw3oVRHMxSlpsj1Xucprb.png)

按照上式进行 deteminstic greedy，恰好能得到 optimal bellman equation，因此上式是_Bellman optimal equation in terms of action value_，证明如下：

![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/AaAWb0iCFoZ1mOxFTbEjWQripBg.png)

![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/DmTib1l2noasbHxeJUvjlttopZd.png)

The implementation of Q-learning diverges into two versions (on-policy and off-policy respectively)

![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/Oj7yb1SLOoCwYGx4wtjjqnU3pie.png)
![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/OCCkbEHAdoKDfExEt9gjv2rSplh.png)

### On policy vs. Off policy

<u>Def.</u>

Two policies exist in any reinforcement learning task:

- The **behavior policy** is the one used to generate experience samples.
- The **target policy** is the one that is constantly updated to converge to an optimal policy.

When the behavior policy is the same as the target policy, such a learning process is called **on-policy**. Otherwise, when they are different, the learning process is called **off-policy**.

<u>Examples</u>

- Sarsa is on-policy (since a_t+1 is generated by the behavior policy, and policy improvement is performed on it as well)

![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/XnZZbjn7bontTbxADHxjxKhepkc.png)

- Q-learning is off-policy (the target policy is always deterministic greedy,  the behaviour function could be literally **anything**)

![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/Za7UbKZgyoFAZCxfrjxjBlrGp6e.png)

### Summary

![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/UZZSbTlKHol9xlxZmPGjf6FGpGd.png)
![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/L2TQbRI8yozzc0x94U1jhCrOpib.png)

## L8 Value function approximation

Use parametric functions, instead of tables, to describe value functions, in order to save storage and utilize generalization capability

为了衡量并优化参数化函数的表现，需要定义一个标量指标，比如 MSE

### State value approximation

![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/JqvKbevlooHzPhxNMuRjbOrspwe.png)

It's basically the average of each state value error, weighted by _stationary distribution of Markov process_

![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/JVz5bl2jGoLJn9xfZ8PjTSnOpbg.png)

Do gradient descent

![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/HSc5bJMjBoXWDTxA2dcjrXxLpod.png)

![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/SuKGbC4rCoxPTuxQjaxjzuuNpLc.png)

Notably, (8.12) is **not implementable** because it requires the true state value vπ, which is unknown and must be estimated.

![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/TrBubotCcoFbZHxfSuBjl5oyprc.png)

*_Strictly speaking, the TD learning with function approximation does NOT exactly optimize the original objective function J(w), due to the introduced TD expression is also a function of w-> will cover it in the following content_

![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/FF46brSxyoNmCvxZIUKjHzhzpRg.png)

### Action value approximation

类似地，可以使用 TD estimation with function approximation 估计 action value

![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/XY5nbL0SxoCQQKxl3ZxjwflMpRf.png)

如果使用 DNN 来描述 q_hat，就得到了_deep Q-learning_，额外地

- 为了适应 batch-optimization，并且在训练中提高稳定性，避免追逐一个 moving target，上式第 1 个 q_hat 的参数 wT 被分离并固定（称为 target network），只是周期性地与频繁更新的参数 w（称为 main network）同步
  ![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/S0u4bN9M1ovDNkx75xyjya8Dpmh.png)
- 为了减少样本间相关性对梯度估计的影响，引入了 experience replay. That is, after we have collected some experience samples, we do not use these samples in the order they were collected. Instead, we store them in a dataset called the replay buffer. In particular, let (s, a, r, s′) be an experience sample and B =. {(s,a,r,s′)} be the replay buffer. Every time we update the main network, we can draw a mini-batch of experience samples from the replay buffer. The draw of samples, or called experience replay, should follow a **uniform distribution**.

Wrap-up:

![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/E8z9bGuJZo4PiIxizSdjHOjcpSg.png)

## L9 Policy Gradient

### Policy function and its metric

将 policy 从 tabular representation->functional representation

![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/NhanbeorUo1upyxKddujWWwWpTc.png)

我们不再用离散化的方式直接处理每一个 state-action entry，同理也需要换一种方法来描述 optimality（而不是离散化地定义在每一个 state value 上），理想情况下它应该是一类标量函数，这样方便对 policy function 进行 optimize。这样的标量函数主要有两类：average state value 和 average (one-step) reward，它们都是策略 pi (theta)的函数，并可以证明是等价的（可以被同时优化）

![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/CFizbt3Dnok00Px5wxAjat48pie.png)
![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/MbZkbfdGMoPz3uxgG1Cjdr1opge.png)

### Gradients of policy function

这一类函数的梯度都可以用(9.8)表示，而(9.9)则将(9.8)改为了期望形式，方便使用 SGD 求解（引入 log 函数的目的是将 action 求和改为按概率求和，从而写进外层的求期望中）

![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/UgeBbi3Mmof7Eqx3U0sjO3JtpSf.png)

![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/CKQZbUYhYoEtx8xNOwlj9SU7pDe.png)

### Optimize with policy gradients

基于式(9.9)SGD 写成如下形式

![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/NLBXb2DnooGf98xUV9YjsuiupUe.png)

本质上是加强 action value 更大的 action 出现的概率，并且会被当前策略对该 action 的概率反向放大，隐式地表达了某种 explore-exploration trade-off

![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/OM9Rb50eToxIcxx47mUjTknppkd.png)

把(9.9)套上 SGD（去掉 expectation）+ Monte Carlo（估计 action value），就得到了实际可以运行的算法（REINFORCE）

![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/BnvRbdE0lokqIYxlAuqjuJF5pRd.png)

## L10 Actor-Critic

REINFORCE 方法是一种 policy based method，但也涉及 value estimation，后者决定了前者更新的幅度；generally，将这类方法称作_actor-crtic_。

### QAC

显然，value estimation 的方法不止一种，如果把 REINFORCE 中估计 action value 的方式改为 TD-learning，同时 fit 一个 policy DNN 和一个 Q function，就得到了 Q-actor-critic（本质上是 policy gradient + SARSA + value function approximation 的缝合怪）

![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/K6jPbs6WlosznyxPvIQjZz73pYd.png)

### Advantage-actor-critic (A2C)

进一步地，式(10.3)允许在期望收益梯度的 formula 中的 critic 项后加上任意只和 state 有关（和策略参数 $\theta
$ 无关）的函数，而不改变期望

![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/KnoObnYCnocwM1x8WTajtdlqp9e.png)

这使得我们可以使用类似 controled variable (CUPED) 的方法来降低 SGD 估计梯度的方差。对于 b(s)一个常见的选择是 state value function，此时的 critic function 整体被称作 advantage

![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/Pn4Nb7Z6zotskcxI948jWepCpX3.png)

实际在使用中，为了避免同时维护 2 个 critic network (q&v)，常使用如下近似

![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/MqeVbIo2YoNRw9xznuojqQ7XpWd.png)

这样一来，就可以只使用一个 value net 完成算法

![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/E7QybKuTKomVrVxijlnjqbe7p8d.png)