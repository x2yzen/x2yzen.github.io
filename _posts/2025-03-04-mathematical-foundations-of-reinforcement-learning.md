---
title: Mathematical Foundations of Reinforcement Learning
date: 2025-03-04 17:00:00 +0800
categories: [reinforcement_learning]
tags: [rl]
pin: false
math: true
---

## Overview

Notes from Mathematical Foundations of Reinforcement Learning ([Youtube series](https://www.youtube.com/watch?v=6M-hpwj6Kb8&list=PLEhdbSEZZbDYwsXT1NeBZbmPCbIIqlgLS&index=13) and [the book](https://github.com/MathFoundationRL/Book-Mathematical-Foundation-of-Reinforcement-Learning)).

One particularly nice aspect of this course is how it effectively connects many concepts in reinforcement learning. To summarize:

![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/wb1.png)

1. The core problem in reinforcement learning is finding the optimal policy (or dually, the optimal value function), i.e., solving the _Bellman optimal equation (BOE)_; everything below represents solution methods
2. Since BOE involves two mutually dependent variables (policy and value function) on both sides of the equation, we use iterative methods until convergence ← _value/policy iteration_
3. Environment models (reward and state transition functions) are typically unknown, so we introduce Monte Carlo sampling in value/policy iteration to estimate action values ← _MC learning_
4. MC depends on complete episode sampling, which is usually inefficient; we can actually update at each step (rather than waiting for the entire trajectory), while still guaranteeing overall convergence ← _TD-learning_, when applied to action value BOE it's called _Q-learning_

---Deep RL Boundary---

Replace discrete tables with parameterized functions (mainly DNNs) to represent values and policies, simplifying storage and gaining extrapolation capabilities—very useful when state spaces are large; at this point, optimality cannot be conveniently expressed via BOE (no longer explicitly operating on and evaluating each entry), requiring scalar metrics to evaluate and optimize functions

1. Replace (action) value functions with DNNs, using MSE between predicted and true values (estimates) as the scalar metric, yielding _deep Q-learning_
2. Replace policy functions with DNNs, using average state value/average reward as the scalar metric, yielding _policy gradient_; add variance reduction techniques to become _advantage policy gradient_, also known as _advantage actor-critic (A2C)_

## L2 Bellman Equation

### State value

<u>Def</u>: Expected discounted total return (denote as Gt) starting from the state (denote as s) and following a policy (denote as pi) from now on

![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/DJzUbumeSoP8CHxwsAAjVMrkpvf.png)

### Bellman equation

Based on MDP properties, Gt can be decomposed into immediate rewards and future rewards, describing the relationship between any state s's value and other state values, called the Bellman equation

![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/JmxJbv0tcogEZxxubmMjAZRmpef.png)

π and p represent the policy and environment model respectively, which are known; v is what we need to solve. Since this equation is valid for every state s, it essentially forms a solvable system of linear equations. Example:

![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/ZMxzblv8AoEK97xHvSqjNj6cpzg.png)
![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/ZBZPb5VS9ouyZnxs9Yojy2TxpZd.png)

This system has an analytical solution, but when there are too many states, iterative methods are often used to obtain approximate solutions. Simply substitute random initial v values into the right side of the equation repeatedly until convergence to obtain the state value function under the given π policy

![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/O473bX9vNoqSjfxeMBrj4z86pnh.png)
![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/VxnVbAFL1oTorDx6gk3jLdfCpBc.png)

### Action value

<u>Def.</u> Action value is defined on (s,a) tuples as the expected discounted total return starting from state s, taking action a, then following a policy thereafter.

With state values, we can calculate action values, which guide us on which action to take

![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/G3EAbdu0HoDUrXx6YOzjgoHap2c.png)

## L3 Optimal Policy and Bellman Optimality Equation

### Optimal Policy

<u>Def.</u> An optimal policy is one where every state's value (expected return) is no less than that state's value under any other policy (otherwise we could adjust the policy for that state)

![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/AhFTbAx8xo4fDmxQtdqjvZsbpbb.png)

Substituting into the Bellman equation gives the relationship that value functions under optimal policies should satisfy, called the Bellman optimality equation (BOE)

![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/EfqqbImIwo1M81x510PjEGThpJh.png)

This equation is quite tricky because π and v are mutually dependent, but it can be proven (via _contraction mapping theorem_) that iterative methods can still solve it (called _value iteration_). The specific method starts from a given initial value v; for state s, the policy π that maximizes value must choose the action a corresponding to the maximum q(s,a), thus computing new v. Repeat until convergence to obtain optimal value and optimal policy (deterministic greedy)

![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/MzrObhYhKooc6xxHYawjYyQQpze.png)
![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/YyTvb8ezZo0ryWxwc6bjp4Gmp8U.png)

## L4 Value and Policy Iteration

### value iteration

Value iteration starts from a given initial value function v0, computes corresponding q(s,a) and deterministic greedy policy π, then uses this π in the Bellman equation to iterate one step to get v1 ← Note that v1 is **not** a value function yet, since it satisfies v1=f(v0) rather than the standard Bellman equation form v1=f(v1). Only as iterations converge and the difference between vk+1 and vk becomes negligible do we obtain the optimal value function and corresponding policy

Pseudocode implementation

![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/ZNpObgN8Bosv1sx5rk7jtjVjp2f.png)

### policy iteration

Similarly, since v and π have a one-to-one correspondence, by symmetry we can also start from a given initial policy π0, compute the corresponding value function v_π0 (← this means **embedding a nested iterative convergence process** to get an accurate value function satisfying the Bellman equation), then use v_π0 to compute q_π0(s,a) to iterate π1 and repeat, eventually making the (π, v) pair converge to their respective optimal values. This method is called policy iteration. Compared to value iteration, policy iteration embeds an additional inner loop in each outer iteration step, requiring the value function to reach convergence

Pseudocode implementation:

![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/INFTbxppqociBAxqP3Cjzl0HpHf.png)

### Truncated policy iteration

Obviously, in policy iteration's value function solving process, we can avoid requiring full convergence and instead specify a fixed number of steps, achieving a beneficial tradeoff under diminishing marginal returns

![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/IYNbbSO9loRCo4x850KjRSoXp9B.png)
![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/Xl2RbpXRooiXWLxj0Bdjs5nSpsc.png)

Convergence relationships for the 3 methods:

<table>
<tr>
<td><br/></td><td>Value iteration<br/></td><td>Policy iteration<br/></td><td>Truncated policy iteration<br/></td></tr>
<tr>
<td>Inner loop iterations<br/></td><td>Fewest<br/></td><td>Most<br/></td><td>Medium<br/></td></tr>
<tr>
<td>Outer loop iterations<br/></td><td>Most<br/></td><td>Fewest<br/></td><td>Medium<br/></td></tr>
</table>

![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/EgR1bzNbxoNvDaxeKTKjsJDepGd.png)

## L5 Monte Carlo Learning

While value and policy iteration is model-based, MC methods are **model-free**

### Basic version

Core idea: The key to policy iteration is computing action value functions (because optimal policy is simply deterministic greedy on action value)

![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/WuvxbMU11oD7mzxpgzojPI1Ypjg.png)

In scenarios without a model (p), we can use data (a.k.a. experience) to compute q(s,a)

![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/IhWabaWsXogUNxxf8vPjZ39GpTc.png)

*The episode length should be **sufficiently long** but **does not have to be infinitely long.**

### Exploring starts

Generalized policy iteration, to use data more **efficiently**

- every-visit method

![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/V5Nhb50e8okW14x9WBSjGGtqpHf.png)

- Policy improvement timing: Improve the policy episode-by-episode, i.e. use the return of a single episode to approximate the action value.

![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/LwanbvYB9o4CKQxJnUojgw8CpKc.png)

<u>Requirement:</u> we need to generate sufficiently many episodes starting from **every** state-action pair ← is **difficult** **to achieve** in practice.

### Soft policies

<u>Def.</u> A policy is called soft if the probability of taking any action is positive. e.g. epsilon-greedy

![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/J5aRbWwbRoI6HRxeBOpj7vATpaf.png)

Simply insert into MC algorithm, we get:

![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/BXOkbAVtSo9rpuxADL1jAVYApih.png)

<u>Discussion</u>

Q: If greedy policies are replaced by ε-greedy policies in the policy improvement step, can we **still guarantee to obtain optimal policies**?

The answer is both yes and no. By yes, we mean that, when given sufficient samples, the algorithm can converge to an ε-greedy policy that is **optimal in the set Πε**. By no, we mean that the policy is merely optimal in Πε but **may not be optimal in Π.** However, if ε is sufficiently small, the optimal policies in Πε are **close** to those in Π.

Q: pros n' cons?

- The advantage of ε-greedy policies is that they have **stronger exploration** ability so that exploring starts condition is not required.
- The disadvantage is that ε-greedy policies are **not optimal** in general (we can only show that there always exist greedy policies that are optimal). With **bigger ε,** the optimal ε-greedy policy will even **lose consistency** with the greedy optimal policy (which means the action with the largest probability in converged soft policy may be different from the deterministic greedy policy, making it impossible to restore deterministic version based on soft alternatives.

## L6 Stochastic approximation and stochastic gradient descent

### Incremental estimation of sample mean

If samples are collected gradually, instead of waiting to collect all samples before computing the mean, we can continuously approximate during collection:

![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/U0inbQRw9oEGXPxZr2ljfudLpFh.png)

More generally, we can replace coefficient 1/k with α_k>0. Under a few broad assumptions, w will still converge to the expectation

![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/DsElb5vIzojb94xQ3I8jbeO8pDd.png)

### Robbins-Monro

<u>Def.</u> Stochastic approximation (SA):

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

As an application, we can formulate an **optimization problem** in which the objective function is J(w) as a root-finding problem: g(w) = ∇wJ(w) = 0. In this case, the condition that **g(w) is monotonically increasing** indicates that **J(w) is convex**, which is a commonly adopted assumption in optimization problems.

The inequality ∇wg(w) ≤ c2 indicates that **the gradient of g(w) is bounded** from above. For example, g(w) = tanh(w − 1) satisfies this condition, but g(w) = w³ − 5 does not.

(b): The second condition of {ak} requires that ak should **converge to zero** as k → ∞ but **not too fast**. One typical choice is **ak = 1/k**

(c): The third condition of noise is **mild**.

Summary: The RM algorithm essentially says **roots of monotonic functions can be obtained through simple iterative function observations (even noisy ones)** without needing to know expressions or other information.

<u>example</u>

![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/KTIPbkJajoBseMxEMsejEktTpmf.png)
![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/ZnOsbpErGoYDbKx01ZijeAJEpng.png)

Also, it could be derived that the _incremental estimation of sample mean_ is a special case of RM algorithm:

![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/RpG8bHOKCog64PxqdvVj7Ohspxe.png)

### Stochastic gradient descent

Simply speaking, SGD = batchGD@1

![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/GG8LbM3G6oka8SxpMm9jLIS1pnh.png)

![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/WKWtbQ46aoISkyxWxGbjlG5Apse.png)
![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/Jpvyb2izDoEKSfx3B7Wj86b8pUc.png)

<u>Discussion</u>

(a): **convergence guarantee**: it could be derived that _SGD_ is a special case of RM algorithm, thus its convergence is guaranteed under the same RM conditions

![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/BfeybrVIMopOtMx0jtnjnCiMpVc.png)

(b) **convergence behaviour:** The relative error between the stochastic and true gradients δk is **inversely proportional to |wk − w∗|**. As a result, when |wk − w∗| is large, δk is small. In this case, the SGD algorithm behaves like the gradient descent algorithm and hence wk quickly converges to w∗. When wk is close to w∗, the relative error δk may be large, and the convergence exhibits more randomness. See analysis:

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

Using TD to solve action value function:

![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/GRKNbyI78ofkxMxuoqljrm69pMc.png)

<u>Derivation</u>

Slightly more complex, mainly two parts: (a) obtain action value form of Bellman equation; (b) use RM method to estimate roots

![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/QP9zbqo4OorNbLx9LXvjLVOMpuc.jpg)

Using SARSA enables policy improvement, iterating toward optimal policy

![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/EAKrb23AGoeFb9xkXZqjv1mfp7g.png)

**expected-SARSA**

Essentially replaces q_t(s_t+1,a_t+1) in SARSA with E[q_t(s_t+1,A)] as action value estimate, clearly a more computation vs. less variance tradeoff

![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/MXK2b2fyKohkFTxrpdjjyiGnpOe.png)

**N-step SARSA**

Essentially expands action value in SARSA using actual rewards for more steps before closing with action value function. As steps n increase, bias decreases and variance increases, eventually becoming MC

![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/GCdlbKtuCo1GawxXm3ojm0s2pjh.png)
![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/KPzabx2lAolQ99xVYkEjQ7yrpdh.png)

### TD learning of optimal action values: Q-learning

- Sarsa can estimate the action values of a **given policy**. It must be combined with a policy improvement step to find optimal policies.
- Q-learning can directly estimate **optimal** action values and hence optimal policies.

![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/WKRDbaD7BoN17xxyht7jf8pGpnc.png)

According to Robbins-Monro, this iterative algorithm is actually solving

![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/MWb3bodw3oVRHMxSlpsj1Xucprb.png)

Following the above equation with deterministic greedy exactly yields the optimal Bellman equation, so the above is the _Bellman optimal equation in terms of action value_, proof below:

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

- Q-learning is off-policy (the target policy is always deterministic greedy, the behaviour function could be literally **anything**)

![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/Za7UbKZgyoFAZCxfrjxjBlrGp6e.png)

### Summary

![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/UZZSbTlKHol9xlxZmPGjf6FGpGd.png)
![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/L2TQbRI8yozzc0x94U1jhCrOpib.png)

## L8 Value function approximation

Use parametric functions, instead of tables, to describe value functions, in order to save storage and utilize generalization capability

To measure and optimize parameterized function performance, we need to define a scalar metric, such as MSE

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

Similarly, we can use TD estimation with function approximation to estimate action value

![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/XY5nbL0SxoCQQKxl3ZxjwflMpRf.png)

If we use DNN to describe q_hat, we get _deep Q-learning_. Additionally:

- To adapt to batch optimization and improve stability during training by avoiding chasing a moving target, the parameters wT of the first q_hat in the above equation are separated and fixed (called target network), only periodically synchronized with frequently updated parameters w (called main network)
  ![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/S0u4bN9M1ovDNkx75xyjya8Dpmh.png)
- To reduce the impact of inter-sample correlation on gradient estimation, experience replay is introduced. That is, after we have collected some experience samples, we do not use these samples in the order they were collected. Instead, we store them in a dataset called the replay buffer. In particular, let (s, a, r, s′) be an experience sample and B = {(s,a,r,s′)} be the replay buffer. Every time we update the main network, we can draw a mini-batch of experience samples from the replay buffer. The draw of samples, or called experience replay, should follow a **uniform distribution**.

Wrap-up:

![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/E8z9bGuJZo4PiIxizSdjHOjcpSg.png)

## L9 Policy Gradient

### Policy function and its metric

Change policy from tabular representation → functional representation

![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/NhanbeorUo1upyxKddujWWwWpTc.png)

We no longer handle each state-action entry discretely, so we also need another way to describe optimality (rather than defining it discretely on each state value). Ideally it should be a scalar function, making it convenient to optimize the policy function. Such scalar functions mainly fall into two categories: average state value and average (one-step) reward. Both are functions of policy π(θ) and can be proven equivalent (can be optimized simultaneously)

![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/CFizbt3Dnok00Px5wxAjat48pie.png)
![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/MbZkbfdGMoPz3uxgG1Cjdr1opge.png)

### Gradients of policy function

Gradients of this class of functions can all be represented by (9.8), while (9.9) converts (9.8) to expectation form, convenient for SGD solving (the log function converts action summation to probability-weighted summation, allowing it to be written into the outer expectation)

![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/UgeBbi3Mmof7Eqx3U0sjO3JtpSf.png)

![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/CKQZbUYhYoEtx8xNOwlj9SU7pDe.png)

### Optimize with policy gradients

Based on equation (9.9), SGD is written as follows

![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/NLBXb2DnooGf98xUV9YjsuiupUe.png)

Essentially strengthens the probability of actions with higher action values, amplified inversely by the current policy's probability for that action, implicitly expressing some explore-exploitation tradeoff

![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/OM9Rb50eToxIcxx47mUjTknppkd.png)

Applying (9.9) with SGD (removing expectation) + Monte Carlo (estimating action value) gives us a practical algorithm (REINFORCE)

![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/BnvRbdE0lokqIYxlAuqjuJF5pRd.png)

## L10 Actor-Critic

REINFORCE is a policy-based method, but also involves value estimation, where the latter determines the update magnitude of the former. Generally, these methods are called _actor-critic_.

### QAC

Obviously, there's more than one way to estimate values. If we change the action value estimation method in REINFORCE to TD-learning, simultaneously fitting a policy DNN and a Q function, we get Q-actor-critic (essentially a Frankenstein's monster combining policy gradient + SARSA + value function approximation)

![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/K6jPbs6WlosznyxPvIQjZz73pYd.png)

### Advantage-actor-critic (A2C)

Furthermore, equation (10.3) allows adding any function that only depends on state (and is independent of policy parameter θ) after the critic term in the expected return gradient formula without changing the expectation

![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/KnoObnYCnocwM1x8WTajtdlqp9e.png)

This enables us to use controlled variable (CUPED-like) methods to reduce SGD gradient estimation variance. A common choice for b(s) is the state value function, where the overall critic function is called advantage

![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/Pn4Nb7Z6zotskcxI948jWepCpX3.png)

In practice, to avoid maintaining 2 critic networks (q&v) simultaneously, the following approximation is commonly used

![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/MqeVbIo2YoNRw9xznuojqQ7XpWd.png)

This way, we can complete the algorithm using only one value network

![](/assets/images/2025-03-04-mathematical-foundations-of-reinforcement-learning/E7QybKuTKomVrVxijlnjqbe7p8d.png)