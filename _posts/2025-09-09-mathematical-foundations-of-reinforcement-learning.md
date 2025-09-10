---
title: Mathematical Foundations of Reinforcement Learning 
date: 2025-09-09 10:00:00 -0700
categories: [reinforcement_learning ]
tags: [rl]
pin: false
math: true
---

## Overview

This blog attempts to clarify the mathematical foundations of the reinforcement learning framework in a concise manner, following these principles:

1. **Minimization**: Introduce only the most essential concepts and algorithms, avoiding improved/hybrid versions
2. **Clarification**: Use equations rather than natural language to describe algorithms for precise definitions
3. **Structure**: Present their intrinsic connections and developmental patterns

The core problem in reinforcement learning is to find the optimal action policy in an environment that can be abstracted as a *Markov decision process*, in order to maximize rewards. There are two main approaches:

1. **Value-based**: Describes the value $q(s,a)$ of each action in each environment state. Indirectly, the optimal policy is to select the action with maximum value in the current environment. These methods belong to classical reinforcement learning theory, solved through dynamic programming to achieve Bellman Optimal Equation, with deep Q-learning being a typical algorithm.
2. **Policy-based**: Directly describes the optimal policy function, i.e., given any state s, output the optimal action $\pi(a|s)$. This approach is closer to solving optimization problems, optimizing expected rewards through standard gradient ascent.

Ultimately, these two methods can be combined through the **actor-critic** algorithmic framework, forming the complete theory of modern reinforcement learning.

## Value-based methods

### Bellman Equation

The Bellman Optimal Equation is a special form of the *Bellman Equation*. In the MDP model, the value of a state is defined as the reward obtained by following a policy $\pi$ from that state, consisting of *immediate rewards* and *future rewards*, where future rewards are actually the values of the next states. Therefore, we can use an equation to describe the relationship between the value of any state s and the values of other states:
$$
V_{\pi}(s) = \sum_{a} \pi(a|s) \left[ \sum_{r} P(r|s,a) \cdot r + \gamma \sum_{s'} P(s'|s,a) V_{\pi}(s') \right] \\
= \mathbb{E}[R + \gamma V_{\pi}(s')] \tag{1}
$$
This equation is called the Bellman Equation. From the action value perspective, the Bellman Equation has a dual form describing the relationship between any action's value and other action values:
$$
\begin{aligned}
q_{\pi}(s,a) &= \sum_{r} P(r|s,a) \cdot r + \gamma \sum_{s'} P(s'|s,a) V_{\pi}(s') \\
&= \sum_{r} P(r|s,a) \cdot r + \gamma \sum_{s', a'} P(s'|s,a) \pi(a'|s') q_{\pi}(s',a') \\
&= \mathbb{E}[R + \gamma q_{\pi}(s',a')]
\end{aligned} \tag{2}
$$
The transformation relationship between them is:
$$
V_{\pi}(s) = \sum_{a} \pi(a|s) q_{\pi}(s,a) \tag{3}
$$

### Optimality of Bellman Equation

From the Bellman Equation, we can see that state values are defined under a specific policy, so the same state has different values under different policies. Therefore, there must exist a policy under which each state's value is no less than that state's value under any other policy (otherwise the policy could be adjusted at that state). We define this as the optimal policy.

Substituting into the Bellman equation, the state value functions under the optimal policy should satisfy:
$$
v(s) = \max_{\pi} \left( r_{\pi} + \gamma P_{\pi} V \right)  \tag{4}
$$
This equation is called the Bellman optimality equation (state value version).
Obviously, it still has a dual form based on action values:
$$
q(s,a) = r_{t+1} + \gamma \max_{a} q(s_{t+1}, a) \tag{5}
$$
BOE is the core concept and essential problem of reinforcement learning. All following content revolves around these two points:

1. How to solve BOE: From the most basic value function iteration → monte-carlo learning, eliminating the requirement for environment model description; monte-carlo learning → temporal difference, achieving incremental updates and improving algorithmic efficiency; the final product is Q-learning.
2. What specific forms to use for v(s) and q(s,a): From discrete tabular form to parametric function form, gaining conciseness and generalization, combined with 1, finally obtaining deep Q-learning.

The following sections will separately introduce methods for solving BOE and forms of value functions.

### Solve BOE

#### Iteration Methods

The most basic BOE solution method, applicable when environment models $p(r|s,a)$ and $p(s'|s,a)$ are known.

##### policy evaluation

A series of iteration-based solution methods, with effectiveness guaranteed by the *Contraction Mapping Theorem*.
Bellman Equation (1) holds for every state s, so theoretically it can be solved simultaneously. However, when there are too many states, to avoid the computational complexity of solving a huge inverse matrix, iterative methods are often used to obtain approximate solutions: simply substitute random initial v values into the right side of the equation continuously, and after convergence, the state value under the given policy $\pi$ can be obtained. This process is also called _policy evaluation_.
$$
v_{k+1} = r_{\pi} + \gamma P_{\pi} v_{k} \tag{6}
$$
Bellman Optimal Equation (3) is more tricky: because $\pi$ and v are mutually dependent, but it can be proven that it can still be solved by initializing either side and then iterating.

##### value iteration

If we choose to initialize v, this method is called *value iteration*, with specific steps:

1. Starting from any given initial state value function v, for state s, calculate q(s,a)
   $$
   q_k(s,a) = \sum_{r}p(r|s,a)r + \gamma\sum_{s'}p(s'|s,a)v_k(s') \tag{7}
   $$
2. Obviously, by equation (3), the policy that maximizes state s value must greedily select the action a corresponding to the maximum q(s,a)
   $$
   \pi_{k+1}(a|s) =
   \begin{cases}
   1 & a = a_k^*(s) \\
   0 & a \neq a_k^*(s)
   \end{cases} \quad \text{where } a_k^*(s) = \arg\max_a q_k(s, a). \tag{8}
   $$
3. Calculate new v(s), noting that v at this time is not a value function, because the equation it satisfies is v1=f(v0) rather than the standard form v1=f(v1) of the Bellman equation
   $$
   v_{k+1}(s) = \max_{a} q_k(s, a) \tag{9}
   $$

Repeat until the difference between vk+1 and vk is negligible, then we obtain the optimal value for each state and the corresponding computable optimal policy (deterministic greedy).

##### policy iteration

Since v and $\pi$ have a one-to-one correspondence, by symmetry, we can also iterate starting from any initial policy. This method is called _policy iteration_:

1. Policy evaluation: Starting from any given initial policy $\pi_{0}$, for each state s, use equation (6) to calculate the corresponding value function $v_{\pi_{0}}$
2. Policy improvement: For each action a under each state s, use $v_{\pi_{0}}$ to calculate $q_{\pi_{0}}(s,a)$, then update $\pi_{0}$ to deterministic greedy

Repeat steps 1-2 iteratively until v and $\pi$ converge and correspond to each other. Compared to value iteration, policy iteration embeds an inner loop in each outer loop step, requiring the value function to reach convergence.

#### Monte-Carlo Learning

In real-world application scenarios, environment models (states, actions, reward mechanisms, etc.) are often difficult to express completely, making value/policy iteration ineffective (they all require environment models to calculate action values). However, calculating action values can also be estimated using empirical sampling, thus introducing a class of model-free methods collectively called monte-carlo learning.

##### Naive Monte-Carlo

The core of the policy iteration algorithm is actually calculating the action value function (because optimal policy is simply deterministic greedy on action value). Returning to the most essential definition, action value $q_{\pi}(s,a)$ is actually "the total discounted return obtained by executing action a in state s and then following policy $\pi$", so it can be estimated by sampling enough episodes and taking the average.

```
Initialization: Initial guess π₀.

Aim: Search for an optimal policy.

While the value estimate has not converged, for the kᵗʰ iteration, do
    For every state s ∈ S, do
        For every action a ∈ A(s), do
            Collect sufficiently many episodes starting from (s, a) following π₋k

    MC-based policy evaluation step:
        q_{π₋k}(s, a) = average return of all the episodes starting from (s, a)

    Policy improvement step:
        a₋k*(s) = argmaxₐ q_{π₋k}(s, a)
        π_{k+1}(a|s) = 1 if a = a₋k* and π_{k+1}(a|s) = 0 otherwise
```

##### Temporal-Difference

Monte-carlo learning requires collecting several complete episodes before each iteration, which is inefficient in practical implementation. Actually, an incremental method can be used for continuous iteration. First, introduce the required lemma:

###### Lemma: Robbins-Monro

```
Problem: Find root of g(w) = 0
Given: Noisy observations g̃(w, η) = g(w) + η

Algorithm (Robbins-Monro):
    w_{k+1} = w_k - a_k*g(w_k, η_k),   k = 1, 2, 3, ...

Convergence Conditions:
(a) 0 < c₁ ≤ ∇g(w) ≤ c₂ for all w        (monotonicity & bounded gradient)
(b) ∑a_k = ∞ and ∑a_k² < ∞              (step sizes decay appropriately)  
(c) E[η_k|H_k] = 0 and E[η_k²|H_k] < ∞   (zero-mean, finite variance noise)
    where H_k = {w_k, w_{k-1}, ...}
```

*Discussion:*

(a) In the first condition, 0 < c₁ ≤ ∇ₓg(w) indicates that g(w) is a **monotonically increasing** function. This condition ensures that the root of g(w) = 0 exists and is unique. The inequality ∇ₓg(w) ≤ c₂ indicates that the gradient of g(w) is bounded from above. For example, g(w) = tanh(w − 1) satisfies this condition, but g(w) = w³ − 5 does not.

(b) The second condition of {aₖ} requires that aₖ should converge to zero as k → ∞ but not too fast. One typical choice is **aₖ = 1/k.**

(c) The third condition of noise is mild.

**Summary:** The RM algorithm essentially states that the root of a monotonic function can be obtained through simple iterative function observations (even noisy ones), without knowing the expression, gradient, or other information. Combined with a schematic diagram, this is quite intuitive:
<div align="center">
<img src="/assets/images/2025-09-09-mathematical-foundations-of-reinforcement-learning/rm_intuition.png" alt="Robbins-Monro Algorithm Intuition" width="500">
</div>  

As a special form, Robbins-Monro can be used to incrementally estimate mathematical expectation $\mathbb{E}[X]$, by setting
$$
g(w) = w - \mathbb{E}[X] \tag{17}
$$
Because
$$
\tilde{g}(w, \eta) = w - x = w - x + \mathbb{E}[X] - \mathbb{E}[X] = (w - \mathbb{E}[X]) + (\mathbb{E}[X] - x) \stackrel{\triangle}{=} g(w) + \eta \tag{18}
$$
So the root of $g(w)$, which is $\mathbb{E}[X]$, can be estimated using
$$
w_{k+1} = w_k - \alpha_k \tilde{g}(w_k, \eta_k) = w_k - \alpha_k(w_k - x_k) \tag{10}
$$
Additionally, Robbins-Monro is also the mathematical foundation of *stochastic gradient descent*: As an application, we can formulate an optimization problem in which the objective function is J(w) as a root-finding problem: g(w) = ∇ₓJ(w) = 0. In this case, the condition that g(w) is monotonically increasing indicates that J(w) is **convex**, which is a commonly adopted assumption in optimization problems.

###### Q-learning

Using equation (10) to solve BOE:
$$
q(s, a) = \mathbb{E}\left[R_{t+1} + \gamma \max_a q(S_{t+1}, a) \mid S_t = s, A_t = a\right] \tag{19}
$$
We can obtain the following algorithm. Compared to monte-carlo learning which must complete an entire episode to update, this algorithm can update at each step through the difference between current estimates and (noisy) targets, hence called _temporal difference learning_. Specifically, since it solves for optimal action values, it's also called *Q-learning*.

```
Initialize q₀(s,a) for all s ∈ S, a ∈ A (randomly or to zero)
Initialize behavior policy πᵦ

Repeat (for each episode):
    Generate episode {s₀, a₀, r₁, s₁, a₁, r₂, ...} using πᵦ
    
    For each step t = 0, 1, 2, ... of the episode, do
        
        Update q-value:
        qₜ₊₁(sₜ, aₜ) = qₜ(sₜ, aₜ) - αₜ(sₜ, aₜ)[qₜ(sₜ, aₜ) - [rₜ₊₁ + γ max_a qₜ(sₜ₊₁, a)]]
        
        Update target policy:
        πₜ₊₁(a|sₜ) = 1 if a = arg max_a qₜ₊₁(sₜ, a)
        πₜ₊₁(a|sₜ) = 0 otherwise

Until convergence (or maximum number of episodes reached)
```  

### Parametric value functions

In previous sections, we assumed that state value function $v(s)$ and action value function $q(s,a)$ are in some **discrete tabular** form. In practical applications, the enumerated values of states and actions can be very large, so to save storage and gain generalization, a natural idea is to use **parametric functions**, instead of tables, to describe value functions.

If we use $\hat{q}(s,a,w)$ to represent action value function $q(s,a)$ and want it to be as accurate as possible, a common objective is:
$$
\min_w \mathbb{E}\left[q_t(s,a) - \hat{q}(s,a;w)\right]^2 \tag{20}
$$
According to gradient descent rules, we can easily obtain the optimization iteration:
$$
w_{t+1} = w_t + \alpha_t \left[q_t(s,a) - \hat{q}(s,a;w_t)\right] \nabla_w \hat{q}(s,a;w_t) \tag{21}
$$
This optimization is still intractable because we don't know the accurate optimization target $q(s,a)$. However, according to TD-Learning ideas, $rₜ₊₁ + γ max_a qₜ(sₜ₊₁, a, w_t)$ can serve as an estimate of optimal $q(s,a)$. Substituting into the above equation:

$$
w_{t+1} = w_t + \alpha_t \left[rₜ₊₁ + γ max_a qₜ(sₜ₊₁, a, w_t) - \hat{q}(s,a;w_t)\right] \nabla_w \hat{q}(s,a;w_t) \tag{11}
$$

By iteratively optimizing $\hat{q}(s,a,w)$ according to the above equation, we can finally learn a parametric estimate of the optimal action value function. Obviously, this method is a parametric version of Q-learning. Since deep neural networks are a common choice for this parametric function, it's also called **deep Q-learning**. Adding some practical tricks (experience replay & target network), the algorithm implementation is as follows:

```
Deep Q-learning (off-policy version)

Initialization: A main network and a target network with the same initial parameter.

Goal: Learn an optimal target network to approximate the optimal action values from 
the experience samples generated by a given behavior policy πb.

Store the experience samples generated by πb in a replay buffer B = {(s, a, r, s')}

For each iteration, do
    Uniformly draw a mini-batch of samples from B
    
    For each sample (s, a, r, s'), calculate the target value as 
    yT = r + γ max_{a∈A(s')} q̂(s', a, wT), where wT is the parameter of the target network
    
    Update the main network to minimize (yT - q̂(s, a, w))² using the mini-batch 
    of samples
    
    Set wT = w every C iterations
```

## Policy-based methods

Methods represented by policy gradient embody another approach to solving reinforcement learning problems: parametric functions can be used not only for value representation, but also for policy, from tabular representation $\pi(a_i|s_k)$ → functional representation $\pi(a|s,\theta)$. If we can use some scalar metric to describe optimality, we can conveniently obtain the optimal policy by optimizing the parametric policy function. _Average state value_ describes the average state value weighted by the Markov steady-state distribution corresponding to a policy, and is a frequently used metric:
$$
\bar{v}_{\pi}(s) = \sum_{s \in S} \eta(s) v_\pi(s) = \sum_{s \in S} \eta(s) \sum_{a \in A} \pi(a|s, \theta) q_\pi(s, a) \tag{12}
$$
To perform gradient ascent, we take the derivative of the above equation:
$$
\nabla_\theta J(\theta) = \sum_{s \in S} \eta(s) \sum_{a \in A} \nabla_\theta \pi(a|s, \theta) q_\pi(s, a) \tag{13}
$$
Computing this gradient requires traversing all states s and actions a, which is actually intractable. However, according to stochastic gradient descent ideas, it can be rewritten as the following sampling-based estimate:
Since
$$
\nabla_\theta \pi(a|s, \theta) = \nabla_\theta \ln \pi(a|s, \theta) \cdot \pi(a|s, \theta) \tag{22}
$$
We have
$$
\nabla_\theta J(\theta)  = \sum_{s \in S} \eta(s) \sum_{a \in A} \pi(a|s, \theta)\nabla_\theta \ln \pi(a|s, \theta) \cdot q_\pi(s, a) \tag{23}
$$
Which can also be written as
$$
\nabla_\theta J(\theta) = \mathbb{E}_{s \sim \eta, a \sim \pi(s, \theta)} \left[ \nabla_\theta \ln \pi(a|s, \theta) \cdot q_\pi(s, a) \right] \tag{14}
$$
That is, the gradient of the objective function is the mathematical expectation of an expression based on $(s,a)$, so we can use random sampling $(s_t,a_t)$ of $(s,a)$ to estimate and perform stochastic gradient descent:
$$
\theta_{t+1} = \theta_t + \alpha\cdot {q_t(s_t, a_t)} \nabla_\theta \ln\pi(a_t|s_t, \theta_t)=\theta_t + \alpha \left( \frac{q_t(s_t, a_t)}{\pi(a_t|s_t, \theta_t)} \right) \nabla_\theta \pi(a_t|s_t, \theta_t)  \tag{15}
$$
Analyzing the above equation, it essentially increases the probability of actions with higher action values, and is inversely amplified by the current policy's probability for that action, implicitly expressing some exploration-exploitation trade-off.
Combining (15) with SGD + Monte Carlo estimation of action values gives us a practically runnable algorithm (REINFORCE):

```
Pseudocode: Policy Gradient by Monte Carlo (REINFORCE)

Initialization: A parameterized function π(a|s, θ), γ ∈ (0, 1), and α > 0.

Aim: Search for an optimal policy maximizing J(θ).

For the kth iteration, do
    Select s₀ and generate an episode following π(θₖ). Suppose the 
    episode is {s₀, a₀, r₁, ..., sₜ₋₁, aₜ₋₁, rₜ}.
    
    For t = 0, 1, ..., T - 1, do
        Value update: qₜ(sₜ, aₜ) = Σ(k=t+1 to T) γ^(k-t-1) rₖ
        
        Policy update: θₜ₊₁ = θₜ + α ∇θ ln π(aₜ|sₜ, θₜ) qₜ(sₜ, aₜ)
    
    θₖ = θₜ
```

## Actor-critic framework

REINFORCE is a policy-based method, but it also involves value estimation, where the latter determines the magnitude of the former's updates. REINFORCE uses the most simple and direct monte-carlo learning to estimate action values, but obviously value estimation methods are not limited to one. We can "plug in" various schemes for estimating action values from value-based methods into REINFORCE to obtain various variants, collectively called _actor-critic_.

### Q-actor-critic
Replacing monte-carlo learning with TD-learning, while fitting both a policy-DNN and a Q-DNN, we get *Q-actor-critic*:
```
Pseudocode: The simplest actor-critic algorithm (QAC)

Initialization: A policy function π(a|s, θ₀) where θ₀ is the initial parameter. A value 
function q(s, a, w₀) where w₀ is the initial parameter. αw, αθ > 0.

Goal: Learn an optimal policy to maximize J(θ).

At time step t in each episode, do
    Generate aₜ following π(a|sₜ, θₜ), observe rₜ₊₁, sₜ₊₁, and then generate aₜ₊₁ following 
    π(a|sₜ₊₁, θₜ).
    
    Actor (policy update):
    θₜ₊₁ = θₜ + αθ ∇θ ln π(aₜ|sₜ, θₜ) q(sₜ, aₜ, wₜ)
    
    Critic (value update):
    wₜ₊₁ = wₜ + αw [rₜ₊₁ + γq(sₜ₊₁, aₜ₊₁, wₜ) - q(sₜ, aₜ, wₜ)] ∇w q(sₜ, aₜ, wₜ)
```
### Advantage-actor-critic  
Furthermore, it can be proven that equation (16) holds, which allows adding any function that only depends on state (and is independent of policy parameter $\theta$) to the critic term in the expected return gradient formula without changing the expectation:
$$
\mathbb{E}_{S \sim \eta, A \sim \pi} \left[ \nabla_\theta \ln \pi(A|S, \theta_t) q_\pi(S, A) \right] = \mathbb{E}_{S \sim \eta, A \sim \pi} \left[ \nabla_\theta \ln \pi(A|S, \theta_t) (q_\pi(S, A) - b(S)) \right] \tag{16}
$$
This enables us to use a method similar to controlled variables (CUPED) to reduce the variance of SGD gradient estimation. A common choice for b(s) is the state value function, in which case the overall critic function is called advantage:
$$
\theta_{t+1} = \theta_t + \alpha \mathbb{E}\left[ \nabla_\theta \ln \pi(A|S, \theta_t) [q_\pi(S, A) - v_\pi(S)] \right] \tag{24}
$$
In practice, to avoid maintaining 2 critic networks (q&v) simultaneously, the following approximation is commonly used:
$$
q_t(s_t, a_t) - v_t(s_t) \approx r_{t+1} + \gamma v_t(s_{t+1}) - v_t(s_t) \tag{25}
$$
This way, only one value network is needed to complete the algorithm:
```
Advantage actor-critic (A2C) or TD actor-critic

Initialization: A policy function π(a|s, θ₀) where θ₀ is the initial parameter. A value 
function v(s, w₀) where w₀ is the initial parameter. αw, αθ > 0.

Goal: Learn an optimal policy to maximize J(θ).

At time step t in each episode, do
    Generate aₜ following π(a|sₜ, θₜ) and then observe rₜ₊₁, sₜ₊₁.
    
    Advantage (TD error):
    δₜ = rₜ₊₁ + γv(sₜ₊₁, wₜ) - v(sₜ, wₜ)
    
    Actor (policy update):
    θₜ₊₁ = θₜ + αθ δₜ ∇θ ln π(aₜ|sₜ, θₜ)
    
    Critic (value update):
    wₜ₊₁ = wₜ + αw δₜ ∇w v(sₜ, wₜ)
```
