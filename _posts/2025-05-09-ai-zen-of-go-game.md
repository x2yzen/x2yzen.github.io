---
title: AI Zen of Go Game
date: 2025-05-09 22:42:00 +0800
categories: [reinforcement_learning]
tags: [rl,fun]
pin: false
math: false
---  

## Paper

### Mastering the game of Go with deep neural networks and tree search

[https://www.nature.com/articles/nature16961](https://www.nature.com/articles/nature16961)

[youtube: AlphaGo & Model-Based RL](https://www.youtube.com/watch?v=zHojAp5vkRE)

Let's start with the fundamental truth: _All games of perfect information have an optimal value function_. For Go, this means every board position has a definitive optimal strategy (greedy on optimal value function). Through policy iteration, starting from any given policy and repeatedly cycling through policy evaluation and policy improvement, we can theoretically reach the unique optimal solution (the God of Go). Following this logic, brute-force MCTS works in theory: for any given state (board position), starting from a purely random move policy, perform massive rollouts until victory or defeat, update the value function of each intermediate state until convergence, then update the policy to greedy, and repeat until policy convergence. Voilà—optimal strategy achieved.

The real problem lies in Go's spatial complexity:

- State space: Excluding illegal states (like filling your own eyes), 19×19 Go has approximately 2×10^170 legal board positions (hence the saying "no two games are ever the same")
- Trajectory space: From any state, the next move can have up to 360 possible actions, with up to ~400 moves remaining before the game ends (though the board only has 361 points, ko fights and other tactical battles can extend the total move count beyond the number of intersections). This makes the number of trajectories to search from each state astronomically large.

This renders brute-force MCTS computationally impossible (impossible yesterday, impossible today, impossible tomorrow).

The original AlphaGo followed the basic MCTS approach but combined deep learning networks to dramatically reduce spatial complexity from both state and trajectory dimensions. It became the first computer program to defeat professional human Go players (5-0 against Fan Hui, 4-1 against Lee Sedol).

- Reducing state space: We can't enumerate astronomical numbers of states, so we use parameterized methods to obtain representations. Since board positions are essentially image information, convolutional neural networks designed for image processing are a natural fit.
- Reducing trajectory space: For any given position, the vast majority of legal moves are unreasonable (moves that any competent human player wouldn't even consider). Therefore, the search width at each step can be significantly pruned.

![](/assets/images/2025-05-09-ai-zen-of-go-game/NTptb5PNVo9m6QxmNASjcjP4pDd.png)

AlphaGo trained three main functions:

- **Supervised learning network**: Sampled state-action pairs (s, a) from human games for behavior cloning. This step provides a cold start by filtering out completely unreasonable actions, reducing the spatial complexity for subsequent training. ← This network alone can beat amateur players
- **Policy gradient network**: Reinforcement learning training based on the SL network. The specific approach involves self-play against previous versions of itself until game completion, with reward function set to 0 for all non-terminal intermediate states, +1 for winning terminal states, and -1 for losing terminal states. By definition, all (s,a) pairs in winning trajectories have action value Q = +1, while losing trajectories have Q = -1, enabling REINFORCE training. ← This network wins 80% of games against the SL network
  ![](/assets/images/2025-05-09-ai-zen-of-go-game/PHJFb2Y9toOTOBxq51rjhUOWpsc.png)
- **Value network**: Predicts the outcome from position s of games played using policy p for both players. The model reuses most convolutional layers from the RL network, replacing the value head to output a scalar rather than probability distribution. Training data consists of 30 million state-outcome pairs (s, z) generated from RL network self-play.

The actual gameplay doesn't rely on any single network but uses them together to construct a Monte Carlo Tree Search (MCTS). The process works as follows:

1. Given an initial state S0
2. For each legal action, record 3 values:
   a. P(s,a) calculated using the SL network ← using the human behavior imitation model as prior
   b. N(s,a): total number of times this action was selected in current simulation ← initial value 0
   c. Q(s,a): value of this action ← initial value 0
3. Select an action according to the following formula to reach state Sl ← as rounds increase, the prior is gradually weakened
![](/assets/images/2025-05-09-ai-zen-of-go-game/UOfrb6RrRoCw42x41ItjbKKxpge.png)

4. Evaluate the value of state Sl, which is a weighted combination of two parts:
   a. Value network score
   b. Fast rollout result using SL network
![](/assets/images/2025-05-09-ai-zen-of-go-game/QOr5b54NKo2zYpxqDIujFmlhpRf.png)

5. Update N(s,a) += 1; Q(s,a) += V(SL) ← Since no non-terminal action has immediate reward, action value equals next state's state value
![](/assets/images/2025-05-09-ai-zen-of-go-game/AhSubNlO0oKfXTxqAD2jSzxYpCd.png)
6. Repeat steps 1-5 thousands of times, then select the most frequently chosen position for the actual move

**Discussion:**

- AlphaGo's core mechanism isn't deep reinforcement learning, but MCTS. Deep RL networks provide auxiliary functions for narrowing search space and shortening search depth.
- Many details reveal that this version of AlphaGo is essentially a patchwork solution full of heuristic designs: On one hand, during MCTS, neither action nor state values can be accurately computed by deep neural networks alone—they require random rollout corrections. On the other hand, relying on MCTS itself indicates insufficient RL training, because theoretically, reinforcement learning should account for all future state value trade-offs when computing state and action values, without needing explicit look-ahead search. Another telling detail: experiments found that MCTS priors work better when provided by the supervised learning network rather than the reinforcement learning network—further evidence of insufficient RL training at this stage.
- MCTS, this explicit forward search, is undoubtedly effective ← Deep RL networks alone can only beat amateur players, but with MCTS, they can even defeat Lee Sedol.
- Current language models' inference-time scaling is essentially the modern version of MCTS—both are forms of conditional computation during inference, using structured or unstructured additional computational processes to enhance model robustness and capability boundaries when facing complex tasks.

### Mastering the game of Go without human knowledge

[https://www.nature.com/articles/nature24270](https://www.nature.com/articles/nature24270)

## Code

demo implementation of alphago-zero in gomoku

[https://github.com/junxiaosong/AlphaZero_Gomoku](https://github.com/junxiaosong/AlphaZero_Gomoku)

## Fun fact

[🎯 After Witnessing AlphaGo's Singularity, I Became a "Human Traitor" - Fan Hui/Dongdong Gun/Beiming Chenghaisheng](https://www.xiaoyuzhoufm.com/episode/675ec9c27d8426f692408889)

[🎯 How AlphaZero's Ambidextrous Combat Technique Was Forged - Fan Hui/Dongdong Gun/Beiming Chenghaisheng](https://www.xiaoyuzhoufm.com/episode/680c61768aed253fa397587c)

[Ke Jie's Peak Masterpiece: Wuzhen Battle Against AlphaGo, Ten Great Dragons Dance Together, Thousands of Go Fans Boil with Excitement!](https://www.bilibili.com/video/BV1Za411P7bV/?share_source=copy_web&vd_source=988c48161c58791d278abb7b1437b14e)