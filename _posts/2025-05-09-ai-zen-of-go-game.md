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

开宗明义，_All games of perfect information have an optimal value function_，即对于围棋，每一个局面都有确定的最优策略（greedy on optimal value function），可以通过 policy iteration，从任意给定策略出发，经过反复的 policy evaluation 和 policy improvement，达到唯一最优解（围棋之神）。遵循这个思路，暴力 MCTS 从理论上是奏效的：对于一个给定的状态（棋局），从纯随机的落子策略出发，进行大量 rollout 直至分出胜负，更新其间每个状态的 value function 直至收敛，随后更新策略到 greedy，重复循环直至策略收敛，就得到了这个状态下的最优策略。

实际的问题在于围棋的空间复杂度：

- 状态数：去除不合法状态（比如无气填子），19*19 围棋共有~2e170 种合法局面（所谓“千古无同局”）
- 轨迹数：在任意状态下，下一步最高可达 360 种可能的 action，而距离分出胜负的终态最高还剩~400 步（虽然棋盘只有 361 个点位，但考虑到劫争等反复争夺的情况，总手数上限会高于点位总数），这使得每一个状态出发需要搜索的轨迹数也成为一个天文数字

这使得上述的暴力 MCTS 从计算量上成为了不可能（昨天不可能，今天不可能，明天也不可能）。

初版 alphago 遵循了 MCTS 的基本思路，但结合深度学习网络从状态和轨迹两个方向上极大地缩减了空间复杂度，第一次在围棋上使用计算机程序战胜了人类职业棋手（先后 5:0 战胜樊麾，4:1 战胜李世石）

- 削减状态数：我们不可能枚举天文数字的状态，因此使用参数化的方法获得其的某种表征；显然，棋局是一种特定的图像信息，因此用来处理图片的卷积神经网络是一个合适的选择
- 削减轨迹数：对于一个给定的局面，所有合法落子的位置中，绝大多数都是不合理的（会下棋的人类根本不会考虑的走法），因此每一步搜索的宽度是可以被缩减的；

![](/assets/images/2025-05-09-ai-zen-of-go-game/NTptb5PNVo9m6QxmNASjcjP4pDd.png)

AlphaGo 一共训练了 3 个主要函数：

- supervised learning network：从人类对局中采样 state-action pairs (s, a) 做 behavior cloning。这一步的目的是冷启动，过滤掉大量完全不合理的 action，减少后续训练的空间复杂度 <-这一步的产物可以击败业余棋手
- policy gradient network：在 SL network 的基础上进行强化学习训练，具体做法是让该 network 和之前某个版本的自身对弈直至棋局结束，奖励函数设定为对每一个非结局的中间状态 reward 都为 0，结局状态赢棋为 +1，输棋为-1。那么根据定义，对赢棋轨迹中的所有(s,a) pair action value Q 均为 +1，输棋轨迹则均为-1，由此可以使用 REINFORCE 进行训练 <- 这一步的产物可以 80% 击败 SL network
  ![](/assets/images/2025-05-09-ai-zen-of-go-game/PHJFb2Y9toOTOBxq51rjhUOWpsc.png)
- value network:  目的是 predicts the outcome from position s of games played by using policy p for both players，模型复用 RL network 的大部分卷积层，替换 value head 使其输出一个 scalar 而不是 probability distribution，训练数据是 30 million RL network 自我对弈产生的 state-outcome pairs (s, z).

实际下棋的过程并不单独依靠任何一个 network，而是一起使用它们构建一个 monte carlo tree search (MCTS)，具体做法是

1. 给定一个初始状态 S0  
2. 对于每个合法的 action，记录 3 个数值：

   a. 使用 SL network 计算的 P(s,a) <- 以人类行为模仿模型当做 prior  
   b. 本次模拟中该 action 被选定的总次数 N(s,a) <- 初始值为 0  
   c. 该 action 的价值 Q(s,a) <- 初始值为 0  
3. 按照下式选择一个 action，来到状态 Sl <- 随着轮次的增多，prior 被逐渐削弱  
![](/assets/images/2025-05-09-ai-zen-of-go-game/UOfrb6RrRoCw42x41ItjbKKxpge.png)

4. 评估状态 Sl 的价值，也由两部分加权组成:  
   a. 价值网络的打分  
   b. 用 SL network 快速 rollout 的结果  
![](/assets/images/2025-05-09-ai-zen-of-go-game/QOr5b54NKo2zYpxqDIujFmlhpRf.png)

5. 更新 N(s,a) +=1; Q(s,a)+=V(SL) <- 由于任何非终局 action 都没有 immadiate reward，所以行为价值就等于下一个状态的状态价值
![](/assets/images/2025-05-09-ai-zen-of-go-game/AhSubNlO0oKfXTxqAD2jSzxYpCd.png)
6. 重复 1-5 数千次，最后选择被最多选到的位置实际落子

讨论：

- alphago 的核心机制不是深度强化学习，而是 MCTS；深度强化学习网络提供了收窄搜索空间和收短搜索深度的辅助作用
- 从很多细节可以看出这个版本的 alphago 是一个补丁攒起来的方案，充满了各种启发式的设计：一方面，在进行 MCTS 时，行为和状态的价值都不能被深度神经网络准确地计算，而是要用随机 rollout 进行修正；另一方面，依赖 MCTS 本身也说明了强化学习训练不充分，因为从理论上说，强化学习对于状态和行为价值的计算就应当考虑到所有未来状态价值的 trade-in，而不必依赖一个显式的 look-foraward search；另外有一个细节是实验发现 MCTS 的 prior 由监督学习网络而不是再次强化学习的网络给出效果竟然更好，也是此阶段强化学习训练并不充分的一个辅证
- MCTS 这种显式的前向搜索无疑是奏效的 <-深度强化学习出的网络本身只能打败业余选手，但加上 MCTS 之后甚至可以击败李世石
- 当前语言模型的 inference-time scale 其实就是当前版本的 MCTS，都是推理时条件计算的一种形式，用结构化或非结构化的额外计算过程来增强模型在面对复杂任务时的鲁棒性与能力边界

### Mastering the game of Go without human knowledge

[https://www.nature.com/articles/nature24270](https://www.nature.com/articles/nature24270)

## Code

demo implementation of alphago-zero in gomoku

[https://github.com/junxiaosong/AlphaZero_Gomoku](https://github.com/junxiaosong/AlphaZero_Gomoku)

## Fun fact

[🎯 亲历 AlphaGo 奇点后，我成了"人奸" - 樊麾/东东枪/北冥乘海生](https://www.xiaoyuzhoufm.com/episode/675ec9c27d8426f692408889)

[🎯 左右互搏的 AlphaZero 大法，是怎样炼成的？ 樊麾/东东枪/北冥乘海生](https://www.xiaoyuzhoufm.com/episode/680c61768aed253fa397587c)

[柯洁最巅峰名局：乌镇大战阿法狗，十条大龙共舞，万千棋迷沸腾！](https://www.bilibili.com/video/BV1Za411P7bV/?share_source=copy_web&vd_source=988c48161c58791d278abb7b1437b14e)
