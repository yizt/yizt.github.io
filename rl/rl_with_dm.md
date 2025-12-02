---
layout: default
title: 基于扩散模型的强化学习
nav_order: 3
---

# 基于扩散模型的强化学习



## QSM
 [Learning a Diffusion Model Policy from Rewards via Q-Score Matching](https://arxiv.org/pdf/2312.11752)


## DDPO
 [TRAINING DIFFUSION MODELS WITH REINFORCEMENT LEARNING](https://arxiv.org/pdf/2305.13301)直接优化扩散模型（diffusion models）以满足特定的下游目标（downstream objectives），而不是仅仅匹配数据分布。  
 大多数扩散模型的使用场景并不直接关注似然度（likelihoods），而是关注于人类感知的图像质量、药物有效性等下游目标。本文提出了一种基于强化学习（reinforcement learning, RL）的方法，称为去噪扩散策略优化（denoising diffusion policy optimization, DDPO），来直接针对这些下游目标训练扩散模型。 
### 核心思想
 去噪过程（denoising process）作为一个多步马尔可夫决策问题（multi-step markov decision-making problem），以便使用策略梯度算法（policy gradient algorithms）来优化扩散模型。

**多步MDP定义**
$$
\begin{equation*}
\begin{gathered}
\mathbf{s}_t \triangleq (\mathbf{c}, t, \mathbf{x}_t) \ \ \ // 状态 \\
\mathbf{a}_t \triangleq \mathbf{x}_{t-1}\ \ \ //动作为单步去噪结果\\
\pi(\mathbf{a}_t | \mathbf{s}_t) \triangleq p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t, \mathbf{c})  \ \ \ //策略 \\
P(\mathbf{s}_{t+1} | \mathbf{s}_t, \mathbf{a}_t) \triangleq (\delta_{\mathbf{c}}, \delta_{t-1}, \delta_{\mathbf{x}_{t-1}}) \ \ \ //确定性状态转移\\
p_0(\mathbf{s}_0) \triangleq (p(\mathbf{c}), \delta_T, \mathcal{N}(\mathbf{0}, \mathbf{I})) \ \ \ //初始状态分布 \\
R(\mathbf{s}_t, \mathbf{a}_t) \triangleq 
\begin{cases} 
r(\mathbf{x}_0, \mathbf{c}) & \text{if } t = 0 \ \ \ //只有最后一步有奖励\\ 
0 & \text{otherwise}
\end{cases}
\end{gathered}
\end{equation*}
$$

**策略梯度估计**
  可直接使用Monte Carlo估计  
  DDPO（去噪扩散策略优化）的第一个变体，我们称之为 DDPOsf，使用得分函数策略梯度估计器(REINFORCE)
$$
\begin{equation*}
\nabla_{\theta} \mathcal{J}_{\text{DDRL}} = \mathbb{E} \left[ \sum_{t=0}^{T} \nabla_{\theta} \log p_{\theta}(x_{t-1} \, | \, x_t, \mathbf{c}) \, r(\mathbf{x}_0, \mathbf{c}) \right]
\tag{DDPOsf}
\end{equation*}
$$
 DDPOsf每轮数据收集只允许进行一次优化步骤，因为梯度必须使用当前参数生成的数据计算。为了执行多步优化，我们可以使用重要性采样估计器DDPOis
$$
\begin{equation*}
\nabla_{\theta} \mathcal{J}_{\text{DDRL}} = \mathbb{E} \left[ \sum_{t=0}^{T} \frac{p_{\theta}(x_{t-1} \, | \, x_t, \mathbf{c})}{p_{\theta_{\text{old}}}(x_{t-1} \, | \, x_t, \mathbf{c})} \nabla_{\theta} \log p_{\theta}(x_{t-1} \, | \, x_t, \mathbf{c}) \, r(\mathbf{x}_0, \mathbf{c}) \right]
\tag{DDPOis}
\end{equation*}
$$

### DDPO应用示例
  奖励函数根据下游任务确定,下图是一个提示-图像对齐的示例，给一个提示文本，扩散模型生成图像，利用VLM模型给生成的图像生成描述，最后再利用LLM模型判断原始提示文本与生成描述的相似度作为奖励。
![](../images/rl_ddpo_example.jpg)


## ReinFlow
&ensp;&ensp;&ensp;&ensp;[ReinFlow: Fine-tuning Flow Matching Policy with Online Reinforcement Learning](https://arxiv.org/pdf/2505.22094)通过注入有界且可学习的噪声来促进在线微调期间的探索。ReinFlow是第一个在线强化学习算法，能够稳定地微调一系列流匹配策略，尤其是在非常少甚至一个去噪步骤下。它具有轻量级的实现、内置的探索机制，并且可以广泛应用于各种流策略变体，包括Rectified Flow和 Shortcut Models。  

&ensp;&ensp;&ensp;&ensp;RL和流匹配结合面临两个问题，一是流匹配是确定性的,而RL需要探索试错来学习;而是RL策略优化需要利用策略下各动作概率分布,流匹配中计算策略下动作概率分布困难



### 流匹配中的动作概率
  流匹配中的向量场$v(t, x)$，它定义了状态的微分方程：
$$
  \frac{d}{dt} \psi_t(x) = v(t, \psi_t(x))
$$
- 其中$\psi_t(x)$是从初始状态x在时间t的流flow，初始条件为$\psi_0(x) = x$。  
- 初始分布为$p_0(\cdot)$，时间t的分布为$p_t(\cdot)$，通过流$\psi_t$ 变换得到。

#### 推导过程
1. **概率密度的连续性方程**：
  在确定性流中，概率密度$p_t(\psi_t(x))$ 满足连续性方程（无扩散项）：  
$$
\begin{equation*}
   \frac{\partial p_t(\psi_t(x))}{\partial t} = - \nabla \cdot (p_t(\psi_t(x)) v(t, \psi_t(x)))
\end{equation*}
$$
  这里，$\nabla \cdot$ 表示散度算子。

2. **沿轨迹的概率密度变化**：
   概率密度$p_t(\psi_t(x))$。这是一个复合函数，其对时间 t 的全导数为：
$$
\begin{equation*}
   \begin{aligned}
    \frac{d}{dt} p_t(\psi_t(x)) &= \frac{\partial p_t}{\partial {d \psi_t(x)}} \cdot \frac{d \psi_t(x)}{dt} \\
    &= \nabla p_t \cdot v \\ 
    &=- p_t \nabla \cdot v \ \ \ \ //分部积分 
    \end{aligned}
\end{equation*}
$$  

3. **对数概率密度的变化**：
   现在计算对数概率密度的导数：  
$$
\begin{equation*}
\begin{aligned}
   \frac{d}{dt} \ln p_t(\psi_t(x)) &= \frac{1}{p_t(\psi_t(x))}  \frac{d}{dt} p_t(\psi_t(x)) \\ 
   &= \frac{1}{-p_t} - p_t \nabla \cdot v(t, \psi_t(x)) \ \ \ //连续性方程定义\\
   &= - \nabla \cdot v(t, \psi_t(x))
\end{aligned}
\end{equation*}
$$  

4. **积分从时间 0 到 1**：
   对上述方程从 `t=0` 到 `t=1` 积分：  
$$
\begin{equation*}
\begin{aligned}
   \int_0^1 \frac{d}{dt} \ln p_t(\psi_t(x)) \, dt = \int_0^1 - \nabla \cdot v(t, \psi_t(x)) \, dt \\
   \ln p_1(\psi_1(x)) - \ln p_0(\psi_0(x)) = - \int_0^1 \nabla \cdot v(t, \psi_t(x)) \, dt
\end{aligned}
\end{equation*}   
$$
   因此，最终得到：  
$$
\begin{equation}
   \ln p_1(\psi_1(x)) = \ln p_0(\psi_0(x)) - \int_0^1 \nabla \cdot v(t, \psi_t(x)) \,dt  \tag 4
\end{equation}
$$
   这就是论文中的公式4。

#### 说明
- 这个公式允许我们计算最终概率密度 $p_1$ 的对数，通过初始概率密度和向量场散度的积分。
- 使用迹估计器（trace estimator）会引入**蒙特卡洛误差**，而通过模拟计算积分会引入**离散化误差**。当使用大步长（即较少的去噪步骤）以追求快速推理时，这种离散化误差会尤其显著。
- 在推理中将流过程视为**离散时间马尔可夫过程**可以缓解这个问题.然而，中间变量遵循确定性转移,那么就无法计算其概率，这使得基于概率的马尔可夫过程方法在此失效。

### 中间动作的确定性
  由于流匹配是ODE建模的，因此每一步都是确定性定，满足如下方程:  
$$
\begin{equation*}
p(X_{t+\Delta t} = x|X_t) = \delta(x - X_t - v_\theta(t,X_t)\Delta t)
\end{equation*}
$$  

### ReinFlow算法

**注入可学习噪声**:ReinFlow通过注入可学习的噪声，将流策略的确定性路径转换为离散时间马尔可夫过程。这一转换使得流模型在任意少的去噪步骤下都能进行精确且简单的似然计算，从而促进探索并确保训练的稳定性。  
ReinFlow训练了一个噪声注入网络，该网络根据当前动作、时间和观测值输出噪声的标准差。这种设计允许噪声网络自动平衡探索与利用，且具有轻量级实现和广泛的适用性。  
**精确似然表达式**: ReinFlow通过将流模型转换为离散时间马尔可夫过程，得到了一个精确且简单的似然表达式。这使得在非常少的去噪步骤下也能进行有效的策略梯度优化。  
**策略梯度定理**: 建立了离散时间马尔可夫过程策略的策略梯度定理，这为ReinFlow的算法设计提供了理论支持，并使得可以应用各种现代深度策略梯度算法来优化噪声注入的流策略。  

  加入噪声网络后每一步的采样概率满足如下公式:  
$$
\begin{equation}
a^0 \sim \mathcal{N}(0, \mathbb{I}_{d_A}), \quad a^{k+1} \sim \mathcal{N}\left( \cdot | a^k + v_\theta(t_i, a^k, o) \Delta t_i, \sigma_{\theta'}^2(t_i, a^k, o) \right)  \tag{6}
\end{equation}
$$  
  整个去噪过程的联合对数概率(马尔可夫性):  
$$
\begin{equation}
\ln \pi(a^0, \ldots, a^K|_0; \theta, \theta') = \ln \mathcal{N}(0, \mathbb{I}_{d_A}) + \sum_{k=0}^{K-1} \ln \mathcal{N}\left( a^{k+1}|a_h^k + v_\theta(t_k, a_h^k, o) \Delta t_k, \sigma_{\theta'}^2(t_k, a^k, o) \right) \tag{7}
\end{equation}
$$  
  在均匀离散化下，$t_k = \frac{k}{K}$ 且 $\Delta t_k = \frac{1}{K}$。

- ReinFlow算法
![算法1](../images/rl_reinflow_alg1.jpg)  
<br>
- 策略梯度-PPO实现 
![算法2](../images/rl_reinflow_alg2.jpg)


## Flow-GRPO
 [Flow-GRPO: Training Flow Matching Models via Online RL](https://arxiv.org/pdf/2505.05470)


## Flow-CPS
[COEFFICIENTS-PRESERVING SAMPLING FOR REINFORCEMENT LEARNING WITH FLOW MATCHING](https://arxiv.org/pdf/2509.05952)

## CFGRL
 [Diffusion Guidance Is a Controllable Policy Improvement Operator](https://arxiv.org/pdf/2505.23458)



### 引理1

### 引理2

### 引理3

### 定理1

### 定理2

## FPO



参考:[](https://arxiv.org/pdf/2107.07599)