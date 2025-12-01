---
layout: default
title: 基于扩散模型的强化学习
nav_order: 3
---

# 基于扩散模型的强化学习



## QSM
 [Learning a Diffusion Model Policy from Rewards via Q-Score Matching](https://arxiv.org/pdf/2312.11752)




## ReinFlow
&ensp;&ensp;&ensp;&ensp;[ReinFlow: Fine-tuning Flow Matching Policy with Online Reinforcement Learning](https://arxiv.org/pdf/2505.22094)通过注入有界且可学习的噪声来促进在线微调期间的探索。ReinFlow是第一个在线强化学习算法，能够稳定地微调一系列流匹配策略，尤其是在非常少甚至一个去噪步骤下。它具有轻量级的实现、内置的探索机制，并且可以广泛应用于各种流策略变体，包括Rectified Flow和 Shortcut Models。  

&ensp;&ensp;&ensp;&ensp;RL和流匹配结合面临两个问题，一是流匹配是确定性的,而RL需要探索试错来学习;而是RL策略优化需要利用策略下各动作概率分布,流匹配中计算策略下动作概率分布困难

公式4来源于流匹配（flow matching）或连续归一化流（continuous normalizing flows）理论，描述了在连续时间变化下概率密度的演化。以下是该公式的详细推导过程。

### 背景设定
我们有一个向量场$v(t, x)$，它定义了状态的微分方程：
$$
  \frac{d}{dt} \psi_t(x) = v(t, \psi_t(x))
$$
- 其中$\psi_t(x)$是从初始状态x在时间t的流flow，初始条件为$\psi_0(x) = x$。  
- 初始分布为$p_0(\cdot)$，时间t的分布为$p_t(\cdot)$，通过流$\psi_t$ 变换得到。

### 推导步骤
1. **概率密度的连续性方程**：
   在确定性流动中，概率密度$p_t(\psi_t(x))$ 满足连续性方程（无扩散项）：  
$$
   \frac{\partial p_t(\psi_t(x))}{\partial t} = - \nabla \cdot (p_t(\psi_t(x)) v(t, \psi_t(x)))
$$
   这里，$\nabla \cdot$ 表示散度算子。

1. **沿轨迹的概率密度变化**：
   概率密度$p_t(\psi_t(x))$。这是一个复合函数，其对时间 t 的全导数为：
   $$\begin{align*}
    \frac{d}{dt} p_t(\psi_t(x)) &= \frac{\partial p_t}{\partial {d \psi_t(x)}} \cdot \frac{d \psi_t(x)}{dt} \\
    &= \nabla p_t \cdot v \\
    &=- p_t \nabla \cdot v
    \end{align*}
   $$

2. **对数概率密度的变化**：
   现在计算对数概率密度的导数：  
$$
   \frac{d}{dt} \ln p_t(\psi_t(x)) = \frac{1}{p_t(\psi_t(x))} \frac{d}{dt} p_t(\psi_t(x)) = - \nabla \cdot v(t, \psi_t(x))
$$

3. **积分从时间 0 到 1**：
   对上述方程从 \( t = 0 \) 到 \( t = 1 \) 积分：
   $$
   \int_0^1 \frac{d}{dt} \ln p_t(\psi_t(x)) \, dt = \int_0^1 - \nabla \cdot v(t, \psi_t(x)) \, dt
   $$
   左边积分是：
   $$
   \ln p_1(\psi_1(x)) - \ln p_0(\psi_0(x)) = - \int_0^1 \nabla \cdot v(t, \psi_t(x)) \, dt
   $$
   因此，最终得到：
   $$
   \ln p_1(\psi_1(x)) = \ln p_0(\psi_0(x)) - \int_0^1 \nabla \cdot v(t, \psi_t(x)) \, dt
   $$
   这就是公式4。

### 说明
- 这个公式允许我们计算最终概率密度 $p_1$ 的对数，通过初始概率密度和向量场散度的积分。
- 在实际应用中，通常需要数值方法（如数值求解器和散度估计）来近似这个积分，因为解析解可能难以获得。
- 公式4在流匹配模型中用于政策梯度方法，特别是在连续控制任务中，以处理动作随机性。


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