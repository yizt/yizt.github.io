---
layout: default
title: 预备知识
nav_order: 1
---

# 预备知识

## ODE中的伴随方法
 本节内容来自[Neural Ordinary Differential Equations](https://arxiv.org/pdf/1806.07366)
 ResNet残差网络可看作连续转换的欧拉离散化：
 \[
\mathbf{h}_{t+1} = \mathbf{h}_t + f(\mathbf{h}_t, \boldsymbol{\theta}_t) \tag 1
\]
 等步长足够小,层数足够多,极限情况下用一个由神经网络常微分方程 (ODE) 来参数化隐藏单元的连续动力学。
\[
\frac{d\mathbf{h}(t)}{dt} = f(\mathbf{h}(t), t, \theta)  \tag 2
\]
 考虑优化一个标量损失函数 $L()$，其输入是 ODE 求解器的结果：

$$
L(\mathbf{z}(t_1)) = L\left(\mathbf{z}(t_0) + \int_{t_0}^{t_1} f(\mathbf{z}(t), t, \theta) dt\right) = L(\text{ODESolve}(\mathbf{z}(t_0), f, t_0, t_1, \theta))  \tag 3
$$

 为优化L，我们需要计算关于θ的梯度。第一步是确定损失函数的梯度如何依赖于每个瞬时的隐藏状态 z(t)。这个量被称为伴随状态 $a(t) = ∂L/∂z(t)$。它的动力学由另一个 ODE 给出，可以看作是链式法则的瞬时模拟.

\[
\frac{d\mathbf{a}(t)}{dt} = -\mathbf{a}(t)^\mathsf{T} \frac{\partial f(\mathbf{z}(t), t, \theta)}{\partial \mathbf{z}}  \tag 4
\]

可以通过再次调用一个 ODE 求解器来计算 $\frac{\partial L}{\partial z(t_0)}$​。该求解器必须反向运行，从 $\frac{\partial L}{\partial z(t_1)}$​ 的初始值开始。

计算关于参数 $\theta$ 的梯度有需要一个积分，该积分同时依赖于 $z(t)$ 和 $a(t)$:

\[
\frac{dL}{d\theta} = - \int_{t_1}^{t_0} \mathbf{a}(t)^\top \frac{\partial f(\mathbf{z}(t), t, \theta)}{\partial \theta} dt  \tag 5
\]

### 伴随方法证明
 令 $z(t)$ 遵循微分方程 $\frac{dz(t)}{dt} = f (z(t), t, θ)$，其中 θ是参数。我们将证明，如果我们定义一个伴随状态:

$$
a(t) = \frac{dL}{dz(t)} \tag {34}
$$
 则伴随状态满足如下微分方程：
$$
 \frac{da(t)}{dt} = -a(t) \frac{\partial f(z(t), t, \theta)}{\partial z(t)} \tag {35}
$$

 在传统的离散层神经网络中，梯度通过链式法则从输出层向回传播：

$$
\frac{dL}{dh_t} = \frac{dL}{dh_{t+1}} \frac{dh_{t+1}}{dh_t}​​ \tag {36}
$$

这里$\frac{dL}{dh_t}$​ 是当前层 $h_t$的梯度，它依赖于下一层 $h_{t+1}$​ 的梯度 $\frac{dL}{dh_{t+1}}$​ 和从 $h_t$​ 到 $h_{t+1}$​ 的转换雅可比矩阵 $\frac{dh_{t+1}}{dh_t}​$​。
 对连续隐藏状态,在很小的时间上$\varepsilon$的改变:
$$
z(t + ε) = \int_{t}^{t+\varepsilon} f(z(t), t, \theta) \mathrm{d} t + z(t) = T_{\varepsilon}(z(t), t)  \tag {37}
$$

应用链式法则有:

$$
\frac{dL}{\partial z(t)} = \frac{dL}{dz(t + \varepsilon)} \frac{dz(t + \varepsilon)}{dz(t)} \quad​ \text{或者} \quad a(t) = a(t + \varepsilon) \frac{\partial T_\varepsilon(z(t), t)}{\partial z(t)} \tag {38}
​$$


## HJB方程
 随机最优控制问题中Hamilton-Jacobi-Bellman (HJB)方程，其推导基于动态规划原理和伊藤引理，下面给出详细步骤。

### 1. 问题设定
考虑如下随机控制系统：
\[
dX_s = \bigl(b(X_s,s) + \sigma(X_s,s) u_s\bigr) ds + \sigma(X_s,s) dW_s, \quad s \in [t,T],
\]
初始条件 \(X_t = x\)。其中 \(u_s\) 是控制变量，\(W_s\) 是标准布朗运动。代价函数为

$$
J(x,t;u) = \mathbb{E}\left[ \int_t^T \Bigl(f(X_s,s) + \frac{1}{2}\|u_s\|^2\Bigr) ds + g(X_T) \;\middle|\; X_t=x \right].
$$

目标是求最优控制 \(u^*\) 最小化代价，定义值函数
\[
V(x,t) = \inf_{u} J(x,t;u).
\]

### 2. 动态规划原理
对于充分小的 \(\Delta t > 0\)，由动态规划原理有

$$
V(x,t) = \inf_{u} \mathbb{E}\left[ \int_t^{t+\Delta t} \Bigl(f+\frac{1}{2}\|u\|^2\Bigr) ds + V(X_{t+\Delta t}, t+\Delta t) \;\middle|\; X_t=x \right].
$$

### 3. 伊藤引理与期望展开
对 \(V(X_{t+\Delta t}, t+\Delta t)\) 应用伊藤引理（假设 \(V\) 足够光滑）：
\[
dV = \left( \partial_t V + (b+\sigma u)\cdot\nabla V + \frac{1}{2}\operatorname{tr}\bigl(\sigma\sigma^\top \nabla^2 V\bigr) \right) dt + (\sigma^\top \nabla V)\cdot dW.
\]
在区间 \([t, t+\Delta t]\) 上积分并取条件期望（注意布朗运动项期望为零）：

$$
\mathbb{E}[V(X_{t+\Delta t}, t+\Delta t) \mid X_t=x] = V(x,t) + \mathbb{E}\left[ \int_t^{t+\Delta t} \left( \partial_t V + (b+\sigma u)\cdot\nabla V + \frac{1}{2}\operatorname{tr}(\sigma\sigma^\top \nabla^2 V) \right) ds \mid X_t=x \right] + o(\Delta t).
$$

代入动态规划方程，两边消去 \(V(x,t)\)，并除以 \(\Delta t\) 令 \(\Delta t\to 0\)，得到：

$$
0 = \inf_u \left\{ \partial_t V + (b+\sigma u)\cdot\nabla V + \frac{1}{2}\operatorname{tr}(\sigma\sigma^\top \nabla^2 V) + f + \frac{1}{2}\|u\|^2 \right\}.
$$

### 4. 极小化处理
对括号内关于 \(u\) 求极小值。由于 $\|u\|^2 + 2u^\top (\sigma^\top \nabla V) = \|u + \sigma^\top \nabla V\|^2 - \|\sigma^\top \nabla V\|^2$，极小化等价于令 $u = -\sigma^\top \nabla V$，此时极小值为
\[
\partial_t V + b\cdot\nabla V + \frac{1}{2}\operatorname{tr}(\sigma\sigma^\top \nabla^2 V) + f - \frac{1}{2}\|\sigma^\top \nabla V\|^2 = 0.
\]

### 5. 最终方程与边界条件
整理即得HJB方程：
\[
(\partial_t + \mathcal{L})V - \frac{1}{2}\|\sigma^\top \nabla V\|^2 + f = 0, \quad V(x,T)=g(x),
\]
其中 \(\mathcal{L}\) 为无穷小生成元：
\[
\mathcal{L} = b\cdot\nabla + \frac{1}{2}\operatorname{tr}(\sigma\sigma^\top \nabla^2).
\]
这正是公式(4)的形式（注意原式中的 \(L\) 定义与此一致，但可能符号表示略有差异，本质相同）。

### 6. 说明
若原问题中扩散项系数为 \(\lambda \sigma\sigma^\top\)（例如带参数 \(\lambda\)），则生成元中的二阶项相应调整，但推导过程完全类似。公式(4)中的非线性项 \(-\frac{1}{2}\|\sigma^\top \nabla V\|^2\) 正是由控制代价的二次型及最优反馈控制代入产生。
