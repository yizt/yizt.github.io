---
layout: default
title: 流匹配
nav_order: 4
---

# 流匹配

## 流匹配基础
 本节所有内容均节选自[An Introduction to Flow Matching and Diffusion Models](https://diffusion.csail.mit.edu/docs/lecture-notes.pdf)

### 流模型定义

流模型的目标是将一个简单分布 $p_{\text{init}}$ 转换为一个复杂分布 $p_{\text{data}}$。因此，常微分方程的模拟是这种变换的自然选择。流模型由如下常微分方程描述：

$$
\begin{equation*}
\begin{aligned}
X_{0} \sim p_{\text{init}} \quad \blacktriangleright \text{随机初始化} \\
\frac{\mathrm{d}}{\mathrm{d}t}X_{t} =u^{ \theta}_{t}(X_{t}) \quad \blacktriangleright \text{常微分方程}
\end{aligned}
\end{equation*}
$$  
 

其中向量场$u_{t}^{\theta}$是一个具有参数$\theta$的神经网络$u_{t}^{\theta}$。
即一个具有参数$\theta$ 的连续函数 $u_{t}^{\theta}: \mathbb{R}^{d} \times [0,1] \rightarrow \mathbb{R}^{d}$ 。目标是使轨迹的终点$X_{1}$ 具有分布$p_{\text{data}}$ ，即

$$
\begin{equation*}
X_{1} \sim p_{\text{data}} \iff \psi^{ \theta}_{1}(X_{0}) \sim p_{\text{data}}
\end{equation*}
$$  

其中 $\psi_{t}^{\theta}$ 描述由 $u_{t}^{\theta}$ 诱导的流。注意：虽然它被称为流模型，<span style="color: red;">神经网络参数化的是向量场，而不是流本身</span>。
### 扩散模型定义
  扩散模型可以看作是流模型的SDE版本,在流模型的ODE上增加一个随机项。

$$
\begin{gather*}
X_0 \sim p_{\text{init}} \quad \blacktriangleright \text{随机初始化} \\
dX_t = u^\theta_t(X_t) dt + \sigma dW_t \quad \blacktriangleright \text{随机微分方程}
\end{gather*}
$$ 

$W_t$是维纳过程(布朗运动)

### 训练目标
 流模型和扩散模型，通过模拟以下ODE/SDE获得轨迹 $(X_{t})_{0\leq t\leq 1}$：

$$
\begin{gather*}
X_{0} \sim p_{\text{init}}, \quad dX_{t} = u_{t}^{\theta}(X_{t})\text{d}t \quad \text{（流模型）} \\
X_{0} \sim p_{\text{init}}, \quad dX_{t} = u_{t}^{\theta}(X_{t})\text{d}t + \sigma_{t}\text{d}W_{t} \quad \text{(扩散模型)}
\end{gather*}
$$ 

其中 $u_{t}^{\theta}$ 是一个神经网络，$\sigma_{t}$ 是固定的扩散系数。最小化一个**均方误差**损失函数 $\mathcal{L}(\theta)$ ：

$$
\begin{equation*}
\mathcal{L}(\theta) = \|u_{t}^{\theta}(x) - \underbrace{u_{t}^{\text{target}}(x)}_{ \text{训练目标}}\|^{2}  
\end{equation*}
$$

其中 $u_{t}^{\text{target}}(x)$ 是我们想要逼近的**训练目标**。接下来是**找到训练目标$u_{t}^{\text{target}}$的一个方程**。

#### 条件/边缘概率路径
 在推到训练目标之前，首先定义条件概率路径(conditional probability path)和边缘概率路径(marginal probability path)。  
 构建训练目标 $u_{t}^{\text{target}}$ 的第一步是指定一条**概率路径**。直观上，一条概率路径指定了噪声 $p_{\text{init}}$ 和数据 $p_{\text{data}}$ 之间的渐进插值。对于一个数据点 $z\in\mathbb{R}^{d}$，用 $\delta_{z}$ 表示 **Dirac delta** "分布"。这是可以想象的最简单的分布：从 $\delta_{z}$ 中采样总是返回 $z$（即它是确定性的）。一条**条件（插值）概率路径** 是一组定义在 $\mathbb{R}^{d}$ 上的分布 $p_{t}(x|z)$，满足：

$$
\begin{gather*}
p_{0}(\cdot|z)=p_{\text{init}},\quad p_{1}(\cdot|z)=\delta_{z}\quad\text{ 对于所有 }z\in\mathbb{R}^{d}.
\end{gather*}
$$

换句话说，一条条件概率路径逐渐地将*单个*数据点转换为分布 $p_{\text{init}}$。可以将概率路径视为分布空间中的一条轨迹。每一条条件概率路径 $p_{t}(x \lvert z)$ 都导出一条**边际概率路径** $p_{t}(x)$，其定义为：我们先从数据分布 $p_{\text{data}}$ 中采样一个数据点 $z$，然后从 $p_{t}(\cdot \lvert z)$ 中采样，所获得的分布：

$$
\begin{gather*}
z\sim p_{\text{data}},\quad x\sim p_{t}(\cdot|z)\quad\Rightarrow x \sim p_{t} \quad\blacktriangleright\text{从边际路径采样} \\
p_{t}(x)=\int p_{t}(x|z)p_{\text{data}}(z)\mathrm{d}z \quad\blacktriangleright\text{边际路径的密度}
\end{gather*}
$$

注意，我们知道如何从 $p_{t}$ 中采样，但我们不知道密度值 $p_{t}(x)$，因为该积分是难以处理的。

#### 条件/边缘向量场
  Conditional and Marginal Vector Fields

**定理 12**（连续性方程）

考虑一个具有向量场 $u_t^{\text{target}}$ 的流模型，其中 $X_0 \sim p_{\text{init}}$。那么，对于所有$0 \leq t \leq 1,X_t \sim p_t$ 当且仅当

$$
\begin{equation*}
\partial_t p_t(x) = -\text{div}(p_t u_t^{\text{target}})(x) \quad \text{对于所有 } x \in \mathbb{R}^d, 0 \leq t \leq 1 \qquad(24)
\end{equation*}
$$

其中 $\partial_t p_t(x) = \frac{d}{dt} p_t(x)$ 表示 $p_t(x)$ 的时间导数。方程 (24) 被称为**连续性方程**。

**定理10（边缘化技巧）**

对于每一个数据点$z \in \mathbb{R}^d$，令 $u_t^{\text{target}}(\cdot \lvert z)$ 表示一个条件向量场，其定义使得对应的ODE能生成条件概率路径 $p_t(\cdot \lvert z)$，即： 

$$
X_0 \sim p_{\text{init}}, \quad \frac{d}{dt} X_t = u_t^{\text{target}}(X_t|z) \implies X_t \sim p_t(\cdot|z) \quad (0 \leq t \leq 1).
$$

那么，由下式定义的**边缘向量场** $u_t^{\text{target}}(x)$：
$$
\begin{equation}
u_t^{\text{target}}(x) = \int u_t^{\text{target}}(x|z)\frac{p_t(x|z)p_{\text{data}}(z)}{p_t(x)}dz, \tag{19}
\end{equation}
$$

遵循**边缘概率路径**，即：

$$X_0 \sim p_{\text{init}}, \quad \frac{d}{dt} X_t = u_t^{\text{target}}(X_t) \implies X_t \sim p_t \quad (0 \leq t \leq 1).$$

特别地，对于此 ODE，有 $X_1 \sim p_{\text{data}}$，因此我们可以说：
**由方程 $X_0 \sim p_{\text{init}},\  \frac{d}{dt} X_t = u_t^{\text{target}}(X_t)$ 描述的向量场 $u_t^{\text{target}}$ 将噪声 $p_{\text{init}}$ 转换为数据 $p_{\text{data}}$。"**

说明:由公式19可知,<span style='color:red'>边缘向量场等于各数据的$z$的条件向量场加权求和/积分,权重等于给定$x$时数据点$z$后验概率$p_t(z \lvert x)=\frac{p_t(x \vert z)p_{\text{data}}(z)}{p_t(x)}$</span>。

**证明：** 根据连续性方程，我们需要证明由方程 (19) 所定义的边缘向量场 $u_{t}^{\text{target}}$ 满足连续性方程。可以通过直接计算来证明这一点： 

$$
\begin{aligned}
\partial_{t}p_{t}(x) &=\partial_{t}\int p_{t}(x|z)p_{\text {data}}(z)\mathrm{d}z \quad //p_{t}定义\\
&=\int\partial_{t}p_{t}(x|z)p_{\text{data}}(z)\mathrm{d}z \\
&= \int-\mathrm{div}(p_{t}(\cdot|z) u_{t}^{\text{target}}(\cdot|z))(x)p_{\text{data}}(z)\mathrm{d}z \quad//条件概率路径的连续性方程\\
&=-\mathrm{div}\left(\int p_{t}(x |z)u_{t}^{\text{target}}(x|z)p_{\text{data}}(z)\mathrm{d}z\right) \quad //积分和散度交换\\
&=-\mathrm{div}\left(p_{t}(x)\int u _{t}^{\text{target}}(x|z)\frac{p_{t}(x|z)p_{\text{data}}(z)}{p_{t}(x)}\mathrm{d }z\right)(x)\quad //分子分母同乘p_{t}(x)\\
&=-\mathrm{div}\left(p_{t}u_{t}^{ \text{target}}\right)(x) \quad //边缘向量场定义
\end{aligned}
$$  


上述等式链的开头和结尾表明，对于 $u_{t}^{\text{target}}$，连续性方程成立。至此已经推导出训练目标$u_{t}^{\text{target}}$的表达式。

#### 流匹配目标推导
  虽然已经有了目标向量场$u_{t}^{\text{target}}$的表达式，但是精确计算需要在所有数据点z上积分,但是条件向量场$u_{t}^{\text{target}}(x|z)$很容易计算。接下来定义`flow matching loss`和`conditional flow matching loss`，并证明它们是等价的。

**flow matching loss**  
  流匹配损失定义如下： 

$$
\begin{align*}
\mathcal{L}_{\mathrm{FM}}(\theta) &= \mathbb{E}_{t \sim \operatorname{Unif}, x \sim p_{t}}\left[\left\|u_{t}^{\theta}(x)-u_{t}^{\mathrm{target}}(x)\right\|^{2}\right] \\
&=\mathbb{E}_{t \sim \operatorname{Unif}, z \sim p_{\mathrm{data}}, x \sim p_{t}(\cdot \mid z)}\left[\left\|u_{t}^{\theta}(x)-u_{t}^{\mathrm{target}}(x)\right\|^{2}\right] \tag{42}
\end{align*}
$$  


**conditional flow matching loss**  
  条件流匹配损失定义如下：
$$
\begin{equation}
\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t \sim \text{Unif}[0,1], \, z \sim p_{\text{data}}, \, x \sim p_t(x|z)} \left[ \lVert u_t^\theta(x) - u_t^{\text{target}}(x|z) \rVert^2 \right] \tag {43}
\end{equation}
$$


**证明：** 该证明通过将均方误差展开为三个分量并去除常数项来完成：

$$
\begin{align*}
\mathcal{L}_{\text{FM}}(\theta) &= \mathbb{E}_{t\sim\text{Unif},x\sim p_t}[\|u^\theta_t(x)-u^{\text{target}}_{t}(x)\|^2] \\
&= \mathbb{E}_{t\sim\text{Unif},x\sim p_t}[\|u^\theta_t(x)\|^2-2u^\theta_t(x)^T u^{\text{target}}_{t}(x)+\|u^{\text{target}}_{t}(x)\|^2] \\
&= \mathbb{E}_{t\sim\text{Unif},x\sim p_t}\left[\|u^\theta_t(x)\|^2\right]-2\mathbb{E}_{t\sim\text{Unif},x\sim p_t}[u^\theta_t(x)^T u^{\text{target}}_{t}(x)]+ \underbrace{\mathbb{E}_{t\sim\text{Unif}_{[0,1]},x\sim p_t}[\|u^{\text{target}}_{t}(x)\|^2]}_{=:C_1} \quad /最后一项跟\theta无关 \\
&= \mathbb{E}_{t\sim\text{Unif},z\sim p_{\text{data}},x\sim p_{t}(\cdot|z)}[\|u^\theta_t(x)\|^2]-2\mathbb{E}_{t\sim\text{Unif},x\sim p_t}[u^\theta_t(x)^T u^{\text{target}}_{t}(x)]+C_1
\end{align*}
$$

重新表达第二项：

$$
\begin{align*}
\mathbb{E}_{t \sim \text{Unif}, x \sim p_t}[u_t^\theta(x)^T u_t^{\text{target}}(x)] &= \int_0^1 \int_{\mathbb{R}^d} p_t(x) u_t^\theta(x)^T u_t^{\text{target}}(x) \, dx \, dt \quad //期望积分表示,0\sim1的均匀分布概率密度恒为1 \\
&= \int_0^1 \int_{\mathbb{R}^d} p_t(x) u_t^\theta(x)^T \left[ \int_{\mathbb{R}^d} u_t^{\text{target}}(x|z) \frac{p_t(x|z)p_{\text{data}}(z)}{p_t(x)} \, dz \right] \, dx \, dt \qquad //边缘向量场定义\\
&= \int_0^1 \int_{\mathbb{R}^d} \int_{\mathbb{R}^d} u_t^\theta(x)^T u_t^{\text{target}}(x|z) p_t(x|z) p_{\text{data}}(z) \, dz \, dx \, dt \quad //p_t(x)消除 \\
&\overset{(iv)}{=} \mathbb{E}_{t \sim \text{Unif}, z \sim p_{\text{data}}, x \sim p_t(\cdot | z)} [u_t^\theta(x)^T u_t^{\text{target}}(x|z)] \quad //积分改期望表示
\end{align*}
$$

 将上式结论代入原流匹配损失等式有：

$$
\begin{align*}
\mathcal{L}_{\text{FM}}(\theta) &= \mathbb{E}_{t\sim\text{Unif},z\sim p_{\text{data}},x\sim p_{t}(\cdot|z)}[\|u_{t}^{\theta}(x)\|^{2}] - 2\mathbb{E}_{t\sim\text{Unif},z\sim p_{\text{data}},x\sim p_{t}(\cdot|z)}[u_{t}^{\theta}(x)^{T}u_{t}^{\text{target}}(x|z)] + C_1 \\
&= \mathbb{E}_{t\sim\text{Unif},z\sim p_{\text{data}},x\sim p_{t}(\cdot|z)}\left[\|u_{t}^{\theta}(x)\|^{2} - 2u_{t}^{\theta}(x)^{T}u_{t}^{\text{target}}(x|z) + {\|u_{t}^{\text{target}}(x|z)\|^{2} - \|u_{t}^{\text{target}}(x|z)\|^{2}}\right] + C_1 \quad //同时加减常数项 \\ 
&= \mathbb{E}_{t\sim\text{Unif},z\sim p_{\text{data}},x\sim p_{t}(\cdot|z)}[\|u_{t}^{\theta}(x) - u_{t}^{\text{target}}(x|z)\|^{2}] + \underbrace{\mathbb{E}_{t,z,x}[-\|u_{t}^{\text{target}}(x|z)\|^{2}]}_{C_2} + C_1 \\
&= \mathcal{L}_{\text{CFM}}(\theta) + \underbrace{C_2 + C_1}_{=:C}
\end{align*}
$$

 至此已经证明流匹配损失与条件流匹配损是等价的，它们仅相差常数项，而条件流匹配损失中的条件向量场是容易计算的。因此流匹配训练目标直接使用<span style="color: red;">公式43定义的条件流匹配损失</span>。

### Score Matching
 同样的方式可以定义条件得分(conditional score)和边缘得分(marginal score).

\(\nabla \log p_t(x)\) 是**边缘得分(marginal score)** 函数定义如下:
\[\nabla \log p_t(x) = \int \nabla \log p_t(x|z) \frac{p_t(x|z)p_{\text{data}}(z)}{p_t(x)} dz. \tag{51}\]
 \(\nabla \log p_t\), 可以使用一个称为得分网络**score network**\( s_t^\theta : \mathbb{R}^d \times [0, 1] \rightarrow \mathbb{R}^d \) 的神经网络来近似，类似向量场定义**score matching** 损失和 **conditional score matching** 损失如下:
$$
\begin{align*}
\mathcal{L}_{\text{SM}}(\theta) &= \mathbb{E}_{t \sim \text{Unif}, z \sim p_{\text{data}}, x \sim p_t(\cdot |z)}[||s_t^\theta(x) - \nabla \log p_t(x)||^2] \quad \Rightarrow \quad \text{score matching loss} \\
\mathcal{L}_{\text{CSM}}(\theta) &= \mathbb{E}_{t \sim \text{Unif}, z \sim p_{\text{data}}, x \sim p_t(\cdot |z)}[||s_t^\theta(x) - \nabla \log p_t(x)||^2] \quad \Rightarrow \quad \text{conditional score matching loss}
\end{align*}
$$
 边缘得分损失与条件得分损失仍然等价.

**定理20**

得分匹配损失与条件得分匹配损失之间相差一个常数：:
\[\mathcal{L}_{\text{SM}}(\theta) = \mathcal{L}_{\text{CSM}}(\theta) + C,\]
 \(C\) 与参数 \(\theta\)无关. 因此，它们的梯度一致:
\[\nabla_\theta \mathcal{L}_{\text{SM}}(\theta) = \nabla_\theta \mathcal{L}_{\text{CSM}}(\theta).\]

**证明.** 注意 \(\nabla \log p_t\) 的公式与 \(u_t^{\text{target}}\) 的公式形式相同，证明过程完全一样。

### 常用的条件流向量场
 目前已知流匹配的训练目标只需要计算条件向量场即可，那么对于常用的高斯概率路径

**条件速度场**

 对于通用噪声调度 $\alpha_t, \beta_t$，条件为数据点z的分布为 $ p_t(\cdot\mid z) = \mathcal{N}(\alpha_t z, \beta_t^2 I_d) $。令 $ \dot{\alpha}_t = \partial_t \alpha_t $ 和 $ \dot{\beta}_t = \partial_t \beta_t $ 分别表示 $\alpha_t$ 和 $\beta_t$ 的时间导数，则条件速度场$u_t^{\text{target}}(x\mid z)$。
$$
\begin{align*}
  u_t^{\text{target}}(x|z) &= \frac{dx}{dt} \\
  &= \frac{d}{dt} (\alpha_t z + \beta_t x_0) \quad //x=\alpha_t z + \beta_t x_0, x_0 \in \\ x=\mathcal{N}(\mathcal{0}, I_d) \\
  &=\dot{\alpha}_t z + \dot{\beta}_t x_0 \\
  &=\dot{\alpha}_t z + \dot{\beta}_t (\frac {x-\alpha_t z} {\beta_t}) \\
  &=\left( \dot{\alpha}_t - \frac{\dot{\beta}_t}{\beta_t} \alpha_t \right) z + \frac{\dot{\beta}_t}{\beta_t} x
\end{align*}
$$

**条件得分**

 对于条件高斯路径 $p_t(x\mid z) = \mathcal{N}(\alpha_t z, \beta_t^2 I_d)$, 其条件得分为：

$$
\begin{equation*}
  \nabla \log p_t(x|z) = -\frac{x - \alpha_t z}{\beta_t^2}
\end{equation*}
$$

**条件速度场与条件得分的关系**

对于条件向量场和条件得分，我们可以推导出：
$$
\begin{align*}
u_{t}^{\text{target}}(x|z) &= \left( \dot{\alpha}_{t} - \frac{\dot{\beta}_{t}}{\beta_{t}} \alpha_{t} \right) z + \frac{\dot{\beta}_{t}}{\beta_{t}} x \\
&= \left( \beta_{t}^{2} \frac{\dot{\alpha}_{t}}{\alpha_{t}} - \dot{\beta}_{t} \beta_{t} \right) \left( \frac{\alpha_{t} z - x}{\beta_{t}^{2}} \right) + \frac{\dot{\alpha}_{t}}{\alpha_{t}} x \quad //展开\dot{\alpha_tz} -\frac {\dot{\alpha_t}} {\alpha_t}x - \frac {\dot{\beta_t}}{\beta_t} \alpha_tz + \frac{\dot{\beta}_{t}}{\beta_{t}} x + \frac{\dot{\alpha}_{t}}{\alpha_{t}} x 与上市相等\\
&= \left( \beta_{t}^{2} \frac{\dot{\alpha}_{t}}{\alpha_{t}} - \dot{\beta}_{t} \beta_{t} \right) \nabla \log p_{t}(x|z) + \frac{\dot{\alpha}_{t}}{\alpha_{t}} x \quad //条件得分函数
\end{align*}
$$

**边缘得分与边缘速度场关系**

  通过积分，同样的恒等式对于边际流向量场和边际得分函数也成立：
$$
\begin{align*}
u_{t}^{\text{target}}(x) &= \int u_{t}^{\text{target}}(x|z) \frac{p_{t}(x|z) p_{\text{data}}(z)}{p_{t}(x)} dz \\
&= \int \left[ \left( \beta_{t}^{2} \frac{\dot{\alpha}_{t}}{\alpha_{t}} - \dot{\beta}_{t} \beta_{t} \right) \nabla \log p_{t}(x|z) + \frac{\dot{\alpha}_{t}}{\alpha_{t}} x \right] \frac{p_{t}(x|z) p_{\text{data}}(z)}{p_{t}(x)} dz \\
&= \left( \beta_{t}^{2} \frac{\dot{\alpha}_{t}}{\alpha_{t}} - \dot{\beta}_{t} \beta_{t} \right)\int  \nabla \log p_{t}(x|z) \frac{p_{t}(x|z) p_{\text{data}}(z)}{p_{t}(x)} dz + \frac{\dot{\alpha}_{t}}{\alpha_{t}} x \int \frac{p_{t}(x|z) p_{\text{data}}(z)}{p_{t}(x)} dz \quad //不含z的常数放到积分外边\\
&= \left( \beta_{t}^{2} \frac{\dot{\alpha}_{t}}{\alpha_{t}} - \dot{\beta}_{t} \beta_{t} \right) \nabla \log p_{t}(x) + \frac{\dot{\alpha}_{t}}{\alpha_{t}} x \quad //边缘得分定义及\int \frac{p_{t}(x|z) p_{\text{data}}(z)}{p_{t}(x)} dz=1
\end{align*}
$$

可以使用转换公式将分数网络 \( s_t^\theta \) 与向量场网络 \( u_t^\theta \) 通过以下方式相互参数化：

\[
u_t^\theta(x) = \left( \beta_t^2 \frac{\dot{\alpha}_t}{\alpha_t} - \dot{\beta}_t \beta_t \right) s_t^\theta (x) + \frac{\dot{\alpha}_t}{\alpha_t} x. \tag{54}
\]

类似地，只要 \( \beta_t^2 \dot{\alpha}_t - \alpha_t \dot{\beta}_t \beta_t \neq 0 \)（对于 \( t \in [0,1] \) 始终成立），则有

\[
s_t^\theta (x) = \frac{\alpha_t u_t^\theta (x) - \dot{\alpha}_t x}{\beta_t^2 \dot{\alpha}_t - \alpha_t \dot{\beta}_t \beta_t}. \tag{55}
\]

因此<span style='color:red'>无需同时训练向量场网络和得分网络,训练了其中任意个,另一个可通过已训练网络转换，公式54和55表示了它们的转换关系</span>。

注意到$\frac {d} {dt} ln(\alpha_t) = \frac{\dot{\alpha}_t}{\alpha_t}$及$\frac {d} {dt} ln(\beta_t) = \frac{\dot{\beta_t}}{\beta_t}$，公式54也可以写为如下公式: 

$$
\begin{align*}
u_t^\theta(x) &= \left( \beta_t^2 \frac{\dot{\alpha}_t}{\alpha_t} - \dot{\beta}_t \beta_t \right) s_t^\theta (x) + \frac{\dot{\alpha}_t}{\alpha_t} x \tag{54} \\
&= \beta_t^2 \left( \frac{\dot{\alpha}_t}{\alpha_t} - \frac {\dot{\beta}_t} {\beta_t} \right) s_t^\theta (x) + \frac{\dot{\alpha}_t}{\alpha_t} x \\
&= \beta_t^2 \left( \frac {d ln(\alpha_t)} {dt}  - \frac {d ln(\beta_t)} {dt}  \right) s_t^\theta (x) + \frac {d ln(\alpha_t)} {dt} x \\
&=\beta_t^2  \frac {d ln(\frac {\alpha_t} {\beta_t})} {dt}   s_t^\theta (x) + \frac {d ln(\alpha_t)} {dt} x
\end{align*}
$$  
 公式55也可以写为如下公式: 
$$
\begin{align*}
s_t^\theta (x) &= \frac{\alpha_t u_t^\theta (x) - \dot{\alpha}_t x}{\beta_t^2 \dot{\alpha}_t - \alpha_t \dot{\beta}_t \beta_t}. \tag{55} \\
&= \frac {u_t^\theta (x) - \frac {d ln(\alpha_t)} {dt} x} {\beta_t^2  \frac {d ln(\frac {\alpha_t} {\beta_t})} {dt}  }
\end{align*}
$$

Rectified Flow、MeanFlow、Consistency-FM、Shortcut Model，OT-CFM