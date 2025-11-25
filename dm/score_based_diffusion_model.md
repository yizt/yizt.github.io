---
layout: default
title: 基于得分的扩散模型
nav_order: 3
---


# 基于得分的扩散模型

## Score Matching
  [Estimation of Non-Normalized Statistical Models by Score Matching](https://jmlr.org/papers/volume6/hyvarinen05a/hyvarinen05a.pdf)提出了一种用于估计非归一化统计模型的新方法——分数匹配。该方法巧妙地避免了计算模型中难以处理的归一化常数，为后续的扩散模型等研究奠定了重要基础。

### 得分定义
一个概率分布的得分函数是其对数概率密度关于数据的梯度 
$$
\boldsymbol{S}(\boldsymbol{x} ;\boldsymbol{\theta})=\left(\begin{array}{c}
\frac{\partial\log p(\boldsymbol{x} ;\boldsymbol{\theta})}{\partial x_{1}}\\
\vdots\\
\frac{\partial\log p(\boldsymbol{x} ;\boldsymbol{\theta})}{\partial x_{n}}
\end{array}\right)=\left(\begin{array}{c}
S_{1}(\boldsymbol{x} ;\boldsymbol{\theta})\\
\vdots\\
S_{n}(\boldsymbol{x} ;\boldsymbol{\theta})
\end{array}\right)=\nabla_{\boldsymbol{x}}\log p(\boldsymbol{x} ;\boldsymbol{\theta})
$$

### 训练目标
目标函数：最小化模型分数函数与真实数据分数函数之间的期望平方差  
$$
\begin{align*}
J(\boldsymbol{\theta}) &=\frac{1}{2} \int_{\boldsymbol{x} \in \mathbb{R}^{n}} p_{\mathbf{d}}(\boldsymbol{x})\left\|\boldsymbol{S_m}(\boldsymbol{x} ; \boldsymbol{\theta})-\boldsymbol{S}_{\mathbf{d}}(\boldsymbol{x})\right\|^{2} d \boldsymbol{x} \\
& = \frac{1}{2}\int_{\mathbb{R}^{n}} p_{\mathrm{d}}(\boldsymbol{x}) \left[ \|\boldsymbol{S}_{m}(\boldsymbol{x};\boldsymbol{\theta})\|^2 - 2\boldsymbol{S}_{m}(\boldsymbol{x};\boldsymbol{\theta})^{\top}\boldsymbol{S}_{\mathrm{d}}(\boldsymbol{x}) + \|\boldsymbol{S}_{\mathrm{d}}(\boldsymbol{x})\|^2 \right] d\boldsymbol{x} \\
& = \frac{1}{2}\int_{\mathbb{R}^{n}} p_{\mathrm{d}}(\boldsymbol{x}) \left[ \|\boldsymbol{S}_{m}(\boldsymbol{x};\boldsymbol{\theta})\|^2 - 2\boldsymbol{S}_{m}(\boldsymbol{x};\boldsymbol{\theta})^{\top}\boldsymbol{S}_{\mathrm{d}}(\boldsymbol{x}) \right] d\boldsymbol{x} + C
\end{align*} 
$$

第二项通过分部积分化简 

$$
\begin{align*}
\int_{\mathbb{R}^{n}} p_{\mathrm{d}}(\boldsymbol{x}) \left( -2 \boldsymbol{S}_{m}(\boldsymbol{x};\boldsymbol{\theta})^{\top} \boldsymbol{S}_{\mathrm{d}}(\boldsymbol{x}) \right) d\boldsymbol{x} & = \int_{\mathbb{R}^{n}} p_{\mathrm{d}}(\boldsymbol{x}) \left( -2 \boldsymbol{S}_{m}(\boldsymbol{x};\boldsymbol{\theta})^{\top} \frac{\nabla_{\boldsymbol{x}} p_{\mathrm{d}}(\boldsymbol{x})}{p_{\mathrm{d}}(\boldsymbol{x})} \right) d\boldsymbol{x} \\
& = -2 \int_{\mathbb{R}^{n}} \boldsymbol{S}_{m}(\boldsymbol{x};\boldsymbol{\theta})^{\top} \nabla_{\boldsymbol{x}} p_{\mathrm{d}}(\boldsymbol{x})  d\boldsymbol{x} \\
& = 2 \int_{\mathbb{R}^{n}} div (\boldsymbol{S}_{m}(\boldsymbol{x};\boldsymbol{\theta}))  p_{\mathrm{d}}(\boldsymbol{x})  d\boldsymbol{x}
\end{align*} 
$$

代入原等式可得:  

$$
\begin{align*}
J(\boldsymbol{\theta}) & = \frac{1}{2}\int_{\mathbb{R}^{n}} p_{\mathrm{d}}(\boldsymbol{x}) \left[ \|\boldsymbol{S}_{m}(\boldsymbol{x};\boldsymbol{\theta})\|^2 - 2\boldsymbol{S}_{m}(\boldsymbol{x};\boldsymbol{\theta})^{\top}\boldsymbol{S}_{\mathrm{d}}(\boldsymbol{x}) \right] d\boldsymbol{x} + C \\
&= \frac{1}{2}\int_{\mathbb{R}^{n}} p_{\mathrm{d}}(\boldsymbol{x}) \left[ \|\boldsymbol{S}_{m}(\boldsymbol{x};\boldsymbol{\theta})\|^2 + 2div(\boldsymbol{S}_{m,i}(\boldsymbol{x};\boldsymbol{\theta}))  \right] d\boldsymbol{x} + C \\
&= \int_{\mathbb{R}^{n}} p_{\mathrm{d}}(\boldsymbol{x}) \left[ \frac{1}{2}\|\boldsymbol{S}_{m}(\boldsymbol{x};\boldsymbol{\theta})\|^2 + div(\boldsymbol{S}_{m,i}(\boldsymbol{x};\boldsymbol{\theta}))  \right] d\boldsymbol{x} + C \\
\end{align*}
$$

### 采样方式
给定得分函数$\boldsymbol{S}(\boldsymbol{x};\boldsymbol{\theta})=\nabla_{\boldsymbol{x}}\log p(\boldsymbol{x};\boldsymbol{\theta})$后可以通过朗之万动力学采样。  
<span style="color: red;">为什么朗之万动力学可以采样得分函数？</span>  
由连续性方程与福克普朗克方程等价关系可知，对于速度场为0的ODE其等价的SDE方程为： 
$$
dX_t = \frac{\sigma^2_t}{2} \nabla_x \log p(X_t) dt + \sigma_t dW
$$

根据欧拉-丸山法，其离散形式为： 
$$
\mathbf{x}_{t+1}=\mathbf{x}_t+ \frac {\sigma^2_t} 2 \nabla_{\mathbf{x}} \log p\left(\mathbf{x}_t\right)\Delta t+\sigma_t \sqrt{\Delta t} \mathbf{z}, \quad \mathbf{z} \sim \mathcal{N}(0, \mathbf{I})  
$$

令$\tau = \frac{\sigma^2_t \Delta t}{2}$及扩散系数$\sigma_t$不随时间变化得到朗之万动力学公式:  
$$\mathbf{x}_{t+1}=\mathbf{x}_t+\tau \nabla_{\mathbf{x}} \log p\left(\mathbf{x}_t\right)+\sqrt{2 \tau} \mathbf{z}, \quad \mathbf{z} \sim \mathcal{N}(0, \mathbf{I})$$

因概率密度不随时间变化(动态平衡)，<span style="color: red;">粒子在不同概率区域停留的时间，正比于该区域的概率密度。</span>最终，我们记录下粒子运动的轨迹，这个轨迹就是目标分布`p(x)`的样本。


参考：
[Fast Sampling of Diffusion Models with Exponential Integrator](https://arxiv.org/pdf/2204.13902)