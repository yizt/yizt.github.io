---
layout: default
title: 基于得分的扩散模型
nav_order: 3
---


# 基于得分的扩散模型

## Score Matching
  [Estimation of Non-Normalized Statistical Models by Score Matching](https://jmlr.org/papers/volume6/hyvarinen05a/hyvarinen05a.pdf)提出了一种用于估计非归一化统计模型的新方法——分数匹配。该方法巧妙地避免了计算模型中难以处理的归一化常数，为后续的扩散模型等研究奠定了重要基础。

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

### Score Matching-采样


参考：
[Fast Sampling of Diffusion Models with Exponential Integrator](https://arxiv.org/pdf/2204.13902)