---
layout: default
title: 去噪扩散模型
nav_order: 2
---


# 去噪扩散模型


## DEIS
 [FAST SAMPLING OF DIFFUSION MODELS WITH EXPONENTIAL INTEGRATOR](https://arxiv.org/pdf/2204.13902)提出Diffusion Exponential Integrator Sampler (DEIS)利用指数积分器减少离散化误差，并结合多项式外推和优化的时间离散化策略，显著提高了扩散模型的采样速度和样本质量。


### 扩散模型回顾


根据你提供的文本内容，该图像描述了**扩散模型中的反向去噪过程**，并给出了对应的**反向时间随机微分方程（Reverse-time SDE）**。以下是内容的识别与说明：

对于公式1定义的前向加噪过程
\[
dx = F_t x dt + G_t \, dw \tag 1
\]

 其对应去燥反向过程的**随机微分方程（SDE）公式（2）如下:。
\[
dx = \left[ F_t x - G_t G_t^T \nabla \log p_t(x) \right] dt + G_t \, dw \tag 2
\]

其中：
- \( F_i x \) 是**前向过程的漂移项**
- \( G_i G_i^T \) 是**扩散系数的平方**
- \( \nabla \log p_t(x) \) 是**得分函数**（score function）




#### **Table 1: 两种常见的SDE（VPSDE与VESDE）**

| SDE 类型 | \(F_t\) | \(G_t\) | \(\mu_t\) | \(\Sigma_t\) |
|:---:|:---:|:---:|:---:|:---:|
| **VPSDE** | \(\frac{1}{2} \frac{d \log \alpha_t}{dt} I\) | \(\sqrt{-\frac{d \log \alpha_t}{dt}} I\) | \(\sqrt{\alpha_t} I\) | \((1 - \alpha_t) I\) |
| **VESDE** | \(0\) | \(\sqrt{\frac{d [\sigma_t^2]}{dt}} I\) | \(I\) | \(\sigma_t^2 I\) |


- 参数 \(\alpha_t\) 随时间递减，且 \(\alpha_0 \approx 1\)，\(\alpha_T \approx 0\)。
- 参数 \(\sigma_t\) 随时间递增。

