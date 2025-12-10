---
layout: default
title: 去噪扩散模型
nav_order: 2
---


# 去噪扩散模型


## DEIS
 [FAST SAMPLING OF DIFFUSION MODELS WITH EXPONENTIAL INTEGRATOR](https://arxiv.org/pdf/2204.13902)提出Diffusion Exponential Integrator Sampler (DEIS)利用指数积分器减少离散化误差，并结合多项式外推和优化的时间离散化策略，显著提高了扩散模型的采样速度和样本质量。


### 扩散模型回顾

#### **Table 1: 两种常见的SDE（VPSDE与VESDE）**

| SDE 类型（作者） | \(F_t\) | \(G_t\) | \(\mu_t\) | \(\Sigma_t\) |
|---|---|---|---|---|
| **VPSDE** (Ho et al., 2020) | \(\frac{1}{2} \frac{d \log \alpha_t}{dt} I\) | \(\sqrt{-\frac{d \log \alpha_t}{dt}} I\) | \(\sqrt{\alpha_t} I\) | \((1 - \alpha_t) I\) |
| **VESDE** (Song et al., 2020b) | \(0\) | \(\sqrt{\frac{d [\sigma_t^2]}{dt}} I\) | \(I\) | \(\sigma_t^2 I\) |

---

#### **表格下方说明：**
- 参数 \(\alpha_t\) 随时间递减，且 \(\alpha_0 \approx 1\)，\(\alpha_T \approx 0\)。
- 参数 \(\sigma_t\) 随时间递增。

---

### 📌 **总结说明：**
- 该表格常见于**扩散模型（Diffusion Models）** 或**生成建模**相关论文中，用于描述**前向扩散过程**的两种不同SDE形式。
- **VPSDE**（如DDPM中所用）强调**方差保持**，\(\alpha_t\) 控制信号衰减。
- **VESDE**（如Score-based SDE中所用）允许**方差随时间增长**，\(\sigma_t\) 控制噪声幅度。

如果需要进一步解释这些参数的意义或它们在扩散模型中的作用，我可以为你补充说明。