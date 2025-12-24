---
layout: default
title: 引导和可控生成
nav_order: 5
---

# 引导和可控生成

## 分类
### Classifier Guidance

### Classifier-Free Guidance

### Training-Free Guidance

\[
\nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t | \mathbf{y}) = \nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t) + \nabla_{\mathbf{x}_t} \log p_t(\mathbf{y} | \mathbf{x}_t)
\]

## Score ALD
 []()

## ILVR
 [Conditioning Method for Denoising Diffusion Probabilistic Models)(http://arxiv.org/abs/2108.02938)核心思想是利用一个低通滤波器将参考图像的结构信息作为条件，引导扩散模型的生成过程，而无需重新训练模型。如果生成的图像与参考图像在低频成分上一致，那么它们在整体结构上就会相似。高频细节则可以自由生成，实现多样性。
## RED-Diff
 [A VARIATIONAL PERSPECTIVE ON SOLVING INVERSE PROBLEMS WITH DIFFUSION MODELS](https://arxiv.org/pdf/2305.04391) 暂时没有看懂


## Denoising Diffusion Null Models (DDNM)

## Denoising Diffusion Restoration Models (DDRM)

## ΠGDM
 [Pseudoinverse-guided diffusion models for inverse problems](https://jankautz.com/publications/PiGDM_ICLR23.pdf)  

 假设我们拥有对某个信号 \( x_0 \in \mathbb{R}^n \) 的测量值 \( y \in \mathbb{R}^m \)，其关系为  
\[y = Hx_0 + z, \tag{2}\]

其中 \( H \in \mathbb{R}^{n \times m} \) 是已知的测量矩阵（模型），\( z \sim \mathcal{N}(0, \sigma_y^2 I) \) 是一个独立同分布的高斯噪声向量，其各维度上的标准差 \( \sigma_y \) 已知。  
 我们的目标是求解这个逆问题，从测量值 \( y \) 中恢复出原始信号 \( x_0 \in \mathbb{R}^n \)。


$p_t(\mathbf{x}_0\vert\mathbf{x}_t)$使用高斯近似(<span style='color:red'>ΠGDM的关键近似</span>):  
\[
p_t(\mathbf{x}_0|\mathbf{x}_t) \approx \mathcal{N}(\hat{\mathbf{x}}_t, r_t^2 \mathbf{I}), \tag 4
\]
均值使用Tweedie公式估计
\[
\hat{\mathbf{x}}_t = \mathbb{E}[\mathbf{x}_0 | \mathbf{x}_t] = \mathbf{x}_t + \sigma_t^2 \nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t) \approx \mathbf{x}_t + \sigma_t^2 S_\theta(\mathbf{x}; \sigma_t).  \tag 5
\]  
 其中$p_t(\mathbf{x}_t \vert \mathbf{x}_0) \sim \mathcal{N}({\mathbf{x}}_0, \sigma_t^2 \mathbf{I})$,注意:<span style='color:red'>$r_t$与$\sigma_t$不同</span>  
 显然$p_t(\mathbf{y} \vert \mathbf{x}_t)$也是高斯分布:  
\[
p_t(\mathbf{y} | \mathbf{x}_t) \approx \mathcal{N}(\mathbf{H}\hat{\mathbf{x}}_t, r_t^2 \mathbf{H}\mathbf{H}^\top + \sigma_y^2 \mathbf{I}).
\]  
 可以推导出$p_t(\mathbf{y} \vert \mathbf{x}_t)$的得分:  
$$
\begin{aligned}
\log p_t(\mathbf{y}|\mathbf{x}_t) &= -\frac{1}{2} (\mathbf{y} - H\hat{\mathbf{x}}_t)^\top \Sigma_t^{-1} (\mathbf{y} - H\hat{\mathbf{x}}_t) + \text{const} \quad // \text{取对数，忽略常数项} \\
\nabla_{\mathbf{x}_t} \log p_t(\mathbf{y}|\mathbf{x}_t) &= -\frac{1}{2} \nabla_{\mathbf{x}_t} ((\mathbf{y} - H\hat{\mathbf{x}}_t )^\top \Sigma_t^{-1} (\mathbf{y} - H\hat{\mathbf{x}}_t )) \quad // \text{记协方差为}\Sigma_t \\
&= - (\frac{\partial {(\mathbf{y} - H\hat{\mathbf{x}}_t})}{\partial \mathbf{x}_t} )^\top \Sigma_t^{-1} ({\mathbf{y} - H\hat{\mathbf{x}}_t}) \\
&= (H \frac{\partial \hat{\mathbf{x}}_t}{\partial \mathbf{x}_t})^\top \Sigma_t^{-1} ({\mathbf{y} - H\hat{\mathbf{x}}_t}) \quad //链式法则 \\
&= (H \frac{\partial \hat{\mathbf{x}}_t}{\partial \mathbf{x}_t})^\top (r_t^2 HH^\top + \sigma_y^2 I)^{-1} ({\mathbf{y} - H\hat{\mathbf{x}}_t}) \quad //带入\Sigma_t \\
&= \left(\underbrace{(\mathbf{y} - H\mathbf{\hat{x}}_t)^\top (r_t^2 HH^\top + \sigma_y^2 I)^{-1} H}_{\text{vector}} \underbrace{\left(\frac{\partial \mathbf{\hat{x}}_t}{\partial \mathbf{x}_t}\right)}_{\text{Jacobian}}\right)^\top \quad //\Sigma_t是对称矩阵  \tag 7
\end{aligned}
$$  

当$\sigma_y=0$时(无噪声测量),公式7可以近似为:  

$$
\begin{aligned}
\nabla_{\mathbf{x}_t} \log p_t(\mathbf{y}|\mathbf{x}_t) &\approx \left(\underbrace{(\mathbf{y} - H\mathbf{\hat{x}}_t)^\top (r_t^2 HH^\top)^{-1} H}_{\text{vector}} \underbrace{\left(\frac{\partial \mathbf{\hat{x}}_t}{\partial \mathbf{x}_t}\right)}_{\text{Jacobian}}\right)^\top\\ 
&= r_t^{-2} \left(\underbrace{(\mathbf{y} - H\mathbf{\hat{x}}_t)^\top (HH^\top)^{-1} H}_{\text{vector}} \underbrace{\left(\frac{\partial \mathbf{\hat{x}}_t}{\partial \mathbf{x}_t}\right)}_{\text{Jacobian}}\right)^\top\\
&= r_t^{-2} \left(\underbrace{(\mathbf{y} - H\mathbf{\hat{x}}_t)^\top (H^\dagger)^\top}_{\text{vector}} \underbrace{\left(\frac{\partial \mathbf{\hat{x}}_t}{\partial \mathbf{x}_t}\right)}_{\text{Jacobian}}\right)^\top \quad //H^\dagger = H^\top(HH^\top)^{-1}\\
&=r_t^{-2} \left(\left(\mathbf{H}^\dagger \mathbf{y} - \mathbf{H}^\dagger \mathbf{H} \mathbf{\hat{x}}_t\right)^\top \frac{\partial \mathbf{\hat{x}}_t}{\partial \mathbf{x}_t}\right)^\top  \tag 8
\end{aligned}
$$  

 其中: $H^\dagger = H^\top(HH^\top)^{-1}$ 是矩阵H的Moore-Penrose伪逆。

## DPS
 [Diffusion posterior sampling for general noisy inverse problems](https://arxiv.org/pdf/2209.14687)

## CCDF
 [Come-Closer-Diffuse-Faster: Accelerating Conditional Diffusion Models for Inverse Problems through Stochastic Contraction](https://arxiv.org/pdf/2112.05146)

## TRAINING-FREE LINEAR IMAGE INVERSES VIA FLOWS
 [TRAINING-FREE LINEAR IMAGE INVERSES VIA FLOWS](http://arxiv.org/abs/2310.04432)

### 参考

[A Survey on Diffusion Models for Inverse Problems](https://arxiv.org/abs/2410.00083v1)