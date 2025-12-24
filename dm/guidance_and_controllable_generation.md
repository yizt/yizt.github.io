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
 [Conditioning Method for Denoising Diffusion Probabilistic Models)(http://arxiv.org/pdf/2108.02938)核心思想是利用一个低通滤波器将参考图像的结构信息作为条件，引导扩散模型的生成过程，而无需重新训练模型。如果生成的图像与参考图像在低频成分上一致，那么它们在整体结构上就会相似。高频细节则可以自由生成，实现多样性。
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
 [TRAINING-FREE LINEAR IMAGE INVERSES VIA FLOWS](http://arxiv.org/pdf/2310.04432)

高斯概率路径:
\[
q(\mathbf{x}_t|\mathbf{y}, \mathbf{x}_1) = q(\mathbf{x}_t|\mathbf{x}_1) = \mathcal{N}(\alpha_t \mathbf{x}_1, \sigma_t^2 \mathbf{I}) \tag 5
\]  
 其中$x_0为纯噪声,x_1为数据分布,\alpha_t=1-t,\sigma_t=t$。

测量值y定义如下:
\[
y = Ax_1 + \epsilon \quad //其中\epsilon \sim \mathcal{N}(0,\sigma_y^2\mathbf{I})  \tag 7
\]

### 得分引导到速度场引导
 由[An Introduction to Flow Matching and Diffusion Models](https://diffusion.csail.mit.edu/docs/lecture-notes.pdf)的公式54可知速度场和得分存在如下关系
$$
\begin{equation*}
u_t(x) =\beta_t^2  \frac {d ln(\frac {\alpha_t} {\beta_t})} {dt}   s_t (x) + \frac {d ln(\alpha_t)} {dt} x  
\end{equation*}
$$  
将变量表示替换为本文的有:  
$$
\begin{equation*}
v(\mathbf{x}_t) =\sigma_t^2  \frac {d ln(\frac {\alpha_t} {\sigma_t})} {dt}   \nabla_{\mathbf{x}_t} \ln q(\mathbf{x_t}) + \frac {d ln(\alpha_t)} {dt} \mathbf{x}_t  
\end{equation*}
$$
响应的条件速度场和条件得分的关系如下:
$$
\begin{equation*}
v(\mathbf{x}_t,\mathbf{y}) =\sigma_t^2  \frac {d ln(\frac {\alpha_t} {\sigma_t})} {dt}   \nabla_{\mathbf{x}_t} \ln q(\mathbf{x_t}|\mathbf{y}) + \frac {d ln(\alpha_t)} {dt} \mathbf{x}_t  
\end{equation*}
$$

两式相减有:
\[
v(\mathbf{x}_t, \mathbf{y}) - v(\mathbf{x}_t) = \sigma_t^2 \frac{d \ln(\alpha_t/\sigma_t)}{dt} \left(\nabla_{\mathbf{x}_t} \ln q(\mathbf{x_t}|\mathbf{y})-\nabla_{\mathbf{x}_t} \ln q(\mathbf{x_t}) \right) \\
\Leftrightarrow v(\mathbf{x}_t, \mathbf{y}) - v(\mathbf{x}_t) =  \sigma_t^2 \frac{d \ln(\alpha_t/\sigma_t)}{dt} \nabla_{\mathbf{x}_t} \ln q(\mathbf{y}|\mathbf{x}_t) \quad // 加恒为0的\nabla_{\mathbf{x}_t}\ln q(\mathbf{y}) \\
\Leftrightarrow v(\mathbf{x}_t, \mathbf{y}) = v(\mathbf{x}_t) + \sigma_t^2 \frac{d \ln(\alpha_t/\sigma_t)}{dt} \nabla_{\mathbf{x}_t} \ln q(\mathbf{y}|\mathbf{x}_t)  \tag 11
\]

使用ΠGDM伪逆近似后有:  
\[
\widehat{\boldsymbol{v}}(\boldsymbol{x}_t, \boldsymbol{y}) = \widehat{\boldsymbol{v}}(\boldsymbol{x}_t) + \sigma_t^2 \frac{d \ln(\alpha_t/\sigma_t)}{dt} \gamma_t \nabla_{\boldsymbol{x}_t} \ln q^{app}(\boldsymbol{y}|\boldsymbol{x}_t) \tag 12
\]
其中$\gamma_t$​ 是一个权重因子，用于调整公式(12)中引导项的强度。  

### 后验分布 \(q(x_1|x_t)\) 推导
 推导基于一个标准的**贝叶斯推断**过程：已知**先验分布** \(q(x_1)\) 和**似然函数** \(q(x_t|x_1)\)，求解**后验分布** \(q(x_1|x_t)\) 的方差参数 \(r_t^2\)。其核心原因是**高斯分布的共轭性质**。

#### 推导步骤与假设
推导基于以下三个关键假设：
1.  **数据先验分布**：\(q(x_1) = \mathcal{N}(x_1; 0, I)\)。这是推导的**起点**，也是得出简洁解析解 \(r_t^2 = \frac{\sigma_t^2}{\sigma_t^2 + \alpha_t^2}\) 的原因。
2.  **扩散前向过程**：\(q(x_t|x_1) = \mathcal{N}(x_t; \alpha_t x_1, \sigma_t^2 I)\)。这是扩散模型的标准参数化形式，其中 \(\alpha_t\) 和 \(\sigma_t\) 是已知的时间依赖函数。
3.  **后验分布形式**：根据高斯共轭性，后验 \(q(x_1|x_t)\) 也是高斯分布，我们将其参数化为 \(\mathcal{N}(x_1; \widehat{x}_1(x_t), r_t^2 I)\)。其中 \(\widehat{x}_1(x_t)\) 是后验均值，\(r_t^2\) 是待求的后验方差标量。

下面是详细的推导过程：

$$
\begin{aligned}
& \text{目标：求后验分布 } q(x_1|x_t) \propto q(x_t|x_1) \cdot q(x_1) \\
\\
& 1. \text{ 写出先验与似然的概率密度函数（忽略常数项）} \\
& \log q(x_1) = -\frac{1}{2} x_1^\top x_1 + \text{const}. \quad // \text{标准高斯先验} \\
& \log q(x_t|x_1) = -\frac{1}{2\sigma_t^2} (x_t - \alpha_t x_1)^\top (x_t - \alpha_t x_1) + \text{const}. \\
\\
& 2. \text{ 相加得到非归一化的对数后验} \\
& \log [q(x_t|x_1) q(x_1)] = -\frac{1}{2} \left[ \frac{1}{\sigma_t^2} (x_t - \alpha_t x_1)^\top (x_t - \alpha_t x_1) + x_1^\top x_1 \right] + \text{const} \\
\\
& 3. \text{ 展开并合并关于 } x_1 \text{ 的二次项和一次项} \\
& \text{展开：} (x_t - \alpha_t x_1)^\top (x_t - \alpha_t x_1) = x_t^\top x_t - 2\alpha_t x_t^\top x_1 + \alpha_t^2 x_1^\top x_1 \\
& \text{代入并合并：} \\
& \log [\cdot] = -\frac{1}{2} \left[ \frac{\alpha_t^2}{\sigma_t^2} x_1^\top x_1 + x_1^\top x_1 - \frac{2\alpha_t}{\sigma_t^2} x_t^\top x_1 + \frac{1}{\sigma_t^2} x_t^\top x_t \right] + \text{const} \\
& = -\frac{1}{2} \left[ \underbrace{\left( \frac{\alpha_t^2}{\sigma_t^2} + 1 \right)}_{\text{二次项系数}} x_1^\top x_1 - \underbrace{\frac{2\alpha_t}{\sigma_t^2} x_t^\top}_{\text{一次项系数}} x_1 \right] + \text{（与 } x_1 \text{ 无关的项）} \\
\\
& 4. \text{ 与高斯分布的标准形式对比，提取后验方差和均值} \\
& \text{高斯对数密度标准形式：} \log \mathcal{N}(x_1; \mu, \Sigma) = -\frac{1}{2} (x_1 - \mu)^\top \Sigma^{-1} (x_1 - \mu) + \text{const} \\
& = -\frac{1}{2} x_1^\top \Sigma^{-1} x_1 + x_1^\top \Sigma^{-1} \mu + \text{const}. \\
\\
& \text{对比系数：} \\
& \text{二次项系数：} \Sigma^{-1} = \left( \frac{\alpha_t^2}{\sigma_t^2} + 1 \right) I = \frac{\sigma_t^2 + \alpha_t^2}{\sigma_t^2} I \\
& \Rightarrow \quad \Sigma = \frac{\sigma_t^2}{\sigma_t^2 + \alpha_t^2} I \\
& \text{一次项系数：} \Sigma^{-1} \mu = \frac{\alpha_t}{\sigma_t^2} x_t \\
& \Rightarrow \quad \mu = \Sigma \cdot \frac{\alpha_t}{\sigma_t^2} x_t = \frac{\sigma_t^2}{\sigma_t^2 + \alpha_t^2} \cdot \frac{\alpha_t}{\sigma_t^2} x_t = \frac{\alpha_t}{\sigma_t^2 + \alpha_t^2} x_t \\
\\
& 5. \text{ 得到结果} \\
& \text{后验方差：} \quad r_t^2 = \frac{\sigma_t^2}{\sigma_t^2 + \alpha_t^2} \quad // \text{即公式 (13)} \\
& \text{后验均值：} \quad \widehat{x}_1(x_t) = \frac{\alpha_t}{\sigma_t^2 + \alpha_t^2} x_t
\end{aligned}
$$

### 参考

[A Survey on Diffusion Models for Inverse Problems](https://arxiv.org/pdf/2410.00083v1)