---
layout: default
title: 常微分方程
nav_order: 2
---

# 常微分方程


## 一阶微分方程

### 变量分离法
 一阶微分方程有如下形式：
\[
\frac{dy}{dx} = g(x)h(y)
\]
操作步骤: 
1. **分离变量**：将方程改写为所有 \(y\) 项与 \(dy\) 在一起，所有 \(x\) 项与 \(dx\) 在一起：
   \[
   \frac{1}{h(y)} \, dy = g(x) \, dx
   \]
   注意：若 \(h(y) = 0\) 有解 \(y = c\)，则 \(y = c\) 也是原方程的解。

2. **两边积分**：
   \[
   \int \frac{1}{h(y)} \, dy = \int g(x) \, dx + C
   \]
   其中 \(C\) 为任意常数。

3. **求解 \(y\)**：若积分可求出原函数，则得到隐式或显式通解。

### 变量替换法
通过变量代换将方程化为可分离或线性方程。常见类型包括：

**齐次方程**
形如 $\frac{dy}{dx} = f\left(\frac{y}{x}\right)$的微分方程。  
令 \(v = \frac{y}{x}\)，则 \(y = vx\)，\(\frac{dy}{dx} = v + x \frac{dv}{dx}\)。  
代入得：
\[
v + x \frac{dv}{dx} = f(v)
\]
分离变量：
\[
\frac{dv}{f(v) - v} = \frac{dx}{x}
\]
积分即可。

**线性组合** 
形如:\(\frac{dy}{dx} = f(ax+by+c)\)  
令 \(u = ax+by+c\)，则 \(\frac{du}{dx} = a + b \frac{dy}{dx} =a+bf(u)\)，代入后可分离变量。

### 常数变异法
对于如下形式微分方程
\[
\frac{dy}{dx} + P(x)y = Q(x)
\]
操作步骤: 
1. **解齐次方程**：
   \[
   \frac{dy}{dx} + P(x)y = 0
   \]
   通解为：
   \[
   y_h = C e^{-\int P(x) dx}
   \]

2. **常数变易**：将常数 \(C\) 替换为函数 \(C(x)\)，设特解：
   \[
   y_p = C(x) e^{-\int P(x) dx}
   \]
则
$$
\begin{equation*}
   \frac{dy_p}{dx} = C'(x) e^{-\int P(x) dx} - P(x) C(x) e^{-\int P(x) dx} 
\end{equation*}
$$  

3. **代入原方程**： 
$$
\begin{align*}
C'(x) e^{-\int P(x) dx} - P(x) C(x) e^{-\int P(x) dx} + P(x) C(x) e^{-\int P(x) dx} &= Q(x) \\
C'(x) e^{-\int P(x) dx} &= Q(x) \\
C'(x) &= Q(x) e^{\int P(x) dx} \\
C(x) &= \int Q(x) e^{\int P(x) dx} \, dx + C
\end{align*}
$$
4. **通解**：
   \[
   y = e^{-\int P(x) dx} \left( \int Q(x) e^{\int P(x) dx} \, dx + C \right)
   \]

### 积分因子法
适用于一阶线性方程：
\[
\frac{dy}{dx} + P(x)y = Q(x)
\]
操作步骤
1. **寻找积分因子 \(\mu(x)\)**，使得乘以 \(\mu(x)\) 后方程左边成为某个函数的导数：
$$
\begin{align*}
\mu(x) \frac{dy}{dx} + \mu(x) P(x) y &= \frac{d}{dx} \left[ \mu(x) y \right] \\
\mu(x) \frac{dy}{dx} + \mu(x) P(x) y &= \mu'(x) y + \mu(x) \frac{dy}{dx} \qquad //\text{右边导数展开} \\
\mu(x) P(x) &= \mu'(x) \\
\mu'(x) &= \mu(x) P(x)
\end{align*}
$$

2. **解出 \(\mu(x)\)**：
   \[
   \frac{d\mu}{\mu} = P(x) \, dx \quad \Rightarrow \quad \ln|\mu| = \int P(x) dx
   \]
   取：
   \[
   \mu(x) = e^{\int P(x) dx}
   \]

3. **乘以积分因子**：
$$
\begin{equation*}
\frac{d}{dx} \left[ \mu(x) y \right] = \mu(x) Q(x)
\end{equation*}
$$

4. **积分求解**：
   \[
   \mu(x) y = \int \mu(x) Q(x) \, dx + C
   \]
   \[
   y = \frac{1}{\mu(x)} \left( \int \mu(x) Q(x) \, dx + C \right)
   \]


### 总结
- **变量分离法**：适用于方程右侧可分离为 \(g(x)h(y)\) 的情况。
- **变量替换法**：通过代换（如齐次替换、线性组合、伯努利替换等）转化为可解形式。
- **常数变异法**：用于线性非齐次方程，将齐次解中的常数变为函数求特解。
- **积分因子法**：针对一阶线性方程，构造积分因子化为全微分形式。
