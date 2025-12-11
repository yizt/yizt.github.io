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
对于如下形式微分方程:
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
操作步骤:
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


## ODE数值方法


### Adams-Bashforth
  Adams-Bashforth公式的推导基于**拉格朗日插值多项式**和**积分近似思想**，其核心是通过前k步的函数值构造插值多项式，近似被积函数，进而推导出递推公式。以下以**二阶（AB2）和四阶（AB4）方法**为例，详细拆解推导过程：

#### **1. 基础思想：从常微分方程到积分方程**
对于初值问题：
\[
y'(t) = f(t, y(t)), \quad y(t_0) = y_0
\]
可转化为积分形式：
\[
y(t_{n+1}) = y(t_n) + \int_{t_n}^{t_{n+1}} f(t, y(t)) \, dt
\]
Adams-Bashforth方法通过**数值积分**近似右侧积分项，而积分核\(f(t, y(t))\)由**前k步的函数值**通过拉格朗日插值多项式近似。

#### **2. 拉格朗日插值多项式的构造**
假设已知前\(k\)步的函数值\(f(t_{n-i}, y_{n-i})\)（\(i=0,1,\dots,k-1\)），在区间\([t_n, t_{n+1}]\)上构造\(k-1\)次拉格朗日插值多项式\(P_{k-1}(t)\)，逼近\(f(t, y(t))\)：
\[
P_{k-1}(t) = \sum_{j=0}^{k-1} f(t_{n-j}, y_{n-j}) \cdot l_j(t)
\]
其中基函数\(l_j(t)\)满足：
\[
l_j(t) = \prod_{\substack{m=0 \\ m \neq j}}^{k-1} \frac{t - t_{n-m}}{t_{n-j} - t_{n-m}}
\]
基函数在对应节点\(t_{n-j}\)处取值为1，其他节点为0，确保插值多项式精确通过已知点。

#### **3. 积分近似与递推公式推导**
将积分近似为：  
$$
\begin{align*}
\int_{t_n}^{t_{n+1}} f(t, y(t)) \, dt &\approx \int_{t_n}^{t_{n+1}} P_{k-1}(t) \, dt \\
\Leftrightarrow y(t_{n+1}) & \approx y(t_n) + \int_{t_n}^{t_{n+1}} P_{k-1}(t) \, dt \\
& = y(t_n) + \int_{t_n}^{t_{n+1}}\sum_{j=0}^{k-1} \left[ f(t_{n-j}, y_{n-j}) \cdot  l_j(t)\right] \, dt  \quad //带入P_{k-1}(t)\\
& = y(t_n) + \sum_{j=0}^{k-1} \left[ f(t_{n-j}, y_{n-j}) \cdot \int_{t_n}^{t_{n+1}} l_j(t) \, dt \right] \quad //求和与积分交换顺序\\
&= y_n + \sum_{j=0}^{k-1} b_j f(t_{n-j}, y_{n-j}) \quad//令 b_j = \int_{t_n}^{t_{n+1}} l_j(t) \, dt
\end{align*}
$$


积分系数\(b_j = \int_{t_n}^{t_{n+1}} l_j(t) \, dt\)，则公式为：
$$
\begin{align*}
b_j &= \int_{t_n}^{t_{n+1}} l_j(t) \, dt \\
&= \int_{t_n}^{t_{n+1}} \prod_{\substack{m=0 \\ m \neq j}}^{k-1} \frac{t - t_{n-m}}{t_{n-j} - t_{n-m}} \, dt \\
&= \int_{t_n}^{t_{n+1}} \prod_{\substack{m=0 \\ m \neq j}}^{k-1} \frac{nh+\tau h - (n-m)h}{(n-j)h - (n-m)h} \, dt \quad //令t_i=i h, t=nh+\tau h ,h为均匀长\\ 
&= h\int_0^1 \prod_{\substack{m=0 \\ m \neq j}}^{k-1} \frac{m+\tau}{m-j} \, d\tau \quad //dt=hd\tau, \tau \in [0,1]\\
\end{align*}
$$


#### **4. 具体阶数公式推导**
**二阶方法（AB2，k=2）**
- **积分系数计算**：
  \[
  b_0 = h\int_0^1 \prod_{\substack{m=0 \\ m \neq j}}^{k-1} \frac{m+\tau}{m-j} \, d\tau = h\int_0^1 \frac{1+\tau}{1-0} \, d\tau = \frac {3h} {2} \quad //k=2,j=0
  \]
  \[
  b_1 = h\int_0^1 \prod_{\substack{m=0 \\ m \neq j}}^{k-1} \frac{m+\tau}{m-j} \, d\tau = h\int_0^1 \frac{0+\tau}{0-1} \, d\tau = -\frac {h} {2} \quad //k=2,j=1
  \]
- **递推公式**：
  \[
  y_{n+1} = y_n + \frac{h}{2} \left[ 3f(t_n, y_n) - f(t_{n-1}, y_{n-1}) \right]
  \]

**四阶方法（AB4，k=4）**
- **积分系数计算**： 
$$
\begin{align*}
b_0 &= h\int_0^1 \prod_{\substack{m=0 \\ m \neq j}}^{k-1} \frac{m+\tau}{m-j} \, d\tau \\
&= h\int_0^1 \frac{1+\tau}{1-0} \cdot \frac{2+\tau}{2-0} \cdot \frac{3+\tau}{3-0} \, d\tau \quad //k=4,j=0 \\
&= \frac {55} {24}h
\end{align*}
$$  

  类似的可以算出$b_1=-\frac {59} {24}h,b_2=\frac {37} {24}h,b_3=-\frac {9} {24}h$
- **递推公式**：
  \[
  y_{n+1} = y_n + \frac{h}{24} \left[ 55f(t_n, y_n) - 59f(t_{n-1}, y_{n-1}) + 37f(t_{n-2}, y_{n-2}) - 9f(t_{n-3}, y_{n-3}) \right]
  \]

#### **5. 关键特性**
- **系数确定**：积分系数\(b_j\)由基函数在区间\([t_n, t_{n+1}]\)上的积分确定，需确保多项式在节点处精确匹配，并通过代数运算简化系数。
- **阶数与步数关系**：s步Adams-Bashforth方法的阶数为s（局部截断误差为\(O(h^{s+1})\)），阶数与步数相等，这是通过插值多项式的次数（s-1次）和积分精度共同决定的。
- **显式特性**：由于插值多项式仅依赖前k步的函数值，递推公式无需解方程，直接由已知值计算下一步解，属于显式方法。

#### **6. 初始值问题与启动方法**
Adams-Bashforth方法非自启动，需前k步的初始值。通常使用同阶或更高阶的Runge-Kutta方法（如四阶Runge-Kutta）提供初始值。例如，四阶AB4需前4步的初始值，由RK4方法计算得到。

#### **7. 误差分析与稳定性**
- **局部截断误差**：s阶方法的局部截断误差为\(O(h^{s+1})\)，随阶数升高而降低。
- **稳定性区域**：显式方法稳定性区域较小，步长过大易发散，常与隐式Adams-Moulton方法组合为预测-校正系统以增强稳定性。

#### **总结**
Adams-Bashforth公式的推导本质是**拉格朗日插值多项式在数值积分中的应用**，通过构造前k步函数值的插值多项式，近似被积函数并积分得到递推关系。其阶数与步数相等，高阶方法精度高但稳定性弱，需结合问题特性选择阶数，并搭配初始值方法和稳定性增强策略。这一方法在工程和科学计算中广泛应用于高精度、光滑函数的常微分方程求解。