---  
title: Diffusion Models  
share: "true"  
---  
 %% TODO  
2023 年扩散模型还有什么可做的方向？ - 谷粒多·凯特的回答 - 知乎 https://www.zhihu.com/question/568791838/answer/3195773725 %%  
# 1 原理篇   
## 1.1 DDPM  
### 1.1.1 前向扩散  
前向扩散指的是将一个复杂分布转换成简单分布的过程$\mathcal{T}:\mathbb{R}^d\mapsto\mathbb{R}^d$，即：  
$$  
\mathbf{x}_0\sim p_\mathrm{complex}\Longrightarrow \mathcal{T}(\mathbf{x}_0)\sim p_\mathrm{prior}  
$$  
在DDPM中，将这个过程定义为**马尔科夫链**，通过不断地向复杂分布中的样本$x_0\sim p_\mathrm{complex}$添加高斯噪声。这个加噪过程可以表示为$q(\mathbf{x}_t|\mathbf{x}_{t-1})$：  
$$  
\begin{align}  
q(\mathbf{x}_t \vert \mathbf{x}_{t-1}) &= \mathcal{N}(\mathbf{x}_t; \sqrt{1 - \beta_t} \mathbf{x}_{t-1}, \beta_t\mathbf{I})\\  
\mathbf{x}_t&=\sqrt{1-\beta_t}\mathbf{x}_{t-1}+\sqrt{\beta_t}\boldsymbol\epsilon \quad \boldsymbol\epsilon\sim\mathcal{N}(\mathbf{0},\mathbf{I})  
\end{align}  
$$  
其中，$\{\beta_t\in(0,1)\}^T_{t=1}$，是超参数。  
从$\mathbf{x}_0$开始，不断地应用$q(\mathbf{x}_t|\mathbf{x}_{t-1})$，经过足够大的$T$步加噪之后，最终得到纯噪声$\mathbf{x}_T$：  
$$  
\mathbf{x}_0\sim p_\mathrm{complex}\rightarrow \mathbf{x}_1\rightarrow \cdots \mathbf{x}_t\rightarrow\cdots\rightarrow \mathbf{x}_T\sim p_\mathrm{prior}  
$$  
除了迭代地使用$q(\mathbf{x}_t|\mathbf{x}_{t-1})$外，还可以使用$q(\mathbf{x}_t|\mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t} \mathbf{x}_0, (1 - \bar{\alpha}_t)\mathbf{I})$一步到位，证明如下（两个高斯变量的线性组合仍然是高斯变量）：  
$$  
\begin{aligned}  
\mathbf{x}_t   
&= \sqrt{\alpha_t}\mathbf{x}_{t-1} + \sqrt{1 - \alpha_t}\boldsymbol{\epsilon}_{t-1} &\quad ;\alpha_t=1-\alpha_t\\  
&= \sqrt{\alpha_t \alpha_{t-1}} \mathbf{x}_{t-2} + \sqrt{1 - \alpha_t \alpha_{t-1}} \bar{\boldsymbol{\epsilon}}_{t-2}  \\  
&= \dots \\  
&= \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon} &\quad ;\boldsymbol{\epsilon}\sim \mathcal{N}(\mathbf{0}, \mathbf{I}),\bar{\alpha}_t=\prod_{i=1}^t \alpha_i\  
\end{aligned}  
$$  
一般来说，超参数$\beta_t$的设置满足$0<\beta_1<\cdots<\beta_T<1$，则$\bar{\alpha}_1 > \cdots > \bar{\alpha}_T\to1$，则$\mathbf{x}_T$会只保留纯噪声部分。  
### 1.1.2 逆向扩散  
在前向扩散过程中，实现了：  
$$  
\mathbf{x}_0\sim p_\mathrm{complex}\rightarrow \mathbf{x}_1\rightarrow \cdots \mathbf{x}_t\rightarrow\cdots\rightarrow \mathbf{x}_T\sim p_\mathrm{prior}  
$$  
如果能够实现将前向扩散过程反转，也就实现了从简单分布到复杂分布的映射。逆向扩散过程则是将前向过程反转，实现从简单分布随机采样样本，迭代地使用$q(\mathbf{x}_{t-1}|\mathbf{x}_t)$，最终生成复杂分布的样本，即：  
$$  
\mathbf{x}_T\sim p_\mathrm{prior}\rightarrow \mathbf{x}_{T-1}\rightarrow \cdots \mathbf{x}_t\rightarrow\cdots\rightarrow \mathbf{x}_0\sim p_\mathrm{complex}  
$$  
为了求取$q(\mathbf{x}_{t-1}|\mathbf{x}_t)$，使用贝叶斯公式：  
$$  
\begin{align}  
q(\mathbf{x}_{t-1}|\mathbf{x}_t)&=\frac{q(\mathbf{x}_t|\mathbf{x}_{t-1})q(\mathbf{x}_{t-1})}{q(\mathbf{x}_t)}  
\end{align}  
$$  
然而，公式中$q(x_{t-1})$和$q(x_t)$不好求，根据DDPM的马尔科夫假设，可以为$q(\mathbf{x}_{t-1}|\mathbf{x}_t)$添加条件（可以证明，如果向扩散过程中的$\beta_t$足够小，那么$q(\mathbf{x}_{t-1}|\mathbf{x}_t)$是高斯分布。）：  
$$  
\begin{align}  
q(\mathbf{x}_{t-1}|\mathbf{x}_t)&=q(\mathbf{x}_{t-1}|\mathbf{x}_t,\mathbf{x}_0)\\  
              &=\frac{q(\mathbf{x}_t|\mathbf{x}_{t-1},\mathbf{x}_0)q(\mathbf{x}_{t-1}|\mathbf{x}_0)}{q(\mathbf{x}_t|\mathbf{x}_0)}\\  
              &=\frac{q(\mathbf{x}_t|\mathbf{x}_{t-1})q(\mathbf{x}_{t-1}|\mathbf{x}_0)}{q(\mathbf{x}_t|\mathbf{x}_0)}\\  
              &=\mathcal{N}(\mathbf{x}_{t-1};\mu(\mathbf{x}_t;\theta),\sigma_t^2I)  
\end{align}  
$$  
其中，$\mu(x_t;\theta)$是高斯分布的均值，$\sigma_t$可以用超参数表示：  
$$  
\begin{align}  
\mu(\mathbf{x}_t;\theta)&=\frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}\mathbf{x}_t+  
\frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1-\bar{\alpha}_t}\mathbf{x}_0\\  
\sigma_t&=\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\cdot\beta_t  
\end{align}  
$$  
式中$x_0$可以反用公式$\mathbf x_t=\sqrt{\bar{\alpha}_t}\mathbf x_0+\sqrt{1-\bar{\alpha}_t}\boldsymbol\epsilon_t$：  
$$  
\mathbf x_0=\frac{1}{\sqrt{\bar{\alpha}_t}}\left(\mathbf{x}_t-\sqrt{1-\bar{\alpha}_t}\boldsymbol\epsilon_t\right)  
$$  
则：  
$$  
\begin{align}  
\mu(\mathbf{x}_t;\theta)&=\frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}\mathbf{x}_t+  
\frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1-\bar{\alpha}_t}\mathbf{x}_0\\  
&=\frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}\mathbf{x}_t+  
\frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1-\bar{\alpha}_t}\frac{1}{\sqrt{\bar{\alpha}_t}}\left(\mathbf{x}_t-\sqrt{1-\bar{\alpha}_t}\boldsymbol\epsilon_t\right)\\  
&=\frac{1}{\sqrt{\alpha_t}}\left(x_t-\frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}\boldsymbol\epsilon_t\right)  
\end{align}  
$$  
综上，逆向扩散过程：  
$$  
\begin{align}  
q(\mathbf{x}_{t-1}|\mathbf{x}_t)&=\mathcal{N}(\mathbf{x}_{t-1};\mu(\mathbf{x}_t;\theta),\sigma_t^2\mathbf I)\\  
&=\mathcal{N}\left(\mathbf x_{t-1};\frac{1}{\sqrt{\alpha_t}}\left(x_t-\frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}\boldsymbol\epsilon_t\right),\left(\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\cdot\beta_t\right)^2\mathbf I\right)\\  
\mathbf x_{t-1}&=\frac{1}{\sqrt{\alpha_t}}\left(x_t-\frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}\boldsymbol\epsilon_t\right)+\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t\cdot\boldsymbol\epsilon\quad\boldsymbol\epsilon\sim\mathcal N(\mathbf 0, \mathbf I)  
\end{align}  
$$  
## 1.2 基于score的生成模型  
基于score的生成模型和扩散模型非常相似，使用了score matching和Langevin dynamics技术进行生成。其中，  
1. score matching是估计目标分布的概率密度的梯度（即score，分数），记$p(x)$是数据分布的概率密度函数，则这个分布的score被定义为$\nabla_x\log p(x)$，score matching则是训练一个网络$s_\theta$去近似score：  
$$\mathcal{E}_{p(x)}\left[ \|\nabla_x\log p(x)-s_\theta(x)\|^2_2 \right]=\int p(x)\|\nabla_x\log p(x)-s_\theta(x)\|^2_2 dx$$  
3. Langevin dynamics是使用score采样生成数据，采样方式如下：  
$$  
x_t=x_{t-1}+\frac{\delta}{2}\nabla_x\log p(x_{t-1})+\sqrt{\delta}\epsilon, \text{    where } \epsilon\sim\mathcal{N}(0, I)  
$$  
## 1.3 DDIM  
#### 1.3.1.1 Review of DDPM  
1. Diffusion阶段  
$$  
\begin{align}  
q(x_t|x_0)&=\boxed{\mathcal{N}(x_t;\sqrt{\bar{\alpha}_t}x_0,(1-\bar{\alpha}_t)I)}\\  
         &=\sqrt{\bar{\alpha}_t}x_0+\sqrt{1-\bar{\alpha}_t}\epsilon  
         \text{ ,where } \epsilon\sim\mathcal{N}(0,I)  
\end{align}  
$$  
  
2. Reverse阶段  
使用贝叶斯公式  
$$  
\begin{align}  
q(x_{t-1}|x_t)&=\frac{q(x_t|x_{t-1})q(x_{t-1})}{q(x_t)}  
\end{align}  
$$  
发现公式中$q(x_{t-1})$和$q(x_t)$不好求，根据DDPM的马尔科夫假设：  
$$  
\begin{align}  
q(x_{t-1}|x_t)&=q(x_{t-1}|x_t,x_0)\\  
              &=\frac{q(x_t|x_{t-1},x_0)q(x_{t-1}|x_0)}{q(x_t|x_0)}\\  
              &=\frac{q(x_t|x_{t-1})q(x_{t-1}|x_0)}{q(x_t|x_0)}\\  
              &=\boxed{\mathcal{N}(x_{t-1};\mu(x_t;\theta),\sigma_t^2I)}  
\end{align}  
$$  
其中，$\sigma_t$可以用超参数表示，$\mu(x_t;\theta)$是一个神经网络，用于预测均值：  
$$  
\begin{align}  
\mu(x_t;\theta)&=\frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}x_t+  
\frac{\sqrt{\bar{x}_{t-1}}\beta_t}{1-\bar{\alpha}_t}x_0\\  
&=\boxed{\frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}x_t+  
\frac{\sqrt{\bar{x}_{t-1}}\beta_t}{1-\bar{\alpha}_t}\hat{x}_{0|t}}\\  
\sigma_t^2&=\boxed{\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\cdot\beta_t}  
\end{align}  
$$  
### 1.3.2 From DDPM to DDIM  
  
同样是对分布$q(x_{t-1}|x_t,x_0)$进行求解：  
$$  
\begin{align}  
q(x_{t-1}|x_t,x_0)  
&=\sqrt{\bar{\alpha}_{t-1}}x_0+\sqrt{1-\bar{\alpha}_{t-1}}\epsilon\\  
&=\sqrt{\bar{\alpha}_{t-1}}  
\hat{x}_{0|t}  
+\sqrt{1-\bar{\alpha}_{t-1}}\epsilon  
\text{ ,where }\epsilon\sim\mathcal{N}(0,I)  
\end{align}  
$$  
在上式中，$\epsilon$是一个噪声，虽然可以重新从高斯分布采样，但是也可以使用噪声估计网络估计出来的结果$\epsilon_\theta(x_t,t)$：  
$$  
\begin{align}  
q(x_{t-1}|x_t,x_0)  
&=\sqrt{\bar{\alpha}_{t-1}}x_0+\sqrt{1-\bar{\alpha}_{t-1}}\epsilon\\  
&=\sqrt{\bar{\alpha}_{t-1}}  
\hat{x}_{0|t}  
+\sqrt{1-\bar{\alpha}_{t-1}}\epsilon  
\text{ ,where }\epsilon\sim\mathcal{N}(0,I)\\  
&=\sqrt{\bar{\alpha}_{t-1}}  
\hat{x}_{0|t}  
+\sqrt{1-\bar{\alpha}_{t-1}}\epsilon_\theta(x_t,t)\\  
\end{align}  
$$  
甚至可以同时考虑$\epsilon$和$\epsilon_\theta(x_t,t)$：  
$$  
\begin{align}  
q(x_{t-1}|x_t,x_0)  
&=\sqrt{\bar{\alpha}_{t-1}}x_0+\sqrt{1-\bar{\alpha}_{t-1}}\epsilon\\  
&=\sqrt{\bar{\alpha}_{t-1}}  
\hat{x}_{0|t}  
+\sqrt{1-\bar{\alpha}_{t-1}}\epsilon  
\text{ ,where }\epsilon\sim\mathcal{N}(0,I)\\  
&=\sqrt{\bar{\alpha}_{t-1}}  
\hat{x}_{0|t}  
+\sqrt{1-\bar{\alpha}_{t-1}}\epsilon_\theta(x_t,t)\\  
\end{align}  
$$  
# 2 应用篇  
## 2.1 SR3  
超分，训练数据是LR和SR配对的图片，以LR图片作为condition，生成SR图片  
## 2.2 CDM  
超分，级联的方式对小图进行超分，采用的方法就是SR3  
![725](../assets/img/Pasted%20image%2020230927200225.png)  
## 2.3 SDEdit  
![900](../assets/img/Pasted%20image%2020230927200246.png)  
由于加噪过程是首先破坏高频信息，然后才破坏低频信息，所以加噪到一定程度之后，就就可以去掉不想要的细节纹理，但仍保留大体结构，于是生成出来的图像就既能遵循输入的引导，又显得真实。但是需要 realism-faithfulness trade-off  
## 2.4 ILVR  
给定一个参考图像$y$，通过调整DDPM去噪过程，希望让模型生成的图像接近参考图像，作者定义的接近是让模型能够满足  
$$  
\phi_N(x_t)=\phi_N(y_t)  
$$  
$\phi_N(\cdot)$是一个低通滤波器（下采样之后再插值回来）。使用如下的算法：  
![450](../assets/img/Pasted%20image%2020230927201110.png)  
即，对DDPM预测的$x'_{t-1}$加上bias：$\phi_N(y_{t-1})-\phi_N(x'_{t-1})$，可以证明，如果上/下采样采用的是最近邻插值，使用这种方法可以使得$\phi_N(x_t)=\phi_N(y_t)$.  
这种方法和classifier guidance很相似，甚至不需要训练一个外部模型，对算力友好。  
## 2.5 DiffusionCLIP  
基于扩散模型的图像编辑，使用到的技术有DDIM Inversion，CLIP  
TODO  
# 3 参考  
1. [https://www.bilibili.com/video/BV13P411J7dm](https://www.bilibili.com/video/BV13P411J7dm)  
2. [https://ayandas.me/blog-tut/2021/12/04/diffusion-prob-models.html](https://ayandas.me/blog-tut/2021/12/04/diffusion-prob-models.html)  
3. [https://lilianweng.github.io/posts/2021-07-11-diffusion-models/](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)