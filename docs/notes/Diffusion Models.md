---  
title: Diffusion Models  
share: "true"  
---  
  
TODO  
2023 年扩散模型还有什么可做的方向？ - 谷粒多·凯特的回答 - 知乎 https://www.zhihu.com/question/568791838/answer/3195773725  
# 原理篇  
## 基于score的生成模型  
基于score的生成模型和扩散模型非常相似，使用了score matching和Langevin dynamics技术进行生成。其中，  
1. score matching是估计目标分布的概率密度的梯度（即score，分数），记$p(x)$是数据分布的概率密度函数，则这个分布的score被定义为$\nabla_x\log p(x)$，score matching则是训练一个网络$s_\theta$去近似score：  
$$\mathcal{E}_{p(x)}\left[ \|\nabla_x\log p(x)-s_\theta(x)\|^2_2 \right]=\int p(x)\|\nabla_x\log p(x)-s_\theta(x)\|^2_2 dx$$  
3. Langevin dynamics是使用score采样生成数据，采样方式如下：  
$$  
x_t=x_{t-1}+\frac{\delta}{2}\nabla_x\log p(x_{t-1})+\sqrt{\delta}\epsilon, \text{    where } \epsilon\sim\mathcal{N}(0, I)  
$$  
## DDIM  
> [ref link](https://www.bilibili.com/video/BV13P411J7dm)  
#### Review of DDPM  
1. Diffusion阶段  
$$  
\begin{align}  
q(x_t|x_0)&=\boxed{\mathcal{N}(x_t;\sqrt{\bar{\alpha_t}}x_0,(1-\bar{\alpha_t})I)}\\  
         &=\sqrt{\bar{\alpha_t}}x_0+\sqrt{1-\bar{\alpha_t}}\epsilon  
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
### From DDPM to DDIM  
  
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
# 应用篇  
## SR3  
超分，训练数据是LR和SR配对的图片，以LR图片作为condition，生成SR图片  
## CDM  
超分，级联的方式对小图进行超分，采用的方法就是SR3  
![725](../assets/img/Pasted%20image%2020230927200225.png)  
## SDEdit  
![900](../assets/img/Pasted%20image%2020230927200246.png)  
由于加噪过程是首先破坏高频信息，然后才破坏低频信息，所以加噪到一定程度之后，就就可以去掉不想要的细节纹理，但仍保留大体结构，于是生成出来的图像就既能遵循输入的引导，又显得真实。但是需要 realism-faithfulness trade-off  
## ILVR  
给定一个参考图像$y$，通过调整DDPM去噪过程，希望让模型生成的图像接近参考图像，作者定义的接近是让模型能够满足  
$$  
\phi_N(x_t)=\phi_N(y_t)  
$$  
$\phi_N(\cdot)$是一个低通滤波器（下采样之后再插值回来）。使用如下的算法：  
![450](../assets/img/Pasted%20image%2020230927201110.png)  
即，对DDPM预测的$x'_{t-1}$加上bias：$\phi_N(y_{t-1})-\phi_N(x'_{t-1})$，可以证明，如果上/下采样采用的是最近邻插值，使用这种方法可以使得$\phi_N(x_t)=\phi_N(y_t)$.  
这种方法和classifier guidance很相似，甚至不需要训练一个外部模型，对算力友好。  
## DiffusionCLIP  
基于扩散模型的图像编辑，使用到的技术有DDIM Inversion，CLIP  
TODO