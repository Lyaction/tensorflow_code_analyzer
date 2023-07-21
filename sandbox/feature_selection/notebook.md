# Variational Dropout For Feature Selection
## Dropout
> Dropout: A Simple Way to Prevent Neural Networks from Overfitting

随机丢弃神经元，增强模型鲁棒性，以此解决数据过拟合问题
### 学习
1. 预设训练步数 $t$ ,神经元丢弃率 $p$ ，随机初始化模型参数 $\theta_{0}$ 
2. 随机采样 $(\mathbf{x_{\mathrm{t}}},y_{t})$ , 随机掩码 $\mathbf{z}_{t}\sim{Bernoulli(p)}$
3. 随机梯度更新公式如下
   $$\theta_{t+1}=\theta_{t}+\eta_{t}\frac{\partial{\log{p(y_{t}|\mathbf{x_{\mathrm{t}}},\mathbf{z}_{t},\theta)}}}{\partial{\theta}}$$
4. $t=t+1$ ,回到步骤二
### 预估
假设 $\mathbf{z}$ 是一个 $m$ 维的向量，则 $2^{m}$ 个模型的输出的期望作为预估值，即：
$$E_{p(\mathbf{z})}[y(\mathbf{x};\mathbf{z},\theta)]\approx{y(\mathbf{x}^*;E_{p(\mathbf{z})}[\mathbf{z}],c)}$$
Keras 中的实现是：学习阶段 dropout 层丢弃神经元，剩余神经元乘以系数 $1/(1-p)$ 进行还原,预估时如下： 
$$y(\mathbf{x};\mathbf{z}=\mathbf{1},\theta)$$
## 变分推断（Variational Inference）
使用简单分布拟合贝叶斯模型中的复杂后验，衡量指标为 KL 散度
### 建模思路
$D$ 为样本全集， $T$ 为样本数， $m$ 为特征数
1. 子模型的权重取决于特征重要性
2. 求解子模型的权重分布，即为掩码 $\mathbf{z}$ 的后验分布
3. 后验直接求解的复杂度过高 $T^m$ ，采用变分分布 $q(\mathbf{z})$ 拟合
### 优化目标推导
最小化变分分布与后验的 KL 散度：
$$KL[q(\mathbf{z})|p(\mathbf{z}|D,\theta)]=E_{q(\mathbf{z})}[\log{\frac{q(\mathbf{z})}{p(\mathbf{z}|D,\theta)}}]$$
展开后可得：
$$E_{q(\mathbf{z})}[\log{q(\mathbf{z})}]-E_{q(\mathbf{z})}[\log{p(\mathbf{z},D|\theta)}]+E_{q(\mathbf{z})}[\log{p(D|\theta)}]$$
第三项为定值，前两项取负即为 ELBO ，最小化 KL 散度转化为最大化 ELBO 问题：
$$ELBO=E_{q(\mathbf{z})}[\log{p(\mathbf{z},D|\theta)}]-E_{q(\mathbf{z})}[\log{q(\mathbf{z})}]$$
对 ELBO 进行如下改写：
$$ELBO=\sum_{t=1}^{T}{E_{q(\mathbf{z})}[\log{p(y_{t}|\mathbf{x_{\mathrm{t}}},\mathbf{z},\theta)}]}+E_{q(\mathbf{z})}[\log{p(\mathbf{z})}]-E_{q(\mathbf{z})}[\log{q(\mathbf{z})}]$$
其中第二项和第三项为变分分布与后验分布的 KL 散度
## 优化算法
### Score Function Estimators
### Reparameterization Trick
## 实验
### 人工构造样本
### 卧龙广告样本
