---
title: A Casual Talk on Linear Regression
date: 2020-08-13 20:00:00 +0800
categories: [statistics]
tags: [statistics]
math: true
---

## Introduction

During my years at a tech company building data analysis and decision-making products—including A/B testing systems, observational causal inference systems, and black-box parameter optimization systems—I've had a unique opportunity to understand and reflect on the essence of data-driven methodologies. I've gradually come to realize that they all share a unified and elegant mathematical core: **linear regression**.

This article aims to clearly articulate these insights in a concise manner:
1. **Bayesian inference** is a shortcut to building an intuitive understanding of the essence of linear regression
2. **Linear regression** is a shortcut to deeply understanding the principles of all the aforementioned systems:
   1. **Randomized A/B experiments** are linear regressions with a single discrete variable
   2. Common observational causal inference methods, such as **backdoor adjustment**, are multivariate linear regressions that incorporate control variable sets alongside the treatment itself
   3. Common **black-box optimization** methods, such as Bayesian optimization based on Gaussian Process Regression, are linear regressions in a feature space defined by a kernel

## Linear Regression Through a Bayesian Lens

### Simple Linear Regression

In simple linear regression, the loss function we wish to minimize is:

$$
L(\mathbf{w}) = \sum_{i=1}^{n} (y_i - \mathbf{w}^\top \mathbf{x}_i)^2 \tag{1.1}
$$

We call this most basic form of linear regression **Ordinary Least Squares (OLS)**.

In practice, we often add terms related to the coefficients $\mathbf{w}$ to the loss function, called **regularizers**, which are claimed to have the effect of **suppressing overfitting**. For example, adding the sum of squares of coefficient terms:

$$
L(\mathbf{w}) = \sum_{i=1}^{n} (y_i - \mathbf{w}^\top \mathbf{x}_i)^2 + \lambda \|\mathbf{w}\|^2 \tag{1.2}
$$

Regression with the above loss function as the optimization objective is called [**Ridge Regression**](https://en.wikipedia.org/wiki/Tikhonov_regularization). There are also other regularization methods such as [**LASSO**](https://en.wikipedia.org/wiki/Lasso_(statistics)).

This is what most materials tell us about linear regression. This perspective, especially after adding regularization terms, makes it difficult for ordinary people to form a clear concept:

> **"What exactly am I optimizing?"**

Although the loss function can be analyzed within the framework of traditional methods, there is a far more intuitive explanation from the perspective of Bayesian generative models.

### Bayesian Linear Regression

The main setup of Bayesian linear regression is also quite simple:

$$
y = \mathbf{w}^\top \mathbf{x} + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma_n^2) \tag{1.3}
$$

In one sentence:

> **The observed sample value $y$ consists of a linear combination of inputs plus an independent Gaussian random noise.**

The purpose of regression is to obtain the **posterior distribution of coefficients**. Recall Bayes' formula:

$$
p(\mathbf{w} | X, \mathbf{y}) = \frac{p(\mathbf{y} | X, \mathbf{w}) \cdot p(\mathbf{w})}{p(\mathbf{y} | X)} \tag{1.4}
$$

It is proportional to the product of **Likelihood** and **Prior**.

#### Likelihood

Since samples are independent, we can conveniently obtain the likelihood of the observed dataset using multiplication:

$$
\begin{align}
p(\mathbf{y}|X, \mathbf{w}) &= \prod_{i=1}^{n} p(y_i|\mathbf{x}_i, \mathbf{w}) = \prod_{i=1}^{n} \frac{1}{\sqrt{2\pi}\,\sigma_n} \exp\left(-\frac{(y_i - \mathbf{x}_i^{\top}\mathbf{w})^2}{2\sigma_n^2}\right) \\
&= \frac{1}{(2\pi\sigma_n^2)^{n/2}} \exp\left(-\frac{1}{2\sigma_n^2}|\mathbf{y} - X^{\top}\mathbf{w}|^2\right) = \mathcal{N}(X^{\top}\mathbf{w}, \sigma_n^2 I) \tag{1.5}
\end{align}
$$

We find that the likelihood can be described using **Euclidean distance**, and unsurprisingly, it is exactly a **multivariate Gaussian distribution**.

#### Prior

Now that we have the likelihood, we only need to specify a prior belief for $\mathbf{w}$ to complete the task. Without loss of generality, let's continue using a normal distribution to describe the prior:

$$
\mathbf{w} \sim \mathcal{N}(\mathbf{0}, \Sigma_p). \tag{1.6}
$$

#### Posterior

Then, multiplying the prior and likelihood together, we find:

$$
p(\mathbf{w}|X, \mathbf{y}) \propto \exp\left(-\frac{1}{2\sigma_n^2}(\mathbf{y} - X^{\top}\mathbf{w})^{\top}(\mathbf{y} - X^{\top}\mathbf{w})\right) \exp\left(-\frac{1}{2}\mathbf{w}^{\top}\Sigma_p^{-1}\mathbf{w}\right) \tag{1.7}
$$

Note that:

1. **The product of exponentials equals the exponential of the sum**
2. **Since exp is monotonic, optimizing it is equivalent to optimizing its argument**

Looking more carefully, the exponent of the above expression is exactly the form of ridge regression—the likelihood constitutes the Euclidean distance term, and the prior constitutes the regularization term:

$$
\underset{\mathbf{w}}{\min}\  \|X^\top \mathbf{w} - \mathbf{y}\|_2^2 + \alpha \|\mathbf{w}\|_2^2 \tag{1.8}
$$

Thus we finally have our epiphany: so-called ridge regression is nothing more than adding a Gaussian prior belief to the parameters $\mathbf{w}$.

From this angle, why a regularization term suppresses overfitting also becomes simple and direct. Without regularization, we are implicitly assuming that any value of $\mathbf{w}$ is equally likely before observing the data—an uninformative prior. In that case, we have no choice but to fully accept what the data (likelihood) tells us, ultimately paying excessive attention to every detail and memorizing some of the noise. Adding a prior is a compromise with the data: we no longer trust the data unconditionally, but weight what it says against our prior belief, thereby suppressing overfitting.

Following this idea, the other commonly used regularizer, L1 LASSO (least absolute shrinkage and selection operator), can be easily derived to correspond to a Laplace prior. The reason L1 produces sparser coefficients and acts as a selection mechanism is simply that the Laplace prior assigns more probability density near zero.

Continuing to simplify the above expression, we can see that the ultimate goal of regression: the posterior distribution of $\mathbf{w}$ is still a **Gaussian distribution** (this property is also called **Gaussian-Gaussian conjugacy**):

$$
p(\mathbf{w} | X, \mathbf{y}) \sim \mathcal{N}\big(\bar{\mathbf{w}} = \sigma_n^{-2} A^{-1} X\mathbf{y},\, A^{-1}\big) \tag{1.9}
$$

where:

$$
A = \sigma_n^{-2} X X^\top + \Sigma_p^{-1} \tag{1.10}
$$

After obtaining the posterior distribution of $\mathbf{w}$, we can weight-average the likelihood over each value of $\mathbf{w}$. For any newly given input $\mathbf{x}_*$, we can give its predicted value $f_*$, which, unsurprisingly, is still a **Gaussian distribution** (Gaussian-Gaussian conjugacy).

$$
\begin{align}
p(f_*|\mathbf{x}_*, X, \mathbf{y}) &= \int p(f_*|\mathbf{x}_*, \mathbf{w})p(\mathbf{w}|X, \mathbf{y}) \, d\mathbf{w} \\
&= \mathcal{N}\left(\frac{1}{\sigma_n^2}\mathbf{x}_*^{\top} A^{-1} X\mathbf{y}, \, \mathbf{x}_*^{\top} A^{-1} \mathbf{x}_*\right).
\end{align} \tag{1.11}
$$

At this point, we can already make inferences about $y_*$ corresponding to other $x_*$ based on existing $(x, y)$ sample observations, completing the main function of linear regression.

## Randomized Experiment As Nominal Regression

Now let's consider the familiar two-sample hypothesis test from a regression perspective. In such problems, the conventional NHST approach relies on the central limit theorem to treat the sample means of two groups as two normal random variables, and performs tests based on sample means, aiming to provide an interval estimate for the difference in population expectations. If the interval doesn't include 0, the difference is declared significant.

Now let's construct this problem from the perspective of linear regression. We adopt the Bayesian simple linear regression model:

$$f(\mathbf{x}) = \mathbf{x}^{\top}\mathbf{w}, \qquad y = f(\mathbf{x}) + \varepsilon \tag{2.1}$$

$$\varepsilon \sim \mathcal{N}(0, \sigma_n^2) \tag{2.2}$$

Let variable $\mathbf{x}$ be a nominal variable, where the first part is the one-hot encoding of the experimental variant, and the last entry is an augmented bias term fixed at 1, used to multiply the intercept term of the linear regression:

$$\tilde{\mathbf{x}}_i = \big(\,\text{one-hot}(\text{variant}_i),\; 1\,\big)^\top \tag{2.3}$$

For example, if an experiment has two groups, $\tilde{\mathbf{x}}$ takes two distinct values representing which group the data comes from:

$$\tilde{\mathbf{x}}_1 = (1, 0, 1)^\top,\quad \tilde{\mathbf{x}}_2 = (0, 1, 1)^\top \tag{2.4}$$

$$\mathbf{w} = (w_1, w_2, w_3)^\top \tag{2.5}$$

Thus we can easily obtain:

$$y_1 = w_1 + w_3 + \epsilon \tag{2.6}$$

$$y_2 = w_2 + w_3 + \epsilon \tag{2.7}$$

We care about the difference in group expectations. By linearity of expectation, this difference equals the difference of the corresponding intercepts:

$$\mathbb{E}[y_2 - y_1] = w_2 - w_1 \tag{2.8}$$

According to the posterior distribution of $\mathbf{w}$ derived in the previous section:

$$p(\mathbf{w}|X, \mathbf{y}) \sim \mathcal{N}\big(\bar{\mathbf{w}} = \tfrac{1}{\sigma_n^2}A^{-1}X\mathbf{y},\, A^{-1}\big) \tag{2.9}$$

We know that $w_1$ and $w_2$ form a bivariate Gaussian distribution. Since the marginal of a multivariate Gaussian is still Gaussian, $w_1$ and $w_2$ are themselves Gaussian. In other words, from a Bayesian perspective, regression similarly describes the difference between two population expectations as the difference of two normal random variables (which is also Gaussian). Following the standard Bayesian inference procedure, we only need to draw $w_2 - w_1$ (called the *contrast*) repeatedly from the posterior—e.g., via MCMC sampling—to obtain a Bayesian interval estimate for the difference in expectations between the two groups.

At this point, we have demonstrated that **randomized A/B experiments are essentially a form of linear regression where the independent variable is a nominal variable**.

## Observational Causal Inference As Long Regression

### Statistical Adjustment

In ideal A/B experiments, complete randomization allows us to draw causal conclusions by directly comparing metric differences between treatment and control groups. However, in many practical scenarios, randomization is difficult to implement. For example, after a new version of an application is released, users typically choose whether to upgrade promptly. Users who choose to upgrade are often more active, younger, and have higher technology acceptance, and these characteristics themselves affect subsequent retention and activity metrics. Therefore, directly comparing upgraded users with non-upgraded users typically severely overestimates the effect of the new version.

```
       Age/Gender/Activity/...
       (Confounders W)
           /        \
          ↓          ↓
    Upgrade(T) → Retention(Y)
```

The characteristic of such scenarios is that the treatment assignment we care about is not random but is influenced by certain user characteristics, and these characteristics simultaneously affect the outcome. In the language of observational causal inference, they are called confounders. At this point, there exists a backdoor path between $T$ and $Y$, so the simple conditional expectation difference is not purely the causal effect:

$$
\mathbb{E}[Y \mid T=1] - \mathbb{E}[Y \mid T=0] \tag{3.1}
$$

Statistical adjustment is the most commonly used correction method. To correctly estimate the true causal relationship from treatment to outcome, it requires "controlling" for all confounders between them. Conceptually, this is a process of stratifying by all confounding variables and then weighting. The actual mathematical implementation is precisely **multivariate linear regression**:

$$
Y = \alpha + \rho \cdot T + \gamma^T W + \epsilon \tag{3.2}
$$

where:

- $T$: treatment variable
- $W = [W_1, \ldots, W_p]^T$: vector of confounding variables
- $\rho$: **partial regression coefficient** of treatment, representing **the causal effect estimate after controlling for confounders**
- $\gamma$: effect coefficients of confounding variables

Comparing this with the regression equation corresponding to A/B experiments in the previous section (which only involves $T$!), we clearly have a "longer" version. So, what impact does adding $W$ to the regressors have on the results?

### Relationship Between Long and Short Regression

In linear regression theory, the difference between long and short regression is described by Omitted Variable Bias.

Simply put: if we ignore variable $W$ and directly fit the short regression:

$$
Y = \beta_0 + \beta_1 \cdot T + u \tag{3.3}
$$

It can be proven that the obtained regression coefficient $\beta_1$ has the following relationship with the partial regression coefficient $\rho$ in equation (3.2):

$$
\hat{\beta}_1 = \hat{\rho} + \hat{\gamma}^T \frac{\text{Cov}(T, W)}{\text{Var}(T)} \tag{3.4}
$$

This tells us:
- When the covariance between $T$ and $W$ is non-zero (i.e., there exist confounding variables affecting treatment assignment), the expected regression coefficients of $T$ (which we consider the causal effect of treatment) obtained from long and short regressions are not equal. The difference between them is the confounding bias in causal inference terminology
- When the covariance between $T$ and $W$ is zero (i.e., various characteristics don't affect treatment assignment), the expected regression coefficients of $T$ obtained from long and short regressions are actually equal (confounding bias is 0). In other words, whether covariates $W$ are included doesn't affect the expected value of the causal effect estimate (but may affect the estimation variance, which is actually the regression explanation for variance reduction methods commonly used in experiments like CUPED)

At this point, we have shown that the commonly used statistical adjustment in observational causal inference is mathematically a multivariate linear regression that is longer than the nominal variable regression of randomized experiments.

## Bayesian Optimization as Linear Regression in Feature Space

In practical business, we often encounter scenarios requiring "black-box optimization," such as certain hyperparameters in recommendation systems and certain key parameters in quantitative strategies, which are tuned through repeated online experiments observing metric changes. The common requirement of such tasks is iteratively exploring a (possibly infinite) parameter space in order to identify optimal configurations in a resource-efficient manner.

The vast search space and extremely limited exploration budget make random search or grid search infeasible, so machine learning-guided approaches are often adopted. Bayesian optimization based on Gaussian processes is one of the most commonly used models in this scenario. It typically assumes the objective follows a Gaussian process with a certain "kernel" in the sample space, thereby using already explored data points to predict unexplored space, then trading off exploration and exploitation to select the next round of exploration, hoping to quickly converge to the global optimum. Some popular open-source components, such as Meta's [Ax (Adaptive Experimentation Platform)](https://github.com/facebook/Ax/tree/main?tab=readme-ov-file), are built on this basis.

In this section, we will elucidate the essence of kernel methods, represented by Gaussian processes, from the perspective of linear regression.

### In the Name of a Kernel

First, let's rewrite equation (1.11) to some extent, defining a function:

$$
k(x, x') = x^T \Sigma_p x' \tag{4.1}
$$

We can find that in the above expression for $f_*$, the mean and variance can be written in the following forms:

$$
\bar{f}_* \triangleq \mathbb{E}[f_*|X, \mathbf{y}, X_*] = K(X_*, X)[K(X, X) + \sigma_n^2 I]^{-1}\mathbf{y} \tag{4.2}
$$

$$
\text{cov}(\bar{f}_*) = K(X_*, X_*) - K(X_*, X)[K(X, X) + \sigma_n^2 I]^{-1}K(X, X_*) \tag{4.3}
$$

For now, this form seems more complex. As for why we do this, we'll explain in the next section.

### Gaussian Process

In the previous section, we derived the posterior of parameter $\mathbf{w}$ in the linear regression model from likelihood and prior in an inferential manner. We further used the posterior distribution to obtain the predictive distribution for predicting the function value corresponding to each point, and after some matrix transformations, we found that the predictive distribution can be written in the form of kernel functions.

In this section, we use stochastic processes for prediction. A stochastic process is a mathematical concept that describes a distribution over functions. Loosely speaking, you can think of a "function" as a very long vector, and a stochastic process defines how each element of this vector takes values. A Gaussian process is a special type of stochastic process where any finite sample set forms a multivariate Gaussian distribution. This means that extracting any (denote $n$) finite samples from this infinitely long vector forms an $n$-variate Gaussian distribution. The mean and covariance between any two points of this multivariate Gaussian distribution are denoted as two functions $m$ and $k$:

$$\mathbb{E}[f(\vec{x})] = m(\vec{x}) \tag{4.4}$$

$$cov(\vec{x}_1, \vec{x}_2) = k(\vec{x}_1, \vec{x}_2) + \sigma_n^2\delta_{pq} \tag{4.5}$$

Usually for simplicity, we set the prior:

$$m(\vec{x}) = \vec{0} \tag{4.6}$$

Now, assuming the regression target y follows a Gaussian stochastic process, according to the definition of a Gaussian process, we obtain:

$$P(f(\vec{x}_1), f(\vec{x}_2), \cdots, f(\vec{x}_n), f(\vec{x}_*)) = \mathcal{N}(\vec{0}, K([\vec{X}, \vec{x}_*], [\vec{X}, \vec{x}_*]) + \sigma_n^2\vec{I}) \tag{4.7}$$

Since f(x1), f(x2), ... f(xn) are all known, finding f(x*) is equivalent to solving the conditional probability P(f(x*) | f(x1), f(x2), ... f(xn)). By the basic definition of conditional probability:

$$P(f(\vec{x}_*) | f(\vec{x}_1), f(\vec{x}_2), \cdots, f(\vec{x}_n)) = \frac{P(f(\vec{x}_*), f(\vec{x}_1), f(\vec{x}_2), \cdots, f(\vec{x}_n))}{P(f(\vec{x}_1), f(\vec{x}_2), \cdots, f(\vec{x}_n))} = \frac{\mathcal{N}(\vec{0}, K([\vec{X}, \vec{x}_*], [\vec{X}, \vec{x}_*]) + \sigma_n^2\vec{I})}{\mathcal{N}(\vec{0}, K(X, X) + \sigma_n^2\vec{I})}\tag{4.8}$$

The conditional probability distribution of a Gaussian distribution is still a Gaussian distribution, so the final result of the above equation is still a Gaussian distribution. In fact, after some rearrangement, the final result takes the form:

$$\mathbf{f}_*|X, \mathbf{y}, X_* \sim \mathcal{N}(\bar{\mathbf{f}}_*, \text{cov}(\bar{\mathbf{f}}_*))\tag{4.9}$$

$$\bar{\mathbf{f}}_* \triangleq \mathbb{E}[\mathbf{f}_*|X, \mathbf{y}, X_*] = K(X_*, X)[K(X, X) + \sigma_n^2 I]^{-1}\mathbf{y}\tag{4.10}$$

$$\text{cov}(\bar{\mathbf{f}}_*) = K(X_*, X_*) - K(X_*, X)[K(X, X) + \sigma_n^2 I]^{-1}K(X, X_*) \tag{4.11}
$$

### Feature Space

#### Linear Kernel

Comparing (4.10-11) with (4.2-3), we find that their forms are completely identical. In other words, simple linear regression is completely equivalent to a special Gaussian process with kernel function:

$$k(\vec{x}_1, \vec{x}_2) = \vec{x}_1^T \Sigma_p \vec{x}_2 \tag{4.12}$$

Unsurprisingly, this kernel is called the linear kernel.

#### Polynomial Kernel

So what role do other commonly used kernel functions of Gaussian processes play in linear regression? Let's look at a simple example. Define a new kernel function:

$$k(x_1, x_2) = (x_1 \cdot x_2 + \frac{1}{2})^2 \tag{4.13}$$

With some simple transformations, we can write the right side as an inner product:

$$(x_1 \cdot x_2 + \frac{1}{2})^2 = x_1^2 x_2^2 + x_1 x_2 + \frac{1}{4} = (x_1^2, x_1, \frac{1}{2}) \cdot (x_2^2, x_2, \frac{1}{2}) \tag{4.14}$$

We discover that what this kernel function fundamentally does is map sample $x$ to another vector:

$$x \rightarrow (x^2, x, \frac{1}{2}) \tag{4.15}$$

The remaining regression part is completely identical. This suggests that if we call the left side of the above arrow the sample space and the right side a feature space derived from the samples, then the Gaussian process defined by the above kernel function is fundamentally a linear regression using polynomial features of the samples.

So what is the purpose of this mapping? Because in most cases, the expressiveness of raw samples themselves is extremely limited. For example, in the original sample space, $x$ and $y$ clearly don't have a linear relationship, so we can't directly use linear regression to describe it. Now we choose to expand each $x$ into a polynomial feature vector $(1, x, x^2)$, and we may find that in this new 3-dimensional feature space, there's a relatively obvious linear relationship between the target value and the basis vectors, fluctuating around a 2-dimensional plane. This suggests that in this polynomial feature space, we can use linear regression to describe the relationship between $(1, x, x^2)$ and $y$. In the semantics of traditional linear regression, this set of operations is called *polynomial regression*. Correspondingly, in the semantics of Gaussian processes, the related kernel function is called the *polynomial kernel*.

#### Gaussian Feature Space

Furthermore, let's take the squared exponential kernel, frequently used in modern Gaussian process regression, as an example to see what kind of mapping it represents. This kernel function is written as:

$$k(\vec{x}_1, \vec{x}_2) = e^{-\frac{\|\vec{x}_1 - \vec{x}_2\|^2}{2l^2}} \tag{4.16}$$

Our goal is still to try to rewrite this kernel function into something like an inner product to explore the properties of its feature space:

$$k(\vec{x}_1, \vec{x}_2) = \vec{x}_1^T \Sigma_p \vec{x}_2 \tag{4.17}$$

For simplicity, let's directly peek at the answer and then work backward from it. Define a new function $\phi$:

$$\phi_c(x) = e^{-\frac{(x-c)^2}{2l^2}} \tag{4.18}$$

This represents mapping sample $x$ to the value of a Gaussian function centered at $c$. Then integrate $c$ from $-∞$ to $∞$:

$$\int_{-\infty}^{\infty} \exp\left(-\frac{(x_p - c)^2}{2\ell^2}\right) \exp\left(-\frac{(x_q - c)^2}{2\ell^2}\right) dc \tag{4.19}$$

Expanding, completing the square, and using Gaussian integral with substitution and integration, we get the final result:

$$k(x_p, x_q) = \sigma_p^2 \int_{-\infty}^{\infty} \exp\left(-\frac{(x_p - c)^2}{2\ell^2}\right) \exp\left(-\frac{(x_q - c)^2}{2\ell^2}\right) dc \tag{4.20}$$

$$= \sqrt{\pi}\ell\sigma_p^2 \exp\left(-\frac{(x_p - x_q)^2}{2(\sqrt{2}\ell)^2}\right) \tag{4.21}$$

The final result, ignoring some constant terms, is exactly the form of the squared exponential kernel. Recall that an integral can be viewed as the limit of a Riemann sum. Discretizing $c$ at evenly spaced points $\{c_i\}_{i=1}^{N}$ with spacing $\Delta c$, we have:

$$\int_{-\infty}^{\infty} \phi_c(x_p)\,\phi_c(x_q)\,dc \;=\; \lim_{\substack{N \to \infty \\ \Delta c \to 0}} \sum_{i=1}^{N} \phi_{c_i}(x_p)\,\phi_{c_i}(x_q)\,\Delta c \tag{4.22}$$

For each fixed $N$, the summation on the right is, up to the constant spacing $\Delta c$, the inner product of two $N$-dimensional vectors:

$$\sum_{i=1}^{N} \phi_{c_i}(x_p)\,\phi_{c_i}(x_q) \;=\; \big(\phi_{c_1}(x_p),\dots,\phi_{c_N}(x_p)\big) \cdot \big(\phi_{c_1}(x_q),\dots,\phi_{c_N}(x_q)\big) \tag{4.23}$$

In other words, the squared exponential kernel actually defines a mapping from the sample space to an infinite-dimensional feature space:

$$x_p \to \big(\phi_{c_1}(x_p),\, \phi_{c_2}(x_p),\, \dots\big) \tag{4.24}$$

Then performs regression in this infinite-dimensional feature space.

### Kernel Trick

At this point, we can basically conclude: Gaussian processes are essentially a special form of linear regression that maps samples to a feature space, where the kernel function defines the specific process of this mapping. Different kernels in Gaussian processes have the effect of merely replacing:

$$k(\vec{x}_1, \vec{x}_2) = \vec{x}_1^T \Sigma_p \vec{x}_2 \tag{4.25}$$

in linear regression with:

$$k(\vec{x}_1, \vec{x}_2) = \phi(\vec{x}_1)^T \Sigma_p \phi(\vec{x}_2) \tag{4.26}$$

Additionally, it's worth mentioning that we only "equivalently" completed the feature space mapping through the kernel, without actually constructing a feature space and performing regression. Carefully comparing the results before and after the kernel trick, we can see: if we denote the number of samples as n and the number of features corresponding to each sample in the feature space as N, before the kernel trick, the matrix we need to invert has size $N*N$, while after the kernel trick it's $n*n$. Considering that the time complexity of general matrix inversion is $O(n^3)$, for some complex feature spaces (such as infinite-dimensional ones), when n is much smaller than N, using kernels greatly reduces computational load and even makes impossible computations possible. Therefore, this ingenious shortcut is called the *kernel trick*.

## Conclusion

```
                    ┌──> Nominal Variable Regression ───> Hypothesis Testing ───> Randomized Experimental System ──┐
                    │                                                                                              |                                 
Regression ─────────┼──> Nominal + Control Variable Regression ───> Long Regression ───> Observational System ─────┤──> Generalized Causal Inference System
                    │                                                                                              |                                 
                    └──> Regression within Feature Space ─────> Kernel Methods  ──> Optimization System ───────────┘
```