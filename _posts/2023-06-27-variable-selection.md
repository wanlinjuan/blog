---
layout: post
title: "Notes - Variable Selection via Nonconcave PenalizedLikelihood and its Oracle Properties"
date: 2023-06-27
excerpt: "Paper summary of Variable Selection via Nonconcave PenalizedLikelihood and its Oracle Properties."
tags: [notes]
comments: true
---

## Penalized least square

Consider the linear regression model $\bm{y}=\bm{X\beta+\epsilon}$, where columns of \textbf{X} are orthonormal.

Let $\bm{z}=\bm{\hat{\beta}^{OLS}}=\bm{X^T y}$, $\bm{\hat{y}}=\bm{XX^Ty}$. The form of penalized least square is
$\begin{equation}
\frac{1}{2} \parallel \bm{y-X\beta}\parallel^2 +\lambda\sum^d_{j=1}p_j(|\beta_j|)
=\frac{1}{2} \parallel \bm{y-\hat{y}}\parallel^2 +\frac{1}{2}\sum_{j=1}^d (z_j-\beta_j)^2+\lambda\sum^d_{j=1}p_j(|\beta_j|)
  \end{equation}$


To solve the minimization problem of (2), we consider a more general version
$
    \frac{1}{2} (z-\theta)^2+p_{\lambda}(|\theta|)
$
, where $p_{\lambda}(\cdot)$ is the penalty function.


### A good penalty function should result in an estimator with three properties.
* Unbiasedness: The resulting estimator is nearly unbiased when the true unknown parameter is large to avoid unnecessary modeling bias.
The sufficient condition for unbiasedness for a large true parameter is that $p'_{\lambda}(|\theta|)=0$ for large $|\theta|$.

* Sparsity: The resulting estimator is a threshold rule.
A sufficient condition for sparsity is that the minimum of the function $|\theta|+p'_{\lambda}(|\theta|)$ is positive.

\paragraph{Continuity: The resulting estimator is continuous in data
z to avoid instability in model prediction.}
The sufficient and necessary condition for continuity is that the minimum of the function $|\theta|+p'_{\lambda}(|\theta|)$ is attained at 0.

### Different penalty functions.

* Bridge regression using $L_q$ penalty: $p_{\lambda}=\lambda|\theta|^q$. The solution is continuous only when $q\ge1$. But it does not produce a sparse solution when $q>1$.
* Ridge regression using $L_2$ penalty: $p_{\lambda}=\lambda|\theta|^2$. Continuous but does not produce a sparse solution.
* Soft thresholding(Lasso) using $L_1$ penalty: $p_{\lambda}=\lambda|\theta|$. Continuous with a thresholding rule. But shift the resulting estimator by constant $\lambda$.
* Smoothly clipped absolute deviation (SCAD) penalty: $p'_{\lambda}=\lambda \{ I(\theta\le \lambda)+ \frac{(a\lambda-\theta)_+}{(a-1)\lambda} I(\theta>\lambda) \}$, for some $a>2$ and $\theta>0$. Continuous. The solution is
$$\hat{\theta}=
\left \{ 
\begin{aligned}
    &sgn(z)(|z|-\lambda)_+, &when&\ |z|\le 2\lambda\\
    &[(a-1)z-sgn(z)a\lambda]/(a-2), &when&\ 2\lambda<|z|\le a\lambda\\
    &z, &when&\ |Z|>a\lambda
\end{aligned}
\right. $$


## Variable selection via penalized likelihood

### Minimization problem
  
* Minimizing penalized least square
$
\frac{1}{2}(\bm{y-X\beta})^T(\bm{y-X\beta})+n\sum^d_{j=1}p_{\lambda}(|\beta_j|)
$

* Minimizing outlier-resistant loss functions
$
\sum^n_{i=1}\Psi(|y_i-\bm{x_i\beta}|)+n\sum^d_{j=1}p_{\lambda}(|\beta_j|)
$

* Minimizing negative penalized likelihood function
$
-\sum^n_{i=1}l_i(g(\bm{x_i^T\beta}),y_i)+n\sum^d_{j=1}p_{\lambda}(|\beta_j|)
$

* Minimizing the Unified form $l(\beta)+n\sum^d_{j=1} p_{\lambda}(|\beta_j|)$.
  
It can be locally approximated by
  
$$l(\beta_0)+\nabla l(\beta_0)^T (\beta-\beta_0) +\frac{1}{2} (\beta-\beta_0)^T\nabla^2 l(\beta_0) (\beta-\beta_0) +\frac{1}{2} n\beta^T\Sigma_{\lambda} (\beta_0) \beta$$

Using Newton-Raphson algorithm, the solution is 
  
$$\beta_1=\beta_0-[\nabla^2 l(\beta_0)+n\Sigma_{\lambda} (\beta_0)]^{-1}[\nabla^2 l(\beta_0)+nU_{\lambda} (\beta_0)]$$
  
where $U_{\lambda}(\beta_0)=\Sigma_{\lambda}(\beta_0)\beta_0$.


### Sampling properties and oracle properties
  
\begin{theorem}{Rate of convergence:}
Let $V_1, ..., V_n$ be iid, each with a density $f(V,\beta)$ that satisfies regularity conditions. If max$\{ |p''_{\lambda_n}(|\beta_{j0}|)| : \beta_{j0}\ne 0 \} \to 0$, then there exists a local maximizer $\hat{\beta}$ of $Q(\beta)$, s.t. $\parallel \hat{\beta}-\beta_0 \parallel =Op(n^{-1/2}+a_n)$, where $a_n= max\{ |p''_{\lambda_n}(|\beta_{j0}|)| : \beta_{j0}\ne 0 \}$.
\end{theorem}

\begin{theorem}{Oracle property:}
Let $V_1, ..., V_n$ be iid, each with a density $f(V,\beta)$ that satisfies regularity conditions. Assume that penalty function $p_{\lambda_n}(|\theta|)$ satisfies 
$lim_{n\to\infty} inf lim_{\theta\to 0+}inf p'_{\lambda_n}(\theta)>0$.
If $\lambda_n\to0$ and $\sqrt{n}\lambda)n\to\infty$ as $n\to \infty$, then with probability tending to 1, the root-n consistent local maximizers $\hat{\beta}=(\hat{\beta_1},\hat{\beta_2})^T$ in Theorem 1 must satisfy:\\
(a) Sparsity: $\hat{\beta_2}=0$.\\
(b) Asymptotic normality: $\sqrt{n}(I_1(\beta_{10})_\Sigma) [\hat{\beta_1}-\beta_{10} + (I_1(\beta_{10})+\Sigma)^{-1}b] \to_d N(0,I_1(\beta_{10}))$
\end{theorem}

Thus, the asymptotic covariance matrix of $\hat{\beta_1}$ is
$$\frac{1}{n} [I_1(\beta_{10})+\Sigma]^{-1} I_1(\beta_{10})[I_1(\beta_{10})+\Sigma]^{-1} $$


\subsection{Standard error formula}
\begin{equation}
    \hat{cov} (\bm{\hat{\beta_1}}) = \{ \nabla^2 l(\bm{\hat{\beta_1}}) + n\Sigma_{\lambda} (\bm{\hat{\beta_1}})\}^{-1}  \hat{cov} \{\nabla l(\bm{ (\hat{\beta_1}})\} \times  \{ \nabla^2 l(\bm{\hat{\beta_1}}) + n\Sigma_{\lambda} (\bm{\hat{\beta_1}})\}^{-1}
\end{equation}



\section{Numerical comparisons}
They use simulation study to compare the performance of proposed approach with existing methods, using median of relative model errors (MRME) and average number of 0 coefficients. They also test the accuracy of the standard error formula.

\section{Conclusion}
\begin{itemize}
    \item With proper choice of regularization parameters, the proposed estimators perform as well as the oracle procedure for variable selection.
    \item The methods are faster and more effective compared with the best subset method.
    \item Standard errors can be estimated accurately.
    \item The proposed methods have strong theoretical backup.
\end{itemize}
