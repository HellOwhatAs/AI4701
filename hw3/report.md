1. 与直线垂直的向量：
   $$
    \mathbf{v} = \begin{bmatrix}
    a\\b
   \end{bmatrix}
   $$
   从点 $(u, v)$ 到线的向量：
   $$
   \mathbf{r} = \begin{bmatrix}
    x-u\\y-v
   \end{bmatrix}
   $$
   点到直线的距离为：
   $$
   \begin{aligned}
    d &= \lvert \text{proj}_\mathbf{v}\mathbf{r} \rvert\\
    &= \frac{\lvert \mathbf{v}\cdot \mathbf{r}\rvert}{\lvert \mathbf{v}\rvert}\\
    &= \lvert\hat{\mathbf{v}}\cdot \mathbf{r}\rvert\\
    &= \frac{\lvert a\left( x-u \right) + b\left(y-v\right) \rvert}{\sqrt{a^2+b^2}}\\
    &= \frac{\lvert ax + by -au -bv \rvert}{\sqrt{a^2+b^2}}\\
    &=\frac{\lvert au+bv+c\rvert}{\sqrt{a^2+b^2}}\\
    &= \lvert au+bv+c \rvert
   \end{aligned}
   $$
2. 