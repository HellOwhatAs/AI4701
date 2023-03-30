1. $$
   \begin{cases}
    &\frac{10 - 0}{0 - x_1'} = \frac{10 - 4}{4 - 0}\\
    &\frac{20 - 0}{0 - y_1'} = \frac{10 - 4}{4 - 0}\\
    &\frac{10 - (-5)}{(-5) - x_2'} = \frac{10 - 4}{4 - 0}\\
    &\frac{20 - 0}{0 - y_2'} = \frac{10 - 4}{4 - 0}
   \end{cases} \Rightarrow
   \begin{cases}
    &x_1' = -\frac{20}{3}\\
    &y_1' = -\frac{40}{3}\\
    &x_2' = -15\\
    &y_2' = -\frac{40}{3}
   \end{cases}\\
   \begin{aligned}
    \left(x_1, y_1\right) &= \left(x_1', y_1'\right) = \left(-\frac{20}{3}, -\frac{40}{3}\right)\\
   \left(x_2, y_2\right) &= \left(x_2' + 5, y_2'\right) = \left(-10, -\frac{40}{3}\right)
   \end{aligned}\\
   \begin{aligned}
    x_d &= \left|x_1 - x_2\right|\\
    &= \left|-\frac{20}{3} - \left(-10\right)\right|\\
    &= \frac{10}{3}\\
   \end{aligned}
   $$   
   
2. 证明：
   记一本征矩阵为 $B$，有
   $$
   B = TR
   $$
   其中 $T$ 为 skew symmetrical matrix，$R$ 为旋转矩阵。
   因此 $T$ 可以写成：
   $$
   T = Q'\begin{bmatrix}
      0 & \phi & 0 \\
      -\phi & 0 & 0 \\
      0 & 0 & 0
   \end{bmatrix} Q
   $$
   其中 $\phi$ 为常实数。
   因此
   $$
   \begin{aligned}
      B^\top B &= R^\top T^\top T R \\
      &=\left(QR\right)^\top \begin{bmatrix}
         \phi^2 & 0 & 0 \\
         0 & \phi^2 & 0 \\
         0 & 0 & 0
      \end{bmatrix} \left(QR\right)
   \end{aligned}
   $$
   因此 $B$ 的奇异值为 $0, \phi^2,\phi^2$，即一个奇异值为 $0$ 且其他两个相等。
3. 