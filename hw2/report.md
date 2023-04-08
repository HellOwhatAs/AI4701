1. 在图中标出两对相似三角形如下：
   ![](./img1.png)
   根据上图中标出的相似三角形关系，列出以下的方程：
   $$
   \begin{cases}
    &\frac{10 - 0}{0 - x_1} = \frac{10 - 4}{4 - 0}\\
    &\frac{20 - 0}{0 - y_1} = \frac{10 - 4}{4 - 0}\\
    &\frac{10 - (-5)}{(-5) - x_2} = \frac{10 - 4}{4 - 0}\\
    &\frac{20 - 0}{0 - y_2} = \frac{10 - 4}{4 - 0}
   \end{cases}
   $$
   解上述方程得到以下解：
   $$
   \begin{cases}
    &x_1 = -\frac{20}{3}\\
    &y_1 = -\frac{40}{3}\\
    &x_2 = -15\\
    &y_2 = -\frac{40}{3}
   \end{cases}
   $$
   求两个投影点在各自相片坐标系中的坐标如下，假设左右 scale factor 分别为 $f_1, f_2$：
   $$
   \begin{aligned}
    \left(x_1', y_1'\right) &= f_1\left(x_1, y_1\right) = \left(-\frac{20}{3}f_1, -\frac{40}{3}f_1\right)\\
   \left(x_2', y_2'\right) &= f_2\left(x_2 + 5, y_2\right) = \left(-10f_2, -\frac{40}{3}f_2\right)
   \end{aligned}
   $$
   计算视差如下：
   $$
   \begin{aligned}
    x_d &= \left|x_1' - x_2'\right|\\
    &= \left|-\frac{20}{3}f_1 + 10f_2\right|\\
   \end{aligned}
   $$
   当 $f_1=f_2=1$ 时，有
   $$
   d = \frac{10}{3}
   $$
   
2. 证明：
   记一本征矩阵为 $E$，则 $E$ 的定义为
   $$
   E = TR
   $$
   其中 $T$ 为 skew symmetrical matrix
   $$
   T = \begin{bmatrix}
      0 & -t_3 & t_2 \\
      t_3 & 0 & -t_1\\
      -t_2 & t_1 & 0
   \end{bmatrix}
   $$
   ，$R$ 为旋转矩阵。
   因此 $T$ 可以分解成：
   $$
   T = Q'\begin{bmatrix}
      0 & \phi & 0 \\
      -\phi & 0 & 0 \\
      0 & 0 & 0
   \end{bmatrix} Q
   $$
   其中 $\phi$ 为常实数，$Q$ 为 orthonormal matrix。
   因此有
   $$
   \begin{aligned}
      E^\top E &= R^\top T^\top T R \\
      &=\left(QR\right)^\top \begin{bmatrix}
         \phi^2 & 0 & 0 \\
         0 & \phi^2 & 0 \\
         0 & 0 & 0
      \end{bmatrix} \left(QR\right)
   \end{aligned}
   $$
   因此 $E$ 的奇异值为 $0, \phi^2,\phi^2$，即一个奇异值为 $0$ 且其他两个相等。
3. 使用的数据集来自 [Chessboard Pictures for Stereocamera Calibration](https://www.kaggle.com/datasets/danielwe14/stereocamera-chessboard-pictures)。我选择了其中的 15 号图片对。