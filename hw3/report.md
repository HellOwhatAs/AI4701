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
2. 给定一个高斯混合模型，目标是针对参数最大化似然函数。
   - 初始化均值 $\mu$、协方差 $\Sigma$ 和混合系数 $\pi$，并计算初始对数似然函数的值。
   - E步：计算每个数据点来自于每个高斯分布的概率，并根据这些概率重新估计隐变量的值。
      $$
      w_{ij} = \frac{\pi_{j}N(x_1|\mu_j, \Sigma_j)}{\sum\limits_{k}\pi_kN(x_i|\mu_k, \Sigma_k)}
      $$
      其中，$w_{ij}$ 表示第 $i$ 个数据点来自第 $j$ 个高斯分布的概率，$x_i$ 表示第 $i$ 个数据点，$\mu_j$ 和 $\Sigma_j$ 分别表示第 $j$ 个高斯分布的均值和方差，$\pi_j$ 表示第 $j$ 个高斯分布的混合系数，$N(x_i| \mu_j, \Sigma_j)$ 表示 $x_i$ 在第 $j$ 个高斯分布下的概率密度函数。
   - M步：根据已知的隐变量和数据，重新估计每个高斯分布的均值、方差和混合系数。
      $$
      \mu_j = \frac{\sum\limits_{i=1}^n w_{ij}x_i}{\sum\limits_{i=1}^nw_{ij}}\\
      \Sigma_j = \frac{\sum\limits_{i=1}^nw_{ij}(x_i-\mu_j)(x_i-\mu_j)^\top}{\sum\limits_{i=1}^nw_{ij}}\\
      \pi_j = \frac{1}{n}\sum\limits_{i=1}^nw_{ij}
      $$
      其中，$\mu_j$ 和 $\Sigma_j$ 分别表示第 $j$ 个高斯分布的均值和方差，$\pi_j$ 表示第 $j$ 个高斯分布的混合系数，$w_{ij}$ 表示第 $i$ 个数据点来自于第 $j$ 个高斯分布的概率。
   - 计算对数似然函数并检查收敛性。
3. 图像分割 Mean-shift 的算法流程伪代码如下：
   ```text
   输入：图像I，窗口大小h，收敛阈值T，最大迭代次数max_iter
   输出：分割结果
   
   1. 初始化：将每个像素点的种子点设置为其自身的像素值，即 s(x_i) = x_i，其中 x_i 表示第 i 个像素点的坐标和像素值。
   2. 对于每个种子点 s(x_i)，进行以下操作：
      a. 初始化平均漂移向量 m = 0。
      b. 初始化迭代次数 iter = 0。
      c. 重复执行以下步骤直到满足收敛条件或达到最大迭代次数：
         i. 计算当前窗口内所有像素点与种子点的距离，并筛选出距离小于 h 的像素点。
         ii. 根据距离加权平均法计算新的中心点，即 m(x) = sum(w(x_j)*x_j)/sum(w(x_j))
             其中 x_j 表示窗口内第 j 个像素点的坐标和像素值，w(x_j) 表示以 s(x_i) 为中心、距离为 x_j 的权重函数。
         iii. 计算当前漂移向量 shift = m - s(x_i)。
         iv. 如果 shift 的模长小于收敛阈值 T，则跳出循环；否则将 s(x_i) 更新为 m，iter 加 1。
      d. 将所有收敛的种子点聚合到一起，得到最终的分割结果。
   
   3. 返回分割结果。
   ```
   影响算法性能的主要因素如下：
   - 窗口大小 h：h越大，算法的平滑效果越好，但速度越慢；h越小，则算法速度越快，但分割结果可能不好。
   - 收敛阈值 T：如果 T 过小，会导致收敛过慢，增加计算量；如果T过大，可能会导致算法过早停止，分割结果不好。
   - 最大迭代次数 max_iter：如果迭代次数过小，可能会导致算法过早停止，分割结果不好；如果迭代次数过大，会导致速度太慢。
   - 初始种子点的选择：初始种子点的选择会影响算法的速度。如果选择很多种子点，则算法的计算量会增加；如果选择过少的种子点，则可能会导致分割结果不好。

4. 编写如下的 [wolfram language](https://www.wolfram.com/language/) 代码，对霍夫变换的阈值参数进行遍历，并将得到的线条数目与检测到的结果的可视化整合在一张表格中，并保存。
   ```wolfram
   SetDirectory[NotebookDirectory[]];
   img = Import["./lines.png"];
   table = Grid[
      Prepend[
         Table[
            {
               N[i/15],
               Length[lines = ImageLines[img, i/15, Method -> "Hough"]],
               HighlightImage[img, lines]
            },
            {i, 2, 6}
         ],
         {"threshold", "line count", "result"}
      ], Frame -> All]
   Export["./Hough.svg", table];
   ```
   程序读取的原图如下：
   ![](./lines.png)
   运行程序得到的输出结果如下：
   <center><img src="./Hough.svg" style="width:70%"/></center>

   从以上结果中可以看出，当阈值参数过小时，会检测出过多的线。而当阈值过大时，又存在不够敏感的情况。

5. 编写如下的 [python](https://www.python.org/) 代码，读取图像，进行特定方向上的线拟合，并根据拟合结果在原始图像上添加删除线，保存为另一幅图片。
   ```py
   import cv2
   import numpy as np
   from itertools import groupby
   
   img = cv2.imread("./text.png")
   thresh = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
   
   for val, elems in groupby(
      ((np.sum(thresh[x]) > 100, x) for x in range(thresh.shape[0])),
      key=lambda x:x[0]
   ):
       if not val: continue
       elems = list(elems)
       start, end = elems[0][1], elems[-1][1]
       tmp = np.where(np.sum(thresh[start:end], axis=0) > 100)[0]
       mean = round(sum(np.sum(thresh[i]) * i for i in range(start, end)) / np.sum(thresh[start:end]))
       cv2.line(img, (tmp[0], mean), (tmp[-1], mean), (0, 0, 255), (end - start) // 10 + 1)
   
   cv2.imwrite("./delete-text.png", img)
   ```
   |程序读取的原图|程序输出的结果|
   |:-:|:-:|
   |![](./text.png)|![](./delete-text.png)|