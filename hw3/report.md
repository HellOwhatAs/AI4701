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
2. ...
3. ...
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
   ![](./Hough.svg)

   从以上结果中可以看出，当阈值参数过小时，会检测出过多的线。而当阈值过大时，又存在不够敏感的情况。

5. ...