import cv2
import numpy as np
from itertools import groupby

img = cv2.imread("./text.png")
thresh = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

for val, elems in groupby(((np.sum(thresh[x]) > 100, x) for x in range(thresh.shape[0])), key=lambda x:x[0]):
    if not val: continue
    elems = list(elems)
    start, end = elems[0][1], elems[-1][1]
    tmp = np.where(np.sum(thresh[start:end], axis=0) > 100)[0]
    mean = round(sum(np.sum(thresh[i]) * i for i in range(start, end)) / np.sum(thresh[start:end]))
    cv2.line(img, (tmp[0], mean), (tmp[-1], mean), (0, 0, 255), (end - start) // 10 + 1)

cv2.imwrite("./delete-text.png", img)