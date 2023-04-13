import numpy as np
import matplotlib.pyplot as plt
import cv2, os
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

sam_checkpoint = "./../../segmentanything/Scripts/sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
mask_generator = SamAutomaticMaskGenerator(sam)

blue_min, blue_max = np.array([105, 161, 108]), np.array([112, 255, 255])
green_min, green_max = np.array([38, 30, 133]), np.array([100, 164, 218])

def findVertex(mask: np.ndarray, max_iter = 20):
    """
    返回车牌区域 mask 的四个顶点
    """
    thresh = mask.astype(np.uint8) * 255
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    factor = 0.01
    for _ in range(max_iter):
        epsilon = factor * cv2.arcLength(contours[0], True)
        approx = cv2.approxPolyDP(cv2.convexHull(contours[0]), epsilon, True)
        if len(approx) == 4: return approx
        if len(approx) > 4: factor += 0.005
        else: factor -= 0.005
    raise Exception("beyound max_iter")

def segment(image: np.ndarray):
    """
    返回 image 中车牌区域的 mask
    """
    masks = mask_generator.generate(image)
    max_img, max_val, max_idx, max_part, max_mask = None, 0, None, None, None
    for idx, ann in enumerate(sorted(masks, key=(lambda x: x['area']), reverse=True)):
        part = np.sum(ann['segmentation'] != 0) / (ann['segmentation'].shape[0] * ann['segmentation'].shape[1])
        if part < 0.005: continue
        mask_img = np.where(ann['segmentation'][:,:,None][:,:,(0,0,0)], image, 0)
        mask_hsv = cv2.cvtColor(mask_img, cv2.COLOR_BGR2HSV)
        val = max(
            (np.sum(cv2.inRange(mask_hsv, blue_min, blue_max) != 0) / np.sum(ann['segmentation'] != 0)),
            (np.sum(cv2.inRange(mask_hsv, green_min, green_max) != 0) / np.sum(ann['segmentation'] != 0))
        )
        if val > 0.7: continue

        try:findVertex(ann['segmentation'])
        except:continue

        if val > max_val:
            max_val = val
            max_img = mask_img
            max_idx = idx
            max_part = part
            max_mask = ann['segmentation']
    return max_mask

def sort_points(points: list):
    """
    对四边形的四个顶点按照（左上、右上、右下、左下）的顺序进行排序
    """
    tmp = sorted(points)
    left, right = tmp[:2], tmp[2:]
    left.sort(key=lambda x:x[1])
    right.sort(key=lambda x:x[1])
    return [left[0], right[0], right[1], left[1]]

def locate_char(image_: np.ndarray):
    """
    返回 image 中非省份的全部字符区域 (x, y, w, h)
    """
    image = image_[:, 100:]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    blue_val = np.sum(cv2.inRange(image_hsv, np.array([105, 161, 108]), np.array([112, 255, 255])) != 0)
    green_val = np.sum(cv2.inRange(image_hsv, np.array([38, 30, 133]), np.array([100, 164, 218])) != 0)
    blocksize = min(image.shape[:2])
    if not blocksize & 1: blocksize -= 1
    if blue_val > green_val:
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blocksize, -30)
    else:
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, blocksize, 30)
    opend = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    contours, hierarchy = cv2.findContours(opend, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ret = []
    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
        if w > 100 and h > 150 and w < 200 and 20000 < w * h:
            ret.append((x + 100, y, w, h))
    return ret

if __name__ == 'segment':
    for impath in (
        './resources/images/easy/1-1.jpg',
        './resources/images/easy/1-2.jpg',
        './resources/images/easy/1-3.jpg',
        './resources/images/medium/2-1.jpg',
        './resources/images/medium/2-2.jpg',
        './resources/images/medium/2-3.jpg',
        './resources/images/difficult/3-1.jpg',
        './resources/images/difficult/3-2.jpg',
        './resources/images/difficult/3-3.jpg'
    ):
        image0 = cv2.imread(impath)
        image = cv2.resize(image0, (0,0), fx=0.5, fy=0.5)
        mask = segment(image)

        vertexs = findVertex(mask) * 2
        # cv2.drawContours(im, [findVertex(mask)], -1, (255, 0, 0), 2)
        # plt.imshow(im)

        width, height = 1280, 300 # 车牌号的形状
        pts1 = np.array(sort_points(np.squeeze(vertexs).astype(np.float32).tolist()), dtype=np.float32)
        pts2 = np.float32([[0, 0],[width,0],[width, height],[0,height]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        dst = cv2.warpPerspective(image0, M, (width, height))
        
        print(impath[impath.rfind('/') + 1:])
        cv2.imwrite("./tmp/" +impath[impath.rfind('/') + 1:], dst)

if __name__ == '__main__':
    for impath in os.listdir('./tmp/'):
        dst = cv2.imread(os.path.join('tmp', impath))
        for (x, y, w, h) in locate_char(dst):
            cv2.rectangle(dst, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imwrite(impath, dst)