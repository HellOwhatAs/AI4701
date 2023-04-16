import numpy as np
import cv2, os
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import cnn_chr_model, cnn_provi_model, torch
from PIL import Image

sam_checkpoint = "./../../segmentanything/Scripts/sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
mask_generator = SamAutomaticMaskGenerator(sam)

net = cnn_chr_model.Net().to(device)
net.load_state_dict(torch.load(cnn_chr_model.PATH))

provi_net = cnn_provi_model.Net().to(device)
provi_net.load_state_dict(torch.load(cnn_provi_model.PATH))

blue_min, blue_max = np.array([105, 161, 108]), np.array([112, 255, 255])
green_min, green_max = np.array([38, 30, 133]), np.array([100, 164, 218])
width, height = 1280, 300 # 车牌号的形状

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

def getbinary(image: np.ndarray):
    """
    根据 (蓝/绿) 颜色，进行二值化
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    blue_val = np.sum(cv2.inRange(image_hsv, blue_min, blue_max) != 0)
    green_val = np.sum(cv2.inRange(image_hsv, green_min, green_max) != 0)
    blocksize = min(image.shape[:2])
    if not blocksize & 1: blocksize -= 1
    if blue_val > green_val:
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blocksize, -30)
    else:
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, blocksize, 30)
    return binary

def locate_char(binary: np.ndarray):
    """
    返回 binary image 中非省份的全部字符区域 (x, y, w, h)
    """
    binary = binary[:, 100:]
    opend = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    contours, hierarchy = cv2.findContours(opend, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ret = []
    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
        if w > 100 and h > 150 and w < 200 and 20000 < w * h:
            ret.append((x + 100, y, w, h))
    return sorted(ret)

def locate_provi(chrs: list):
    """
    返回 根据其他字符区域 chrs [(x, y, w, h)] 确定的省份汉字区域 (x, y, w, h)
    """
    chrs.sort()
    dx, dy, dw, dh = np.mean([np.array(chrs[i + 1]) - chrs[i] for i in range(len(chrs) - 1)][1:], dtype = int, axis=0)
    c_x, c_y = chrs[0][0] - dx, chrs[0][1] - dy
    c_w, c_h = chrs[0][2] - dw, chrs[0][3] - dh
    return max(0, c_x - 10), max(0, c_y), c_w + 20, min(c_h, height - c_y)

def fill_rect(img: np.ndarray):
    h, w = img.shape[:2]
    padding = abs(h - w) // 2
    tmp = 10
    if h > w: img = cv2.copyMakeBorder(img, tmp, tmp, padding + tmp, padding + tmp, cv2.BORDER_CONSTANT,value=0)
    else: img = cv2.copyMakeBorder(img, padding + tmp, padding + tmp, tmp, tmp, cv2.BORDER_CONSTANT,value=0)
    return img

def plate_recognition(image0: np.ndarray, show_intermediate = False):
    scale = 2000 / image0.shape[1]
    image0 = cv2.resize(image0, (0, 0), fx = scale, fy = scale)
    image = cv2.resize(image0, (0,0), fx=0.5, fy=0.5)
    mask = segment(image)

    vertexs = findVertex(mask) * 2

    if show_intermediate:
        im = image.copy()
        cv2.drawContours(im, [findVertex(mask)], -1, (0, 0, 255), 2)
        yield cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    
    pts1 = np.array(sort_points(np.squeeze(vertexs).astype(np.float32).tolist()), dtype=np.float32)
    pts2 = np.float32([[0, 0],[width,0],[width, height],[0,height]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(image0, M, (width, height))
    
    if show_intermediate:
        yield cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)

        

    dst_copy = dst.copy()
    binary = getbinary(dst_copy)
    chrs, chr_images = [], []
    for idx, (x, y, w, h) in enumerate(locate_char(binary)):
        chr_images.append(fill_rect(binary[y: y + h, x: x + w]))
        
        if show_intermediate:
            cv2.rectangle(dst_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)

        chrs.append((x, y, w, h))
    
    chr_images_input = torch.stack([cnn_chr_model.transform(Image.fromarray(im).convert('RGB')) for im in chr_images], dim=0)
    output = net(chr_images_input.to(device))
    _, predicted = torch.max(output, 1)

    chr_result = [cnn_chr_model.int2chr[i.item()] for i in predicted]

    x, y, w, h = locate_provi(chrs)
    provi_img = fill_rect(binary[y: y + h, x: x + w])

    if show_intermediate:
        cv2.rectangle(dst_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
        yield cv2.cvtColor(dst_copy, cv2.COLOR_BGR2RGB)
        yield provi_img
        for i in chr_images:
            yield i

    provi_image_input = torch.stack([cnn_chr_model.transform(Image.fromarray(provi_img).convert('RGB'))], dim=0)
    provi_output = provi_net(provi_image_input.to(device))
    _, provi_predicted = torch.max(provi_output, 1)
    provi_result = cnn_provi_model.provi2zh[cnn_provi_model.int2provi[provi_predicted.item()]]

    ret = provi_result + chr_result[0] + ' ' + "".join(chr_result[1:])
    yield ret

def result_only(image: np.ndarray):
    try: return list(plate_recognition(image))[0]
    except: return '识别失败了！'

if __name__ == '__main__':
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
        print(result_only(cv2.imread(impath)))