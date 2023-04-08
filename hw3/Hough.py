import cv2, numpy as np, os, json
from itertools import product

def hough(_rho, _theta, _threshold):
    img_lines = img.copy()
    lines = cv2.HoughLines(edges, _rho, _theta, _threshold)
    if lines is None: return img_lines, 0
    num_lines = len(lines)
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + l * (-b))
        y1 = int(y0 + l * (a))
        x2 = int(x0 - l * (-b))
        y2 = int(y0 - l * (a))
        cv2.line(img_lines, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return img_lines, num_lines

if __name__ == '__main__':
    result_path = "HoughResult"
    if not os.path.exists(result_path): os.makedirs(result_path)
    img = cv2.imread('lines.png')
    l = 2*max(img.shape)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 220, apertureSize=3)
    data_idx = {}
    for rho, _theta, threshold in product(range(1, 6), range(1, 6), range(170, 250, 20)):
        theta = _theta * np.pi / 180
        img_name = f"rho={rho};_theta={_theta};threshold={threshold}.jpg"
        img_lines, line_count = hough(rho, theta, threshold)
        cv2.imwrite(os.path.join(result_path, img_name), img_lines)
        data_idx[img_name] = line_count
    with open(os.path.join(result_path, "index.json"), "w", encoding="utf-8") as f:
        json.dump(data_idx, f, indent=4, ensure_ascii=False)