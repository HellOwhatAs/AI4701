from math import *
import cv2, numpy as np, os

def r(val):
    return int(np.random.random() * val)

def rot(img,angel,shape,max_angel):
    size_o = [shape[1],shape[0]]
    size = (shape[1]+ int(shape[0]*sin((float(max_angel )/180) * 3.14)),shape[0])
    interval = abs( int( sin((float(angel) /180) * 3.14)* shape[0]))
    pts1 = np.float32([[0,0],[0,size_o[1]],[size_o[0],0],[size_o[0],size_o[1]]])
    if(angel>0):
        pts2 = np.float32([[interval,0],[0,size[1]  ],[size[0],0  ],[size[0]-interval,size_o[1]]])
    else:
        pts2 = np.float32([[0,0],[interval,size[1]  ],[size[0]-interval,0  ],[size[0],size_o[1]]])
    M  = cv2.getPerspectiveTransform(pts1,pts2)
    dst = cv2.warpPerspective(img,M,size)
    return dst

def rotRandrom(img, factor, size):
    shape = size
    pts1 = np.float32([[0, 0], [0, shape[0]], [shape[1], 0], [shape[1], shape[0]]])
    pts2 = np.float32([[r(factor), r(factor)],
                        [ r(factor), shape[0] - r(factor)],
                        [shape[1] - r(factor), r(factor)],
                        [shape[1] - r(factor), shape[0] - r(factor)]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(img, M, size)
    return dst

com0 = cv2.imread('./6.jpg', 0)

for _ in range(len(os.listdir('./data/6/'))):
    com = rot(com0, r(20)-10, com0.shape, 10)
    com = rotRandrom(com, 5, (com.shape[1], com.shape[0]))
    cv2.imwrite(f'./data/6/new_energy_6-{_}.jpg', com)