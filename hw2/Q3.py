import numpy as np

def stereoRectify(A1, A2, RT1, RT2, dims1, dims2):
    P1, P2 = A1.dot(RT1), A2.dot(RT2)
    X = [np.vstack((P1[1,:], P1[2,:])), np.vstack((P1[2,:], P1[0,:])), np.vstack((P1[0,:], P1[1,:]))]
    Y = [np.vstack((P2[1,:], P2[2,:])), np.vstack((P2[2,:], P2[0,:])), np.vstack((P2[0,:], P2[1,:]))]
    F = np.array([[np.linalg.det(np.vstack((X[j], Y[i]))) for j in range(3)] for i in range(3)])
    if np.all(np.equal(F/F[2,1], np.array([[0,0,0],[0,0,-1],[0,1,0]]))): w1 = w2 = np.array([0,0,1])
    else:
        bv = np.linalg.inv(RT2[:,:3]).dot(RT2[:,3]) - np.linalg.inv(RT1[:,:3]).dot(RT1[:,3])
        B = (bv.dot(bv) * np.eye(3) - bv[:,np.newaxis].dot(bv[np.newaxis,:])).dot(np.linalg.inv(A1.dot(RT1[:,:3])))
        L1 = np.transpose(np.linalg.inv(A1.dot(RT1[:,:3]))).dot(B)
        L2 = np.transpose(np.linalg.inv(A2.dot(RT2[:,:3]))).dot(B)
        P1 = (dims1[0]*dims1[1]/12)*np.array([[dims1[0]**2 - 1, 0, 0],[0, dims1[1]**2 - 1,0],[0, 0, 0]])
        Pc1 = np.array([[(dims1[0] - 1)**2/4, (dims1[0] - 1)*(dims1[1] - 1)/4, (dims1[0] - 1)/2],
                        [(dims1[0] - 1)*(dims1[1] - 1)/4, (dims1[1] - 1)**2/4, (dims1[1] - 1)/2],
                        [(dims1[0] - 1)/2, (dims1[1] - 1)/2, 1]])
        P2 = (dims2[0]*dims2[1]/12)*np.array([[dims2[0]**2 - 1, 0, 0],[0, dims2[1]**2 - 1,0],[0, 0, 0]])
        Pc2 = np.array([[(dims2[0] - 1)**2/4, (dims2[0] - 1)*(dims2[1] - 1)/4, (dims2[0] - 1)/2],
                        [(dims2[0] - 1)*(dims2[1] - 1)/4, (dims2[1] - 1)**2/4, (dims2[1] - 1)/2],
                        [(dims2[0] - 1)/2, (dims2[1] - 1)/2, 1]])
        M1 = L1.T.dot(P1).dot(L1)
        C1 = L1.T.dot(Pc1).dot(L1)
        M2 = L2.T.dot(P2).dot(L2)
        C2 = L2.T.dot(Pc2).dot(L2)
        m = [M1[1,2]*C1[1,2] - M1[2,2]*C1[1,1], M1[1,1]*C1[1,2] - M1[1,2]*C1[1,1]]
        if np.all(np.equal(RT1[:,:3], RT2[:,:3])) and np.all(
            np.equal(A1, A2)) and np.all(np.equal(P1, P2)) and np.all(np.equal(Pc1, Pc2)): sol = [-m[0]/m[1]]
        else:
            m += [C2[1,2]/C2[1,1], C2[1,1]/C1[1,1], 
                  M2[1,2]*C2[1,2] - M2[2,2]*C2[1,1], 
                  M2[1,1]*C2[1,2] - M2[1,2]*C2[1,1], C1[1,2]/C1[1,1], 1/(C2[1,1]/C1[1,1])]
            alpha = [m[1]*m[3] + m[5]*m[7], m[0]*m[3] + 3*m[1]*m[2]*m[3] + m[4]*m[7] + 3*m[5]*m[6]*m[7],
                3*(m[0]*m[2]*m[3] + m[1]*m[2]**2*m[3] + m[4]*m[6]*m[7] + m[5]*m[6]**2*m[7]),
                3*m[0]*m[2]**2*m[3] + m[1]*m[2]**3*m[3] + 3*m[4]*m[6]**2*m[7] + m[5]*m[6]**3*m[7],
                m[0]*m[2]**3*m[3] + m[4]*m[6]**3*m[7]]
            beta = [(8*alpha[0]*alpha[2] - 3 * alpha[1]**2) / (8 * alpha[0]**2),
                12*alpha[0]*alpha[4] - 3*alpha[1]*alpha[3] + alpha[2]**2,
                27*alpha[0]*alpha[3]**2 - 72*alpha[0]*alpha[2
            ]*alpha[4] + 27*alpha[1]**2*alpha[4] - 9*alpha[1
            ]*alpha[2]*alpha[3] + 2*alpha[2]**3]
            Q = (1/2) * (-(2/3)*beta[0] + 1/(3*alpha[0]) * ((D0 := np.power(
                (1/2)*(beta[2]+(beta[2]**2 - 4*beta[1]**3) ** 0.5), 1/3)) + beta[1] / D0)) ** 0.5
            S = (8*alpha[0]**2*alpha[3] - 4*alpha[0]*alpha[1]*alpha[2] + alpha[1]**3) / (8*alpha[0]**3)
            sol = ([-alpha[1] / (4*alpha[0]) - Q - (1/2)*(-4*Q**2 - 2*beta[0] + S/Q) ** 0.5,
                 -alpha[1] / (4*alpha[0]) - Q + (1/2)*(
                -4*Q**2 - 2*beta[0] + S/Q) ** 0.5] if -4*Q**2 - 2*beta[0] + S/Q >= 0 else []) + (
                [-alpha[1] / (4*alpha[0]) + Q - (1/2)*(
                -4*Q**2 - 2*beta[0] - S/Q) ** 0.5,
                 -alpha[1] / (4*alpha[0]) + Q + (1/2)*(
                -4*Q**2 - 2*beta[0] - S/Q) ** 0.5] if -4*Q**2 - 2*beta[0] - S/Q >= 0 else [])
        w1, w2 = ((tmpfunc := lambda ss: (((tmp := (Rnew := np.array([
            (xv := bv / np.linalg.norm(bv)),
            (yv := (yv := np.cross(
            (zv := (p1w := np.linalg.inv(RT1[:,:3]).dot(
            np.linalg.inv(A1).dot(np.array(
            [0,ss,1])) - RT1[:,3])) - ((p1w + np.linalg.inv(RT2[:,:3]).dot(RT2[:,3])
            ).dot(xv) * xv - np.linalg.inv(RT2[:,:3]).dot(RT2[:,3]))), bv)) / np.linalg.norm(yv)),
            (zv := zv / np.linalg.norm(zv))])).dot(np.linalg.inv(A1.dot(RT1[:,:3])))[2,:]) / tmp[2]),
            ((tmp := Rnew.dot(np.linalg.inv(A2.dot(RT2[:,:3])))[2,:]) / tmp[2]))))(min(zip(sol,
            map(lambda x: [w := tmpfunc(x), float(w[0].dot(P1).dot(w[0])/w[0].dot(Pc1).dot(w[0])) + float(
            w[1].dot(P2).dot(w[1])/w[1].dot(Pc2).dot(w[1]))][1], sol)), key=lambda x:x[1])[0])
    vc2 = -min(
        min(((tmp := (Hp1 := np.array([[1,0,0], [0,1,0], w1])).dot(np.array([[0],[0],[1]]))[:,0]) / tmp[2])[1],
            ((tmp := Hp1.dot(np.array([[dims1[0]-1],[0],[1]]))[:,0]) / tmp[2])[1],
            ((tmp := Hp1.dot(np.array([[dims1[0]-1],[dims1[1]-1],[1]]))[:,0]) / tmp[2])[1],
            ((tmp := Hp1.dot(np.array([[0],[dims1[1]-1],[1]]))[:,0]) / tmp[2])[1]),
        min(((tmp := (Hp2 := np.array([[1,0,0], [0,1,0], w2])).dot(np.array([[0],[0],[1]]))[:,0]) / tmp[2])[1],
            ((tmp := Hp2.dot(np.array([[dims2[0]-1],[0],[1]]))[:,0]) / tmp[2])[1],
            ((tmp := Hp2.dot(np.array([[dims2[0]-1],[dims2[1]-1],[1]]))[:,0]) / tmp[2])[1],
            ((tmp := Hp2.dot(np.array([[0],[dims2[1]-1],[1]]))[:,0]) / tmp[2])[1]))
    return F, np.array([[
        ((dims1[1] *
          (x := ((tmp := (Hrp1 := np.array([ [F[2,1]-w1[1]*F[2,2], w1[0]*F[2,2]-F[2,0], 0],
                     [w1[0]*F[2,2]-F[2,0], w1[1]*F[2,2]-F[2,1], -(F[2,2] + vc2)],
                     [0, 0, 1] ]).dot(Hp1)).dot([(dims1[0] - 1),
            (dims1[1] - 1) / 2, 1])) / tmp[2]) -
           ((tmp := Hrp1.dot([0, (dims1[1] - 1) / 2, 1])) / tmp[2]))[1])**2 +
         (dims1[0] *
          (y := ((tmp := Hrp1.dot([(dims1[0] - 1) / 2,
            (dims1[1] - 1), 1])) / tmp[2]) -
           ((tmp := Hrp1.dot([(dims1[0] - 1) / 2, 0, 1])) / tmp[2]))[1])**2) /
        (dims1[0] * dims1[1] * (x[1] * y[0] - x[0] * y[1])),
        ((dims1[1]**2) * x[0] * x[1] + (dims1[0]**2) * y[0] * y[1]) /
        (dims1[0] * dims1[1] * (x[0] * y[1] - x[1] * y[0])), 0
    ], [0, 1, 0], [0, 0, 1]]).dot(Hrp1), np.array([[
        ((dims2[1] *
          (x := ((tmp := (Hrp2 := np.array([ [F[1,2]-w2[1]*F[2,2], w2[0]*F[2,2]-F[0,2], 0],
                     [F[0,2]-w2[0]*F[2,2], F[1,2]-w2[1]*F[2,2], vc2],
                     [0, 0, 1] ]).dot(Hp2)).dot([(dims2[0] - 1),
            (dims2[1] - 1) / 2, 1])) / tmp[2]) -
           ((tmp := Hrp2.dot([0, (dims2[1] - 1) / 2, 1])) / tmp[2]))[1])**2 +
         (dims2[0] *
          (y := ((tmp := Hrp2.dot([(dims2[0] - 1) / 2,
            (dims2[1] - 1), 1])) / tmp[2]) -
           ((tmp := Hrp2.dot([(dims2[0] - 1) / 2, 0, 1])) / tmp[2]))[1])**2) /
        (dims2[0] * dims2[1] * (x[1] * y[0] - x[0] * y[1])),
        ((dims2[1]**2) * x[0] * x[1] + (dims2[0]**2) * y[0] * y[1]) /
        (dims2[0] * dims2[1] * (x[0] * y[1] - x[1] * y[0])), 0
    ], [0, 1, 0], [0, 0, 1]]).dot(Hrp2)

if __name__ == "__main__":
    import cv2
    img1 = cv2.imread("./data/imgs/leftcamera/Im_L_1.png")
    img2 = cv2.imread("./data/imgs/rightcamera/Im_R_1.png")
    dims1 = img1.shape[::-1][1:]
    dims2 = img2.shape[::-1][1:]

    data = np.load("./data/out/parameters.npz")
    A1 = data["L_Intrinsic"]
    A2 = data["R_Intrinsic"]
    RT1 = data["L_Extrinsics"][0][:-1]
    RT2 = data["R_Extrinsics"][0][:-1]

    distCoeffs1 = np.array([])
    distCoeffs2 = np.array([])

    F, Rectify1, Rectify2 = stereoRectify(A1, A2, RT1, RT2, dims1, dims2)

    # lpt, rpt = np.array([161, 123, 1]), np.array([373, 104, 1])
    # (a1, b1, c1), (a2, b2, c2) = F @ lpt, F @ rpt
    # cv2.line(img1, (0, round((-c1-a1*0)/b1)), (2000, round((-c1-a1*2000)/b1)), (0, 255, 0), 2)
    # cv2.line(img2, (0, round((-c2-a2*0)/b2)), (2000, round((-c2-a2*2000)/b2)), (0, 255, 0), 2)

    tL1, tR1, bR1, bL1 = [(x,y) for x, y in np.squeeze(cv2.undistortPoints(np.array([
        [[0,0]], [[dims1[0]-1,0]], [[dims1[0]-1,dims1[1]-1]], [[0, dims1[1]-1]]
    ], dtype=np.float32), A1, np.zeros(5) if distCoeffs1 is None else distCoeffs1, R=Rectify1.dot(A1)))]
    tL2, tR2, bR2, bL2 = [(x,y) for x, y in np.squeeze(cv2.undistortPoints(np.array([
        [[0,0]], [[dims2[0]-1,0]], [[dims2[0]-1,dims2[1]-1]], [[0, dims2[1]-1]]
    ], dtype=np.float32), A2, np.zeros(5) if distCoeffs2 is None else distCoeffs2, R=Rectify2.dot(A2)))]
    minX1, minX2 = min(tR1[0], bR1[0], bL1[0], tL1[0]), min(tR2[0], bR2[0], bL2[0], tL2[0])
    maxX1, maxX2 = max(tR1[0], bR1[0], bL1[0], tL1[0]), max(tR2[0], bR2[0], bL2[0], tL2[0])
    minY = min(tR2[1], bR2[1], bL2[1], tL2[1], tR1[1], bR1[1], bL1[1], tL1[1])
    maxY = max(tR2[1], bR2[1], bL2[1], tL2[1], tR1[1], bR1[1], bL1[1], tL1[1])
    flipX = -1 if tL1[0] > tR1[0] else 1
    flipY = -1 if tL1[1] > bL1[1] else 1
    scaleX, scaleY = flipX * dims1[0] / (maxX2 - minX2
    ) if maxX2 - minX2 > maxX1 - minX1 else flipX * dims1[0]/(maxX1 - minX1), flipY * dims1[1] / (maxY - minY)
    Fit = np.array([
        [scaleX, 0, -(min(minX1, minX2) if flipX == 1 else min(maxX1, maxX2)) * scaleX], 
        [0, scaleY, -minY * scaleY if flipY == 1 else -maxY * scaleY], 
        [0, 0, 1]])

    mapx1, mapy1 = cv2.initUndistortRectifyMap(A1, distCoeffs1, Rectify1.dot(A1), Fit, dims1, cv2.CV_32FC1)
    mapx2, mapy2 = cv2.initUndistortRectifyMap(A2, distCoeffs2, Rectify2.dot(A2), Fit, dims1, cv2.CV_32FC1)
    img1_rect = cv2.remap(img1, mapx1, mapy1, interpolation=cv2.INTER_LINEAR)
    img2_rect = cv2.remap(img2, mapx2, mapy2, interpolation=cv2.INTER_LINEAR)
    rectImgs = np.hstack((img1_rect, img2_rect))

    cv2.imwrite("result.png", rectImgs)
