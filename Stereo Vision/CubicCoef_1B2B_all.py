# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 16:35:30 2022

@author: wuaki
"""
# 此算法有冗贅多餘的計算，但可以允許外部給定不同的Size，並且易於理解。
# 第二個引數Size表示輸出之coef矩陣之邊長
# Constant coefficient
import numpy as np
Cubic_X = np.array([(1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1),\
                  (1, 0, 0, 0, -1, 0, 0, 0, 1, 0, 0, 0, -1, 0, 0, 0),\
                  (1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1),\
                  (1, 2, 4, 8, -1, -2, -4, -8, 1, 2, 4, 8, -1, -2, -4, -8),\
                  (1, -1, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),\
                  (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),\
                  (1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),\
                  (1, 2, 4, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),\
                  (1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1),\
                  (1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0),\
                  (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),\
                  (1, 2, 4, 8, 1, 2, 4, 8, 1, 2, 4, 8, 1, 2, 4, 8),\
                  (1, -1, 1, -1, 2, -2, 2, -2, 4, -4, 4, -4, 8, -8, 8, -8),\
                  (1, 0, 0, 0, 2, 0, 0, 0, 4, 0, 0, 0, 8, 0, 0, 0),\
                  (1, 1, 1, 1, 2, 2, 2, 2, 4, 4, 4, 4, 8, 8, 8, 8),\
                  (1, 2, 4, 8, 2, 4, 8, 16, 4, 8, 16, 32, 8, 16, 32, 64)], dtype=int)
Cubic_Xinv = np.linalg.inv(Cubic_X)

def Calculate(img, Size):
    import numpy as np
    # Interpolation area Size*Size
    coef = np.zeros((Size,Size,16), dtype=float)
    for i in range(0,Size,1):
        for j in range(0,Size,1):
            Gvalue=img[i:i+4, j:j+4]
            Gvalue = np.reshape(Gvalue, (16,1), order='F')
            coef[i, j, ...] = np.transpose(Cubic_Xinv.dot(Gvalue))
    return coef, Cubic_Xinv


def CalculateALL(img, row, col):
    import numpy as np
    coef = np.zeros((row,col,16), dtype=float)
    for i in range(0,row,1):
        for j in range(0,col,1):
            Gvalue=img[i:i+4, j:j+4]
            Gvalue = np.reshape(Gvalue, (16,1), order='F')
            coef[i, j, ...] = np.transpose(Cubic_Xinv.dot(Gvalue))
    return coef, Cubic_Xinv



