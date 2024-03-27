# -*- coding: utf-8 -*-
"""
Created on Sun Aug 21 14:01:22 2022

@author: Andy
"""
import numpy as np
from scipy.linalg import solve
from sympy import symbols, solve
# ========= ( AX = B )===========

# (6,6,0):x、(6,6,1):y、(6,6,2):z、
def normalVector(coorZone, SL):
    # 
    A = []
    for i in range(0,SL,1):
        for j in range(0,SL,1):
            A = np.append(A,coorZone[i][j][0]) #coor_x
            A = np.append(A,coorZone[i][j][1]) #coor_y
            A = np.append(A,coorZone[i][j][2]) #coor_z

    # 將一維陣列coefMatrix轉為SL*SL*3矩陣
    A = A.reshape(SL*SL,3)
    # B matrix
    B = np.ones(SL*SL) # column vector
    # Solve X by least square method
    X = np.linalg.lstsq(A, B, rcond=None)
    
    coef = X[0]
    return coef
    

def project(point3d, nVector):
    t = symbols('t')
    temp = nVector[0]*(point3d[0]+t*nVector[0])+\
           nVector[1]*(point3d[1]+t*nVector[1])+\
           nVector[2]*(point3d[2]+t*nVector[2])
    # Solve t
    sol = solve(temp-1)
    point3d_new = point3d + sol*nVector
    return point3d_new


# p1 = np.array((0, 0, 1), dtype=float)
# p2 = np.array((0, 1, 0.8), dtype=float)
# p3 = np.array((1, 0, 1.1), dtype=float)
# p4 = np.array((1, 1, 0.9), dtype=float)
# p5 = np.array((1, 2, 1), dtype=float)
# p6 = np.array((2, 1, 1), dtype=float)

# A = np.array(((0, 0, 1),\
#              (0, 1, 0.8),\
#              (1, 0, 1.1),\
#              (1, 1, 0.9),\
#              (1, 2, 1),\
#              (2, 1, 1)), dtype=float)
    
# B = np.array((1,1,1,1,1,1), dtype=float)

# X = np.linalg.lstsq(A, B, rcond=None)
    
   