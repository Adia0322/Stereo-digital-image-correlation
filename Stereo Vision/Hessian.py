# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 18:30:41 2022

@author: wuaki
"""

def Calculate(Size, IGrad_u, IGrad_v):
    import numpy as np
    # Half of Size length
    Len = int(0.5*(Size-1))
    # Hessian matrix
    H = np.zeros((6,6), dtype=int)
    # Jacobiab of reference subset warp_function (dW_dP)
    dW_dP = np.array([(1, -Len, -Len, 0, 0, 0),\
                      (0, 0, 0, 1, -Len, -Len)], dtype=int)
    # Image gradient (F: computed using Sobel operator)
    dF = np.zeros((1, 2), dtype=int)
    # Storage zone of Jacobiab of reference subset (F*dW_dP)
    J = np.zeros((Size,Size,6), dtype=float) 
    # Compute jacobian of reference subset point(F*dW_dP), and then compute hessian matrix.
    for i in range(0,Size,1):
        for j in range(0,Size,1):
            dF[0][0] = IGrad_u[i][j] # x
            dF[0][1] = IGrad_v[i][j] # y
            dW_dP[0][1] = i-Len   #np.array([(1, -Len+j, -Len+i, 0, 0, 0), (0, 0, 0, 1, -Len+j, -Len+i)], dtype=int)
            dW_dP[0][2] = j-Len
            dW_dP[1][4] = i-Len
            dW_dP[1][5] = j-Len
            # Jacobian matrix: J
            J_TEMP = dF.dot(dW_dP)
            J[i][j][:] = J_TEMP
            H = np.transpose(J_TEMP).dot(J_TEMP) + H
    # inverse of matrix H
    H_inv = np.linalg.inv(H)
    
    return H_inv, J
    
    
    
    
    
    