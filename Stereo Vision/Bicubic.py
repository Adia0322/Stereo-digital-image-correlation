# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 13:59:25 2022
@author: wuaki
"""

def Bicubic_int(u, v, coefficient):
    import numpy as np
    
    U = np.array([1, u, u*u, u*u*u], dtype=float)
    V = np.array([1, v, v*v, v*v*v], dtype=float)
    
    gray_value = U.dot(coefficient.dot(np.transpose(V))) # U*coefficient*V
    return gray_value
