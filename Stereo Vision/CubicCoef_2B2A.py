# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 16:17:45 2022

@author: wuaki

Interpolation_2B2A
"""
from ctypes import cdll, c_int, c_double, POINTER

def Calculate(Cubic_Xinv, Length, img):
    import numpy as np
    Coef_2B2A = np.zeros((2*Length+1, 2*Length+1,16), dtype=float)
    Length = np.array([Length], dtype=int)
    img = img.astype('int')
    #============================ C ============================#
    # 載入SO 動態連結檔案:
    m = cdll.LoadLibrary('./CubicCoef_2B2A.so')
    # 設定 SO 檔案中 CubicCoef 函數的參數資料型態:
    m.CubicCoef.argtypes = [POINTER(c_double), POINTER(c_int),\
                            POINTER(c_int), POINTER(c_double)]
    # 設定 SO 檔案中 CubicCoef 函數的傳回值資料型態
    #m.CubicCoef.restype = c_int #似乎可以不設定
    # 取得陣列指標 4個
    Cubic_Xinv_Ptr = Cubic_Xinv.ctypes.data_as(POINTER(c_double))
    Length_Ptr = Length.ctypes.data_as(POINTER(c_int))
    img_Ptr = img.ctypes.data_as(POINTER(c_int))
    Coef_2B2A_Ptr = Coef_2B2A.ctypes.data_as(POINTER(c_double))
    # 呼叫 SO 檔案中的 CubicCoef 函數 
    m.CubicCoef(Cubic_Xinv_Ptr, Length_Ptr, img_Ptr, Coef_2B2A_Ptr)
    #===========================================================#
    return Coef_2B2A