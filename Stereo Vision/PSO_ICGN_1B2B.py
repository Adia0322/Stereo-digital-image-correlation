# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 16:32:26 2022

@author: wuaki
注意: 執行前請先確認 Size Scan 在python c 皆一致，否則無法執行卻沒有錯誤信息!!
注意: 不同圖片尺寸需要在c程式裡修改 img_cloumn 的大小 !!
"""

def Calculate_1B2B(img_1B, img_2B, C1_B_x, C1_B_y,\
                   Size_1B2B, Size_2B2A, Scan_1B2B, H_inv_1B2B,\
                   J_1B2B, Cubic_coef_1B2B, Trans1B2B):
    import numpy as np
    from ctypes import cdll, c_int, c_double, POINTER
    import Bicubic_test1
    import cv2 as cv
    """ =============== Initial setting =============== """
    # 取得圖片尺寸
    #ROW, COL = img_1B_GRAY.shape # 注意 ROW,COL不會存到變數...
    # 設定子矩陣大小(邊長) 需要是奇數!!
    Size = Size_1B2B
    # 設定掃瞄方陣之邊長
    Scan = Scan_1B2B 
    # 設定插值方陣之邊長 (在主程式已經有算了，為了不再增加函式變數因此重算一遍)
    Length = int(0.5*(Size-1)+0.5*(Scan-1))
    # 子集合之半邊長
    Len = int(0.5*(Size-1)) 

    # 將 img_aft 轉為int32型態 (不確定需不需要?!!)
    img_bef = np.array(img_1B, dtype=int)
    img_aft = np.array(img_2B, dtype=int)  #似乎不需要??   !!!!!!!!!!!!!!

    # 建立位移暫存區
    Displacement = np.zeros((2,), dtype=int) # 依序為 [u, v]
    # 係數index、CoefValue
    CoefValue = np.zeros((2,), dtype=float)
    # 所選取目標點的位置 
    Object_point = np.array((C1_B_y,C1_B_x-Trans1B2B), dtype=int)
    # 預先在影像裡搜尋起點平移一個距離Trans1B2B: 以利快速搜尋目標點

    # 將 img_1B_sub 轉為int32型態 (不確定需不需要?!!)
    # 建構變形前後影像之子矩陣: img_bef_sub
    img_bef_sub = img_bef[C1_B_y-Len:C1_B_y+Len+1,\
                          C1_B_x-Len:C1_B_x+Len+1]  

    # Reference subset (undeformed subset)
    Mean_bef = np.array(np.mean(img_bef_sub), dtype=float)
    img_bef_sub = img_bef_sub.astype(int)     # 將float轉int

    # Target subset (deformed subset)
    img_aft_sub = np.zeros((Size,Size), dtype=int)
    
    """ =============== Compute integer displacement ==============="""
    #============================ 使用C計算位移 ============================#
    # 載入SO 動態連結檔案: test_2D_DIC_displacement.so
    m = cdll.LoadLibrary('./PSO_ICGN_1B2B.so')

    # 設定 SO 檔案中 SCAN 函數的參數資料型態:
    m.SCAN.argtypes = [POINTER(c_int), POINTER(c_int), POINTER(c_int),\
                       POINTER(c_double), POINTER(c_int), POINTER(c_double),\
                       POINTER(c_double)]

    # 設定 SO 檔案中 SCAN 函數的傳回值資料型態
    m.SCAN.restype = c_int #似乎可以不設定

    # 取得陣列指標 7個
    img_aft_Ptr = img_aft.ctypes.data_as(POINTER(c_int))
    img_aft_sub_Ptr = img_aft_sub.ctypes.data_as(POINTER(c_int))
    img_bef_sub_Ptr = img_bef_sub.ctypes.data_as(POINTER(c_int))
    Mean_bef_Ptr = Mean_bef.ctypes.data_as(POINTER(c_double))
    Object_point_Ptr = Object_point.ctypes.data_as(POINTER(c_int))
    Displacement_Ptr = Displacement.ctypes.data_as(POINTER(c_double))
    CoefValue_Ptr = CoefValue.ctypes.data_as(POINTER(c_double))                        

    # 呼叫 SO 檔案中的 SCAN 函數 
    m.SCAN(img_aft_Ptr, img_aft_sub_Ptr, img_bef_sub_Ptr,\
           Mean_bef_Ptr, Object_point_Ptr, Displacement_Ptr, CoefValue_Ptr)
        
    #=================== 位移計算完成 計算結果在Displacement =========================#

    #============== 數據 ===============#
    # u, v 座標
    #img_u = Displacement[0] + Object_point[0]
    #img_v = Displacement[1] + Object_point[1]

    # print("\nPSO_ICGN_1B2B整數位移(單位:pixels):")
    # print("垂直:", Displacement[0], "(注意:不包含Trans1B2B)")  #垂直以下為正
    # print("水平:", Displacement[1])  #水平以右為正
    # print("相關係數:", CoefValue[1])
    
    # Target subset point coordinate: (u2,v2)   
    #C2_B_y_int = Displacement[0] + C1_B_y  
    #C2_B_x_int = Displacement[1] + C1_B_x - Trans1B2B
    # (u2,v2) is used to define displacement vector: P
    
    # Integer displacement for subpixels algorithm
    Int_u = Displacement[0]
    Int_v = Displacement[1]

    
    """============================ Precomputation ==========================="""
    # define Incremental displacement vector: delta_P
    u_inc = 0.01
    ux_inc = 0.01
    uy_inc = 0.01
    v_inc = 0.01
    vx_inc = 0.01
    vy_inc = 0.01
    delta_P = np.array([u_inc, ux_inc, uy_inc, v_inc, vx_inc, vy_inc], dtype=float)

    # Incremental warp function coefficient
    warp_inc_coef = np.array([(1+ux_inc, uy_inc, u_inc),\
                              (vx_inc, 1+vy_inc, v_inc),\
                              (0, 0, 1)], dtype=float)
    
    # Reference subset (Simplied: delta_p = 0) !!!!!!!!
    Gvalue_ref_sub = img_bef_sub
    # Mean of Reference subset 
    f_average = np.mean(Gvalue_ref_sub)
    # Delta_f
    delta_f = np.sqrt(np.sum(np.square(Gvalue_ref_sub-f_average)))

    # define displacement vector: P
    u = Int_u # obtain by PSO
    ux = 0
    uy = 0
    v = Int_v # obtain by PSO
    vx = 0
    vy = 0
    # warp function coefficient of deformed subset
    warp_aft_coef = np.array([(1+ux, uy, u),\
                              (vx, 1+vy, v),\
                              (0, 0, 1)], dtype=float)
        
    """========================== Iteration ============================="""
    count = 0
    limit = np.sqrt(np.square(delta_P[0]) + np.square(delta_P[1]*Len)+\
                           np.square(delta_P[2]*Len) + np.square(delta_P[3])+\
                           np.square(delta_P[4]*Len) + np.square(delta_P[5]*Len))
    while limit > 0.0001 and count < 20:
        # Average gray value of deformed subset points(with interpolation)
        Gvalue_g = np.zeros((Size,Size), dtype=float)
        for i in range(0,Size,1):
            for j in range(0,Size,1):
                # Start to compute gray value of deformed subset
                Position = np.transpose(np.array([i-Len, j-Len, 1], dtype=float)) # compute 25*25 gray value
                warp_aft = warp_aft_coef.dot(Position)
                # Find the coefficient in lookup table
                a1 = Length + int(np.floor(warp_aft[0])) # 
                a2 = Length + int(np.floor(warp_aft[1])) #
                # A: cubic coefficient in a1, a2
                if a1>2*Length:
                    a1=2*Length
                    print("a1 is out of bound: a1>2*Length")
                if a1<0:
                    a1=0
                    print("a1 is out of bound: a1<0")
                if a2>2*Length:
                    a1=2*Length
                    print("a2 is out of bound: a2>2*Length")
                if a2<0:
                    a2=0
                    print("a2 is out of bound: a2<0")
                A = Cubic_coef_1B2B[a1][a2][:] # 加上Length後表示起點為子集合中心開始算
                A_re = np.reshape(A, (4,4), order='F')
                # Calculate interpolation and store in matrix Gvalue_g[][]
                Gvalue_g[i][j] = Bicubic_test1.Bicubic_int(warp_aft[0]-np.floor(warp_aft[0]),\
                                                           warp_aft[1]-np.floor(warp_aft[1]), A_re)

        # compute g_average
        g_average = np.mean(Gvalue_g)

        # compute delata_g
        delta_g = np.sqrt(np.sum(np.square(Gvalue_g-g_average)))
        
        Corelation_sum = 0
        dF_dP = (Gvalue_ref_sub - f_average) - (delta_f/delta_g)*(Gvalue_g - g_average)
        for i in range(0,Size,1):
            for j in range(0,Size,1):
                Corelation_sum += np.transpose(J_1B2B[i][j][:])*dF_dP[i][j]

        delta_P = -H_inv_1B2B.dot(Corelation_sum)
        # Update limit (if limit is enough small, then quit)
        limit = np.sqrt(np.square(delta_P[0]) + np.square(delta_P[1]*Len)+\
                        np.square(delta_P[2]*Len) + np.square(delta_P[3])+\
                        np.square(delta_P[4]*Len) + np.square(delta_P[5]*Len))
        
        warp_inc_coef = np.array([(1+delta_P[1], delta_P[2], delta_P[0]),\
                                  (delta_P[4], 1+delta_P[5], delta_P[3]),\
                                  (0, 0, 1)], dtype=float)
        
        warp_inc_coef_inv = np.linalg.inv(warp_inc_coef)
        # 更新warp function
        warp_aft_coef = warp_aft_coef.dot(warp_inc_coef_inv)
        # 計次數
        count += 1
    
    # Subpixel displacement
    U = warp_aft_coef[0][2] # 垂直
    V = warp_aft_coef[1][2] # 水平
    C2_B_y = U + C1_B_y
    C2_B_x = V + C1_B_x - Trans1B2B
    
    """ Calculate new gray value and image gradient of C2_B image: for 2B2A """
    # 計算小數座標之灰階值，並以SOBEL計算影像梯度(image gradient)
    Size_TEMP = Size_2B2A + 2 # No value in boundary, so we allocated 2 more elements in x, y direction,...
    Len_TEMP = int(0.5*(Size_TEMP-1)) # ...make sure we have (Size)*(Size) size matrix.
    warp_aft_coef = np.array([(1, 0, U), (0, 1, V), (0, 0, 1)], dtype=float)
    Gvalue_TEMP = np.zeros(((Size_TEMP),(Size_TEMP)), dtype=float)
    for i in range(0,Size_TEMP,1): # Size需要多2個元素，因為稍後需計算新的Sobel影像梯度
        for j in range(0,Size_TEMP,1):
            Position = np.transpose(np.array([i-Len_TEMP, j-Len_TEMP, 1], dtype=float))
            warp_aft = warp_aft_coef.dot(Position)
            # 查表以尋找目標點之插值係數(位移記得取整數)
            a1 = Length + int(np.floor(warp_aft[0]))
            a2 = Length + int(np.floor(warp_aft[1]))
            A = Cubic_coef_1B2B[a1][a2][:]
            A_re = np.reshape(A, (4,4), order='F')
            # 計算插值之灰階值，並且儲存至Gvalue_g
            Gvalue_TEMP[i][j] =\
                Bicubic_test1.Bicubic_int(warp_aft[0]-np.floor(warp_aft[0]),\
                                          warp_aft[1]-np.floor(warp_aft[1]), A_re)
    # new gray value matrix of C2B (with interpolation): 
    img_2B_sub = Gvalue_TEMP[1:Size_TEMP-1,1:Size_TEMP-1]
    # Image gradient of new C2B image
    IGrad_2B2A_u_temp = cv.Sobel(Gvalue_TEMP, cv.CV_64F, 0, 1)*0.125 # y方向
    IGrad_2B2A_v_temp = cv.Sobel(Gvalue_TEMP, cv.CV_64F, 1, 0)*0.125 # x方向
    Sobel_2B_u = IGrad_2B2A_u_temp[1:Size_TEMP-1,1:Size_TEMP-1]
    Sobel_2B_v = IGrad_2B2A_v_temp[1:Size_TEMP-1,1:Size_TEMP-1]
    
    return C2_B_x, C2_B_y, Sobel_2B_u, Sobel_2B_v, img_2B_sub

