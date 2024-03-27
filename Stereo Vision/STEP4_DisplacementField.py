
print("\n<< Stereo_DIC_PSO_ICGN >>")

import numpy as np
import cv2 as cv
import time
import matplotlib.pyplot as plt
import matplotlib as m
from scipy import interpolate
import PSO_ICGN_1B2B
import PSO_ICGN_1B1A
import PSO_ICGN_2B2A
import CubicCoef_1B2B_all
import CubicCoef_1B1A
import CubicCoef_2B2A
import Hessian
import Image_Calibration as Img_cal    
import Points2Plane
import os

# displacement(image)
disRigid = str(0.1)
# weights(image)
kg = 5

# folder address
folder_dir = 'Target20230901-1'

# in-plane:0, out-of-plane:1
plane_flag = 0
if plane_flag == 0:
    plane = str('in')
else:
    plane = str('out')

# image rectification? no:0 yes:1
rec_flag = 1

# image rotation: no:0 yes:1
angle_flag = 0

# fixed point (optional)
u1 = 265
v1 = 468
u1c2 = 263
v1c2 = 168

""" ======== images ====== """
# case1
img_1B_adress = './images/'+folder_dir+'/'+str(plane)+'/camera1/cal_0_'+\
                  str(kg)+'kg_0cm.image1.jpg'
img_2B_adress = './images/'+folder_dir+'/'+str(plane)+'/camera2/cal_0_'+\
                  str(kg)+'kg_0cm.image1.jpg'


img_1B = cv.imread(str(img_1B_adress))
img_2B = cv.imread(str(img_2B_adress))


def rotate(image, angle, center=None, scale=1.0):
    # 獲取圖片尺寸
    (h, w) = image.shape[:2]
 
    # 若未指定旋轉中心，則將圖像中心設定為旋轉中心
    if center is None:
        center = (w / 2, h / 2)
 
    # 執行旋轉
    M = cv.getRotationMatrix2D(center, angle, scale)
    rotated = cv.warpAffine(image, M, (w, h))

    # 傳回旋轉後的圖像
    return rotated


# image rotation
if angle_flag == 1:
    img_1B = rotate(img_1B,-90)
    img_2B = rotate(img_2B,90)


# image rectification
if rec_flag == 1:
    img_1B_new, img_2B_new = Img_cal.undistortRectify(img_1B, img_2B)
else:
    img_1B_new = img_1B
    img_2B_new = img_2B
    
# save images
cv.imwrite('thesis_img/img_1B_new_thesis.jpg', img_1B_new)
cv.imwrite('thesis_img/img_2B_new_thesis.jpg', img_2B_new)
# copy images
img_1B_new_temp = img_1B_new
img_2B_new_temp = img_2B_new
""" ============== choose the point you want ================ """
cv.putText(img_1B_new_temp, 'set a reference point on img_1B', (20, 60),\
            cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

# build two windows
cv.namedWindow("img_1B_new_temp", cv.WINDOW_NORMAL)
cv.namedWindow("img_2B_new_temp", cv.WINDOW_NORMAL)
cv.imshow("img_1B_new_temp", img_1B_new_temp)
cv.imshow("img_2B_new_temp", img_2B_new_temp)

# ===================== function ========================= #
def click_event(event, x, y, flags, params):
    # checking for left mouse clicks
    if event == cv.EVENT_LBUTTONDOWN:
        global u1, v1
        # displaying the coordinates
        # on the Shell
        print("點選的座標:",x, ' ', y)
        # displaying the coordinates
        # on the image window
        font = cv.FONT_HERSHEY_SIMPLEX
        cv.putText(img_1B_new_temp, str(x) + ',' +
                    str(y), (x,y), font,
                    1, (0, 255, 0), 2) # 無設定第8種的線條種類，直接忽略不寫
        cv.imshow('img_1B_new_temp', img_1B_new_temp)       # 注意影像名稱要與圖片名稱相同 !!
        u1 = y
        v1 = x
# ====================================================== #

print('Please set a reference point')
cv.setMouseCallback('img_1B_new_temp', click_event)
cv.waitKey(0) # 等待輸入時間
cv.destroyAllWindows() # 案任意鍵清除退出

# ============= 選擇粗略對應點 =========================
cv.putText(img_2B_new_temp, 'set a corresponding point on img_2B', (20, 60),\
            cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
# 建立2個相機視窗
cv.namedWindow("img_1B_new_temp", cv.WINDOW_NORMAL)
cv.namedWindow("img_2B_new_temp", cv.WINDOW_NORMAL)
cv.imshow("img_1B_new_temp", img_1B_new_temp)
cv.imshow("img_2B_new_temp", img_2B_new_temp)

# ===================== function ========================= #
def click_event2(event, x, y, flags, params):
  
    # checking for left mouse clicks
    if event == cv.EVENT_LBUTTONDOWN:
  
        global u1c2, v1c2
        # displaying the coordinates
        # on the Shell
        print("點選的座標:",x, ' ', y)
        # displaying the coordinates
        # on the image window
        font = cv.FONT_HERSHEY_SIMPLEX
        cv.putText(img_2B_new_temp, str(x) + ',' +
                    str(y), (x,y), font,
                    1, (0, 255, 0), 2) # 無設定第8種的線條種類，直接忽略不寫
        cv.imshow('img_2B_new_temp', img_2B_new_temp)       # 注意影像名稱要與圖片名稱相同 !!
        u1c2 = y
        v1c2 = x
# ====================================================== #
print('Please choose a reference point')
cv.setMouseCallback('img_2B_new_temp', click_event2)
cv.waitKey(0) 
cv.destroyAllWindows() 

# 讀入影像校正檔案，取得投影矩陣 (也可在Image_Calibration直接取得)
cv_file = cv.FileStorage()
cv_file.open('stereoMap.xml', cv.FileStorage_READ)
Q = cv_file.getNode('Q').mat()


# 重讀影像(目的去除圖片上滑鼠點擊留下的座標)
img_1B = cv.imread(str(img_1B_adress))

# 影像校正
if rec_flag == 1:
    img_1B_new, img_2B_new = Img_cal.undistortRectify(img_1B, img_2B)
else:
    img_1B_new = img_1B
    img_2B_new = img_2B
        

""" =============== 參數設定 ==============="""
# 需連同: PSO_ICGN_1B2B、PSO_ICGN_1B1A、PSO_ICGN_2B2A、\
       # Iteration_1B1A、Iteration_2B2A\
       # 一共5個檔案進行更改!


# 設定分析點數量
analysisNum = 25
SL = int(np.sqrt(analysisNum))
SLH = int((SL-1)/2)             # side length half: (SLH)
anaMat = np.zeros((int(np.sqrt(analysisNum)),int(np.sqrt(analysisNum))), dtype=int)
# 分析點間隔 (pixel)
interval = 10
# 設定子集合大小(方陣邊長).
Size_1B2B = 31
Size_1B1A = 31
Size_2B2A = 31

# 設定掃瞄範圍(方陣邊長)
Scan_1B2B = 31
Scan_1B1A = 31
Scan_2B2A = 31

# <<由於兩部相機之間距離較大(視差)，1B1A特別設定平移距離，以利於DIC能快速找到在2B影像裡的對應點>>
# Trans1B2B = 380
Trans1B2B = v1 - v1c2

# 焦距 (unit:pixel)
focal = Q[2][3] 

# baseline (unit:mm)
baseline = 1/Q[3][2]

# 影像尺寸
row, col, tunnel = img_1B_new.shape

# 兩相機中心點xy座標
principal_x = -Q[0][3]
principal_y = -Q[1][3]

# 起始點: (C1_B_x, C1_B_y)
C1_B_x = v1
C1_B_y = u1

TOTALtime = 0

# 高斯模糊
Flag_gau = 0

"""=========== 高斯模糊影像處理 (降低高頻誤差) =============="""
if Flag_gau == 1:
    img_1B_new = cv.GaussianBlur(img_1B_new, (3,3), sigmaX=1, sigmaY=1)
    img_2B_new = cv.GaussianBlur(img_2B_new, (3,3), sigmaX=1, sigmaY=1)
    # cv.imshow('img_1B_new_gau', img_1B_new)
    # cv.imshow('img_2B_new_gau', img_2B_new)
    # cv.waitKey(0) # 等待輸入(任意鍵繼續)
    # cv.destroyAllWindows() # 按任意鍵清除退出
    print("GaussianBlur")


""" 預先計算影像梯度 插值係數 等等資訊 """
""" ============= Compute image gradient Part1 =============="""
# Convert to gray image
img_1B_new_gray = cv.cvtColor(img_1B_new, cv.COLOR_BGR2GRAY)
img_2B_new_gray = cv.cvtColor(img_2B_new, cv.COLOR_BGR2GRAY)
# precompute the img_bef image gradient by Sobel operator
# C1B
Sobel_1B_u = cv.Sobel(img_1B_new_gray, cv.CV_64F, 0, 1)*0.125 # y方向
Sobel_1B_v = cv.Sobel(img_1B_new_gray, cv.CV_64F, 1, 0)*0.125 # x方向
# C2B
# Sobel_2B_u = cv.Sobel(img_2B_new_gray, cv.CV_64F, 0, 1)*0.125 # y方向 
# Sobel_2B_v = cv.Sobel(img_2B_new_gray, cv.CV_64F, 1, 0)*0.125 # x方向
""" ========== Interpolation: Bicubic part1 =========="""
# 填充額外數值以供插值係數計算: 2B_full
img_2B_new_gray_pad = np.pad(img_2B_new_gray,[(1,2),(1,2)], mode = 'edge')
row_2B, col_2B = img_2B_new_gray.shape             
CubicCoef_1B2B_ALL, Cubic_Xinv =\
    CubicCoef_1B2B_all.CalculateALL(img_2B_new_gray_pad, row_2B, col_2B)

# 建立儲存區
C1B_points = np.zeros((SL,SL,2), dtype=int) #像素座標C1
C2B_points = np.zeros((SL,SL,2), dtype=float) #像素座標C2
WC_bef_zone = np.zeros((SL,SL,3), dtype=float)
WC_aft_zone = np.zeros((SL,SL,3), dtype=float)
H1B1A_inv_all = np.zeros((SL,SL,6,6), dtype=float)
H2B2A_inv_all = np.zeros((SL,SL,6,6), dtype=float)
J1B1A_all = np.zeros((SL,SL,Size_1B1A,Size_1B1A,6), dtype=float)
J2B2A_all = np.zeros((SL,SL,Size_2B2A,Size_2B2A,6), dtype=float)
img_2B_sub_zone = np.zeros((SL,SL,Size_2B2A,Size_2B2A), dtype=float)
disM = np.zeros((SL,SL,3), dtype=float)
disM_out = np.zeros((SL,SL), dtype=float)
disM_in_1 = np.zeros((SL,SL), dtype=float)
disM_in_2 = np.zeros((SL,SL), dtype=float)
stress_in = np.zeros((SL,SL), dtype=float)
stress_out = np.zeros((SL,SL), dtype=float)

# 計算多點對應點
for P in range(-SLH,SLH+1,1):
    for L in range(-SLH,SLH+1,1):  
        C1_B_x = int(interval*L + v1)
        C1_B_y = int(interval*P + u1)
        C1B_points[P+SLH][L+SLH][0] = C1_B_y
        C1B_points[P+SLH][L+SLH][1] = C1_B_x

        """ ============= Compute image gradient Part1 =============="""
        # Convert to gray image
        img_1B_new_gray = cv.cvtColor(img_1B_new, cv.COLOR_BGR2GRAY)
        img_2B_new_gray = cv.cvtColor(img_2B_new, cv.COLOR_BGR2GRAY)

        # Image gradient of 1B2B
        Len_1B2B = int(0.5*(Size_1B2B-1))
        IGrad_1B2B_u = Sobel_1B_u[C1_B_y-Len_1B2B:C1_B_y+Len_1B2B+1,\
                                  C1_B_x-Len_1B2B:C1_B_x+Len_1B2B+1]
        IGrad_1B2B_v = Sobel_1B_v[C1_B_y-Len_1B2B:C1_B_y+Len_1B2B+1,\
                                  C1_B_x-Len_1B2B:C1_B_x+Len_1B2B+1]
        H_inv_1B2B, J_1B2B = Hessian.Calculate(Size_1B2B, IGrad_1B2B_u, IGrad_1B2B_v)
        
        # 從全圖插值表找插值(用於1B2B搜尋)
        Length_1B2B = int(0.5*(Size_1B2B-1)+0.5*(Scan_1B2B-1))
        CubicCoef_1B2B = CubicCoef_1B2B_ALL[C1_B_y-Length_1B2B:C1_B_y+Length_1B2B+1,\
                                            C1_B_x-Trans1B2B-Length_1B2B:C1_B_x-Trans1B2B+Length_1B2B+1]
        # 1B2B尋找對應點與2B之影像梯度
        C2_B_x, C2_B_y, Sobel_2B_u, Sobel_2B_v, img_2B_sub =\
        PSO_ICGN_1B2B.Calculate_1B2B(img_1B_new_gray, img_2B_new_gray,\
                                     C1_B_x, C1_B_y,\
                                     Size_1B2B, Size_2B2A, Scan_1B2B, H_inv_1B2B,\
                                     J_1B2B, CubicCoef_1B2B, Trans1B2B)
            
        C2B_points[P+SLH][L+SLH][0] = C2_B_y
        C2B_points[P+SLH][L+SLH][1] = C2_B_x
        """ 計算初始三維座標 """
        # 計算視差 xl-xr (unit:pixel)
        Disparity_1B2B = (C1_B_x - C2_B_x) 
        Disparity_1B2B_reci = np.divide(1, Disparity_1B2B)
        # 3D coordinate of Reference point (initial)
        X_origin = (C1_B_x-principal_x)*baseline*Disparity_1B2B_reci
        Y_origin = (C1_B_y-principal_y)*baseline*Disparity_1B2B_reci
        Z_origin = focal*baseline*Disparity_1B2B_reci
        WC_bef_zone[P+SLH][L+SLH][0] = X_origin
        WC_bef_zone[P+SLH][L+SLH][1] = Y_origin
        WC_bef_zone[P+SLH][L+SLH][2] = Z_origin
        
        # << 預計算H_inv_2A2B, J_2A2B >>
        # 1B1A
        Len_1B1A = int(0.5*(Size_1B1A-1))
        IGrad_1B1A_u = Sobel_1B_u[C1_B_y-Len_1B1A:C1_B_y+Len_1B1A+1,\
                                  C1_B_x-Len_1B1A:C1_B_x+Len_1B1A+1]
        IGrad_1B1A_v = Sobel_1B_v[C1_B_y-Len_1B1A:C1_B_y+Len_1B1A+1,\
                                  C1_B_x-Len_1B1A:C1_B_x+Len_1B1A+1]
        H_inv_1B1A, J_1B1A =\
            Hessian.Calculate(Size_1B1A, IGrad_1B1A_u, IGrad_1B1A_v)
        # store H and J
        H1B1A_inv_all[P+SLH][L+SLH][:][:] = H_inv_1B1A[:][:]
        J1B1A_all[P+SLH][L+SLH][:][:][:] = J_1B1A[:][:][:]
        
        # 2B2A (注意:影像梯度矩陣尺寸需調整至2B2A尺寸)
        Len_2B2A = int(0.5*(Size_2B2A-1))
        IGrad_2B2A_u = Sobel_2B_u
        IGrad_2B2A_v = Sobel_2B_v
        H_inv_2B2A, J_2B2A =\
            Hessian.Calculate(Size_2B2A, IGrad_2B2A_u, IGrad_2B2A_v) 
        H_inv_2B2A_test = H_inv_2B2A 
        # store H and J
        H2B2A_inv_all[P+SLH][L+SLH][:][:] = H_inv_2B2A[:][:]
        J2B2A_all[P+SLH][L+SLH][:][:][:] = J_2B2A[:][:][:]
        # store img_2B_sub
        img_2B_sub_zone[P+SLH][L+SLH][:][:] = img_2B_sub

# 指定陣列中心之追蹤點在2B之位置
u2 = C2B_points[SLH][SLH][0]
v2 = C2B_points[SLH][SLH][1]

disSum = 0

# ==================================================================

"""  決定擬合平面與追蹤點第4個點  """
# 平面法向量
nVector = Points2Plane.normalVector(WC_bef_zone, SL)
# 正規化
nVector = nVector/np.linalg.norm(nVector)

ImgNum = 1
for ImgNum in range(1,2,1):
    # print('ImgNum:',ImgNum)
    import PSO_ICGN_1B2B
    import PSO_ICGN_1B1A
    import PSO_ICGN_2B2A
    import CubicCoef_1B2B
    import CubicCoef_1B1A
    import CubicCoef_2B2A
    import Hessian
    import Image_Calibration as Img_cal  
    # 讀入影像
    img_1A_adress = './images/'+folder_dir+'/'+str(plane)+'/camera1/cal_'+\
                      str(kg)+'_'+str(kg)+'kg_0cm.image'+str(ImgNum)+'.jpg'
    img_2A_adress = './images/'+folder_dir+'/'+str(plane)+'/camera2/cal_'+\
                      str(kg)+'_'+str(kg)+'kg_0cm.image'+str(ImgNum)+'.jpg'
    
    img_1A = cv.imread(str(img_1A_adress))
    img_2A = cv.imread(str(img_2A_adress))
    
    # 旋轉影像
    if angle_flag == 1:
        img_1A = rotate(img_1A,-90)
        img_2A = rotate(img_2A,90)
    
    # 影像校正
    if rec_flag == 1:
        img_1A_new, img_2A_new = Img_cal.undistortRectify(img_1A, img_2A)
    else:
        img_1A_new = img_1A
        img_2A_new = img_2A
        
    # 儲存影像
    cv.imwrite('thesis_img/img_1A_new_thesis.jpg', img_1A_new)
    cv.imwrite('thesis_img/img_2A_new_thesis.jpg', img_2A_new)
    
    # (選擇性)高斯模糊影像處理: 降低高頻誤差
    if Flag_gau == 1:
        img_1A_new = cv.GaussianBlur(img_1A_new, (3,3), sigmaX=1, sigmaY=1)
        img_2A_new = cv.GaussianBlur(img_2A_new, (3,3), sigmaX=1, sigmaY=1)
        print("GaussianBlur")
    
    # Convert to gray image
    img_1A_new_gray = cv.cvtColor(img_1A_new, cv.COLOR_BGR2GRAY)
    img_2A_new_gray = cv.cvtColor(img_2A_new, cv.COLOR_BGR2GRAY)
    # 計算1A影像插值係數
    # Length
    Length = int(0.5*(Size_1B1A-1)+0.5*(Scan_1B1A-1)+20)   #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    Length_1B1A = int(0.5*(Size_1B1A-1)+0.5*(Scan_1B1A-1))
    Length_2B2A = int(0.5*(Size_2B2A-1)+0.5*(Scan_2B2A-1))
    
    # Time start
    start2 = time.time()
    
    # 同張影像陣列目標點追蹤
    for P in range(-SLH,SLH+1,1):
        for L in range(-SLH,SLH+1,1):
            C1_B_y = C1B_points[P+SLH][L+SLH][0] #integer
            C1_B_x = C1B_points[P+SLH][L+SLH][1] #integer
            C2_B_y = C2B_points[P+SLH][L+SLH][0] #decimal
            C2_B_x = C2B_points[P+SLH][L+SLH][1] #decimal
            
            # Time start
            start = time.time()
            
            # Length = Length_1B1A = Length_2B2A
            Length = int(0.5*(Size_1B1A-1)+0.5*(Scan_1B1A-1))
            
            # Time start _1B1A
            start_1B1A = time.time()
            
            # 插值係數1B1A
            Gvalue_1B1A = img_1A_new_gray[int(C1_B_y)-Length-1:int(C1_B_y)+Length+3,\
                                          int(C1_B_x)-Length-1:int(C1_B_x)+Length+3] 
            Cubic_coef_1B1A =\
                CubicCoef_1B1A.Calculate(Cubic_Xinv, Length, Gvalue_1B1A)
            # H, J
            H_inv_1B1A[:][:] = H1B1A_inv_all[P+SLH][L+SLH][:][:]
            J_1B1A[:][:][:] = J1B1A_all[P+SLH][L+SLH][:][:][:]
    
            # 搜尋對應點
            C1_A_x, C1_A_y, Coef_1B1A =\
                PSO_ICGN_1B1A.Calculate_1B1A(img_1B_new_gray, img_1A_new_gray,\
                                             C1_B_x, C1_B_y, Size_1B1A, Scan_1B1A,\
                                             H_inv_1B1A, J_1B1A, Cubic_coef_1B1A)
            # Time end!
            end_1B1A = time.time()
            time_1B1A = end_1B1A - start_1B1A
            # print('time_1B1A:',time_1B1A)
            
            # Time start _2B2A
            start_2B2A = time.time()
            
            # 提取1B2B計算之img_2B灰階值矩陣
            img_2B_sub = img_2B_sub_zone[P+SLH][L+SLH]

            # 插值係數2B2A      
            Gvalue_2B2A = img_2A_new_gray[int(C2_B_y)-Length-1:int(C2_B_y)+Length+3,\
                                          int(C2_B_x)-Length-1:int(C2_B_x)+Length+3] 
            Cubic_coef_2B2A =\
                CubicCoef_2B2A.Calculate(Cubic_Xinv, Length, Gvalue_2B2A)           
            # H, J
            H_inv_2B2A[:][:] = H2B2A_inv_all[P+SLH][L+SLH][:][:]
            J_2B2A[:][:][:] = J2B2A_all[P+SLH][L+SLH][:][:][:]         
            # 搜尋對應點
            C2_A_x, C2_A_y, Coef_2B2A =\
                PSO_ICGN_2B2A.Calculate_2B2A(img_2B_new_gray, img_2A_new_gray,\
                                             C2_B_x, C2_B_y, Size_2B2A, Scan_2B2A,\
                                             H_inv_2B2A, J_2B2A, Cubic_coef_2B2A,\
                                             img_2B_sub)
            # Time end!
            end_2B2A = time.time()
            time_2B2A = end_2B2A - start_2B2A
            # print('time_2B2A:',time_2B2A)
            
            """ 計算當前目標點之世界座標  """
            # 計算視差 xl-xr (unit:pixel)
            Disparity_1A2A = (C1_A_x - C2_A_x)
            Disparity_1A2A_reci = np.divide(1, Disparity_1A2A)
            X_after = (C1_A_x-principal_x)*baseline*Disparity_1A2A_reci
            Y_after = (C1_A_y-principal_y)*baseline*Disparity_1A2A_reci
            Z_after = focal*baseline*Disparity_1A2A_reci
            
            # Displacement between reference point and target point
            WC_aft_zone[P+SLH][L+SLH][0] = X_after
            WC_aft_zone[P+SLH][L+SLH][1] = Y_after
            WC_aft_zone[P+SLH][L+SLH][2] = Z_after
            disM[P+SLH][L+SLH][:] = WC_aft_zone[P+SLH][L+SLH][:] - WC_bef_zone[P+SLH][L+SLH][:]
            # out:z, in1:x(水平向右+), in2:y(垂直向下+)
            dis_out = WC_aft_zone[P+SLH][L+SLH][2]-WC_bef_zone[P+SLH][L+SLH][2]
            dis_out2 = np.dot(disM[P+SLH][L+SLH],nVector)
            dis_in_1 = WC_aft_zone[P+SLH][L+SLH][0]-WC_bef_zone[P+SLH][L+SLH][0]
            dis_in_2 = WC_aft_zone[P+SLH][L+SLH][1]-WC_bef_zone[P+SLH][L+SLH][1]
            dis_in_sum = np.sqrt(dis_in_1**2 + dis_in_2**2)
            
            if plane_flag==0: # in plane
                print(np.round(dis_in_sum, 6))
                disSum += dis_in_sum
            else: # out of plane
                print(np.round(dis_out2, 6))
                disSum += dis_out
            
            img_1A_new = cv.circle(img_1A_new, (int(C1_A_x), int(C1_A_y)), 5,\
                                  (0, 255, 255), 1)  
            img_2A_new = cv.circle(img_2A_new, (int(C2_A_x), int(C2_A_y)), 5,\
                                  (0, 255, 255), 1)  
            # Time end!
            end = time.time()
            total_time = end - start
            TOTALtime = TOTALtime + total_time   
            
            disM_out[P+SLH][L+SLH] = dis_out
            disM_in_1[P+SLH][L+SLH] = dis_in_1
            disM_in_2[P+SLH][L+SLH] = dis_in_2

    # Time end!
    end2 = time.time()
    total_time2 = end2 - start2
    # print('total_time2: ', total_time2)
    

cv.imshow('img_1A_new', img_1A_new)
cv.imshow('img_2A_new', img_2A_new)
cv.waitKey(0) # 等待輸入時間
cv.destroyAllWindows()
cv.imwrite('disField/img_1A_new.jpg', img_1A_new)
cv.imwrite('disField/img_2A_new.jpg', img_2A_new)


print('Average time per point: ', TOTALtime/(analysisNum*10))
print('Average dis:',disSum/(ImgNum*SL*SL))
print("End")

