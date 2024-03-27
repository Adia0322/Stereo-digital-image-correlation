""" Interger_pixels_searching: Particle swarm optimization (PSO) """
""" Success rate test !!

2022/5/9 22:32
相片尺寸要求 1920*1080
將變形前影像向右平移 1 pixel 作為變形後影像 !!
計算100次，紀錄失敗次數
注意!! :
    1.子矩陣大小(邊長)、掃描範圍Scan: 需要是奇數!!
    2.PSO圖片尺寸需一致
    
    
"""
import time
import numpy as np
import cv2 as cv
from ctypes import cdll, c_int, POINTER
import Calculate_coef
import Bicubic_test1


img_bef_address = "./in-plane (FHD)/0.jpg"
img_aft_address = "./in-plane (FHD)/0_tran+1.jpg"    # 此行在成功率測試用不到

img_bef = cv.imread(str(img_bef_address), 0)
img_aft = cv.imread(str(img_aft_address), 0)
# image size: 1920*1080

# 取得圖片尺寸
ROW, COL = img_bef.shape # 注意 ROW,COL不會存到變數...

# 讀入影像校正檔案，取得投影矩陣
cv_file = cv.FileStorage()
cv_file.open('stereoMap.xml', cv.FileStorage_READ)
projMatrixL = cv_file.getNode('projMatrixL').mat()
projMatrixR = cv_file.getNode('projMatrixR').mat()

cv.namedWindow("img_bef", cv.WINDOW_NORMAL)    # 注意影像名稱要與圖片名稱相同 !!
cv.imshow("img_bef", img_bef)   

# =====================函式:選取目標點========================= #
# function to display the coordinates of5
# of the points clicked on the image 
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
        cv.putText(img_bef, str(x) + ',' +
                    str(y), (x,y), font,
                    1, (255, 0, 0), 2) # 無設定第8種的線條種類，直接忽略不寫
        cv.imshow('img_bef', img_bef)       # 注意影像名稱要與圖片名稱相同 !!
        u1 = y
        v1 = x
# ====================================================== #

cv.setMouseCallback('img_bef', click_event) # 注意影像名稱要與圖片名稱相同 !!
cv.waitKey(0) # 等待輸入時間
cv.destroyAllWindows() # 案任意鍵清除退出


u1 = 488
v1 = 938

"""================開始計時================="""
time_start = time.time() 


# 設定子矩陣大小(邊長) 需要是奇數!!
Size = 23
# 設定掃瞄方形範圍 (邊長) (scan只須在c語言內設定即可)
Scan = 51   #  !!!!!!! 往後急需修正 !!!!!!!!!   此定義與C內的定義不同 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!5

""" 注意: 執行前請先確認 Size scan 在python c 皆一致，否則無法執行卻沒有錯誤信息!! """
""" 注意: 不同圖片尺寸需要在c程式裡修改 img_aft: Column的大小 !! """

# 將 img_aft, img_bef 轉為int32型態
img_bef = np.array(img_bef, dtype=int)
img_aft = np.array(img_aft, dtype=int)

# 建立位移暫存區，存放位移量x1 y1 (為了區別座標x y，位移使用x1 y1)
Displacement = np.zeros((4,), dtype=int) # 依序為 [u, v, Gbest_point, Gbest_value]

# 所選取目標點的位置 
Object_point = np.array((u1,v1), dtype=int)


# 建構變形前後影像之子矩陣: img_bef_sub
img_bef_sub = img_bef[u1-int((Size-1)/2):u1+int((Size-1)/2)+1, v1-int((Size-1)/2):v1+int((Size-1)/2)+1] # 注意!! 要加1，因為不包含尾項!!
img_bef_sub = img_bef_sub-np.mean(img_bef_sub) 
img_bef_sub = img_bef_sub.astype(int)     # 將float轉int      注意!!! 驗證成功後建議改回float以增加準確度!!!! 

img_aft_sub = np.zeros((Size,Size), dtype=int)

# sensor: [Pi[i][0], Pi[i][1], Vi[i][0], Vi[i][1]
sensor = np.zeros((5,2), dtype=int)
sensor_coef = np.zeros(3, dtype=int) #  Coefficient1、2、3

#============================ 使用C計算位移 ============================#
#所需引數(7): SCAN(float img_aft[][1920], float img_aft_sub[][Size],
#                  float img_bef_sub[][Size], int Object_point[2], int Displacement[4], float sensor[][4]) 

# 載入SO 動態連結檔案: test_2D_DIC_displacement.so
m = cdll.LoadLibrary('./PSO_ICGN_FHD.so')

# 設定 SO 檔案中 SCAN 函數的參數資料型態:
m.SCAN.argtypes = [POINTER(c_int), POINTER(c_int), POINTER(c_int),\
                   POINTER(c_int), POINTER(c_int), POINTER(c_int),\
                   POINTER(c_int)]

# 設定 SO 檔案中 SCAN 函數的傳回值資料型態
m.SCAN.restype = c_int #似乎可以不設定

# 取得陣列指標 7個
img_aft_Ptr = img_aft.ctypes.data_as(POINTER(c_int))
img_aft_sub_Ptr = img_aft_sub.ctypes.data_as(POINTER(c_int))
img_bef_sub_Ptr = img_bef_sub.ctypes.data_as(POINTER(c_int))
Object_point_Ptr = Object_point.ctypes.data_as(POINTER(c_int))
Displacement_Ptr = Displacement.ctypes.data_as(POINTER(c_int))
sensor_ptr = sensor.ctypes.data_as(POINTER(c_int))
sensor_coef_ptr = sensor_coef.ctypes.data_as(POINTER(c_int))

# 呼叫 SO 檔案中的 SCAN 函數 
m.SCAN(img_aft_Ptr, img_aft_sub_Ptr, img_bef_sub_Ptr, Object_point_Ptr,\
       Displacement_Ptr, sensor_ptr, sensor_coef_ptr)
#=================== 位移計算完成 計算結果在Displacement =========================#

#============== 數據 ===============#
#Displacement
min_index = Displacement[2]
min_value_Gbest = Displacement[3]

print("垂直:", Displacement[0])  #垂直以下為正
print("水平:", Displacement[1])  #水平以右為正

#print("min_index:", Displacement[2])
#print("min_value_Gbest:", Displacement[3])
#print("總位移:",np.sqrt(Displacement[0]**2 + Displacement[1]**2))

#print("\n sensor_coef:")
#print("位移(0,0)的相關係數:", sensor_coef[0])

#u2 = Displacement[0]
#v2 = Displacement[1]
    
    
#print('Success_List:', Success_List)
#print('Success rate (%):', (100-np.sum(Success_List)))


"""===============計時結束================="""
time_end = time.time()    #結束計時
time_c = time_end - time_start   #執行所花時間
print('PSO time cost', time_c, 's') 




