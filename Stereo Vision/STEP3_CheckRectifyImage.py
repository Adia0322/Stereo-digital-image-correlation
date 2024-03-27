# -*- coding: utf-8 -*-
"""
Stereo Vision

執行前注意事項:
    1.相機對應編號
    
手動影像存檔位址:
    ./images/Target/camera1/
"""

import cv2 as cv
import time

"""================== 參數設定 ==================="""
# displacement
dis = str(0.0)
# 手動設定焦距: 1:YES 0:NO
Flag_focal = 1
# 焦距大小
focal_1 = 70
focal_2 = 70
# 相機編號
cap_left_num = 1
cap_right_num = 0

num = 1   # 相片編號

print("\n<< Stereo_DIC_PSO_ICGN >>")
# ========= 定義函式:選取目標點 ============== #
def click_event(event, x, y, flags, params):
  
    # checking for left mouse clicks
    if event == cv.EVENT_LBUTTONDOWN:
  
        global u1, v1
        print("點選的座標:",x, ' ', y)
        u1 = y
        v1 = x      

# ==================== 旋轉影像 =======================
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
# ====================================================


# Camera parameters to undistort and rectify images
cv_file = cv.FileStorage()
cv_file.open('stereoMap.xml', cv.FileStorage_READ)

stereoMapL_x = cv_file.getNode('stereoMapL_x').mat()
stereoMapL_y = cv_file.getNode('stereoMapL_y').mat()
stereoMapR_x = cv_file.getNode('stereoMapR_x').mat()
stereoMapR_y = cv_file.getNode('stereoMapR_y').mat()


# Open both cameras (注意相機編號!!!)
cap_left =  cv.VideoCapture(cap_left_num, cv.CAP_DSHOW)
cap_right = cv.VideoCapture(cap_right_num, cv.CAP_DSHOW)                    

# close auto setting
# cap_left.set(21,0)
# cap_right.set(21,0)
# cap_left.set(39,0)
# cap_right.set(39,0)
# 手動設定相機焦距(若相機無自動對焦) cv2.CAP_PROP_FOCUS
if Flag_focal == 1:
    cap_left.set(cv.CAP_PROP_FOCUS,focal_1)
    cap_right.set(cv.CAP_PROP_FOCUS,focal_2)


# 建立2個相機視窗
cv.namedWindow("frame left", cv.WINDOW_NORMAL)
cv.namedWindow("frame right", cv.WINDOW_NORMAL)

# 選擇的目標點初始在img_C1_new上座標 (數值隨意不重要)
u1 = -10
v1 = -10

# 呼叫函數(目標點座標)
cv.setMouseCallback('frame left', click_event)

while(cap_right.isOpened() and cap_left.isOpened()):
    # 計時開始
    time_start = time.time()
    # 讀入影像
    succes_left, frame_left0 = cap_left.read()
    succes_right, frame_right0 = cap_right.read()
    # 轉正影像
    frame_left_ori = rotate(frame_left0,0)
    frame_right_ori = rotate(frame_right0,0)
    # Undistort and rectify images
    frame_left_rec = cv.remap(frame_left_ori, stereoMapL_x, stereoMapL_y, cv.INTER_LANCZOS4, cv.BORDER_CONSTANT, 0)
    frame_right_rec = cv.remap(frame_right_ori, stereoMapR_x, stereoMapR_y, cv.INTER_LANCZOS4, cv.BORDER_CONSTANT, 0)
    
    # 計時結束
    time_end = time.time()
    time_c = time_end - time_start   # 執行所花時間
    fps = round(1/time_c)
    # cv.putText(frame_left1, 'FPS:' + str(fps), (20, 60),\
    #                  cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv.putText(frame_left_rec, str(v1) + ',' +str(u1), (v1+10, u1-10),\
               cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    #cv.putText(img_1B_new_temp,  'Please choose the target', (30, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    frame_left_rec = cv.circle(frame_left_rec, (v1, u1), 5, (0, 255, 255), 1)
    
    # Show the frames
    cv.imshow("frame left", frame_left_rec)
    cv.imshow("frame right", frame_right_rec) 

    k = cv.waitKey(5)
    if k==27 or num==11: 
        break
    
    # 按s儲存影像(注意!! 存的是未校正的原圖!!)
    elif k == ord('s'):
        cv.imwrite('./images/Target20230901-1/in/camera1/cal_0_5kg_0cm.image' + str(num) + '.jpg', frame_left_ori)
        cv.imwrite('./images/Target20230901-1/in/camera2/cal_0_5kg_0cm.image' + str(num) + '.jpg', frame_right_ori)
        print("images" + str(num) + " save!")
        num += 1

# Release and destroy all windows before termination
cap_right.release()
cap_left.release()
cv.destroyAllWindows()
print('End')