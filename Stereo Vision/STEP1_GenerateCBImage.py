
""" Generate_Chessboard_Image """
import cv2 as cv
import os
import glob
import time

"""================== 參數設定 ==================="""
# 手動設定焦距: 1:YES 0:NO
Flag_focal = 1
# 焦距大小
focal_1 = 70
focal_2 = 70
# 相機編號
cap_num = 1
cap2_num = 0

""" Delete old images in folders """
jpg1_files = glob.glob('images/StereoLeft/*.jpg')
jpg2_files = glob.glob('images/StereoRight/*.jpg')

for jpg_file in jpg1_files:
    try:
        os.remove(jpg_file)
    except OSError as e:
        print(f"Error:{ e.strerror}")
        
for jpg_file in jpg2_files:
    try:
        os.remove(jpg_file)
    except OSError as e:
        print(f"Error:{ e.strerror}")
        
""" 由於兩個攝像頭配置不同，需要將影像轉正，因此定義旋轉矩陣函式 """
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


""" 建構並拍攝棋盤影像 """
print('Opening Cameras...')
# 擷取網路攝影機影像: 0表示相機的編號
cap = cv.VideoCapture(cap_num, cv.CAP_DSHOW)
cap2 = cv.VideoCapture(cap2_num, cv.CAP_DSHOW)

# close auto setting
cap.set(21,0)
cap2.set(21,0)
cap.set(39,0)
cap2.set(39,0)
cap.set(cv.CAP_PROP_AUTO_WB,0)
cap2.set(cv.CAP_PROP_AUTO_WB,0)
# 手動設定相機焦距(若相機無自動對焦)
if Flag_focal == 1:
    cap.set(28,focal_1)
    cap2.set(28,focal_2)


num = 0
print('\nDone!')

# 建立2個相機視窗
cv.namedWindow("Img1", cv.WINDOW_NORMAL)
cv.namedWindow("Img2", cv.WINDOW_NORMAL)

while (cap.isOpened() and cap.isOpened()):
    # 計時開始
    time_start = time.time()
    # 讀入影像1 2
    sucess1, img = cap.read() 
    sucess2, img2 = cap2.read()
    
    # ()內設定等待秒數，利用回傳值取得按鍵的 ASCII 碼值
    k = cv.waitKey(5)
    
    # 轉正影像
    Img1 = rotate(img,0)
    Img2 = rotate(img2,0)
    
    # 設定ESC退出迴圈
    if k == 27: 
        break
    
    elif k == ord('s'):  # 按S儲存影像(未退出迴圈)
        cv.imwrite('./images/StereoLeft/camera1.image' + str(num) + '.jpg', Img1)
        cv.imwrite('./images/StereoRight/camera2.image' + str(num) + '.jpg', Img2)
        # 儲存未矯正影像
        # cv.imwrite('./images/Target/camera1/camera1.origin_image' + str(num) + '.jpg', Img1)
        # cv.imwrite('./images/Target/camera2/camera2_.origin_image' + str(num) + '.jpg', Img2)
        print("images save!")
        num += 1
    
    # 設定可調整大小的視窗
    #cv.namedWindow("Img1", cv.WINDOW_NORMAL)
    #cv.namedWindow("Img2", cv.WINDOW_NORMAL)
    # 計時結束 
    time_end = time.time()
    time_c = time_end - time_start   # 執行所花時間
    fps = round(1/time_c)
    cv.putText(Img1, 'FPS:' + str(fps), (20, 60),\
               cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    # 顯示影像:   
    cv.imshow('Img1', Img1) 
    cv.imshow('Img2', Img2)


cap.release()
cap2.release()
    
cv.destroyAllWindows()