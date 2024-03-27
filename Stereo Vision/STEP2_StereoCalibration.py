"""
Camera Calibration
"""
import numpy as np
import cv2 as cv
import glob

""" 尋找棋盤點 """
# 設定棋盤點與影像尺寸
chessboardSize = (9,6)
frameSize = (640,480)

# 設定停止執行條件 (2種條件任一種滿足則停止)
# 1.EPS:達到目標精確度(epsilon)則停止
# 2.MAX_ITER:超過迭代次數則停止
# 輸入格式:(type,max_iter,epsilon）= (停止條件,迭代次數,精確度)
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.0001)

# 建構object矩陣
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)

objp = objp * 8 # 填入棋盤方格邊長 (mm)
print('objp=',objp)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpointsL = [] # 2d points in image plane.
imgpointsR = [] # 2d points in image plane.

# 搜尋並取得指定文件的位址 (星號: 標定特定檔案格式)
# sorted: 將資料依編號排序
imagesLeft = sorted(glob.glob('images/StereoLeft/*.jpg'))
imagesRight = sorted(glob.glob('images/StereoRight/*.jpg'))

for imgLeft, imgRight in zip(imagesLeft, imagesRight): # zip: 將2組array中對應的元素打包成一個位組，最後傳回一個列表。
    imgL = cv.imread(imgLeft) # 將影像讀入儲存成array
    imgR = cv.imread(imgRight)
    grayL = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY) # 轉成灰階影像
    grayR = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)
    
    # Find the chess board corners
    retL, cornersL = cv.findChessboardCorners(grayL, chessboardSize, None)
    retR, cornersR = cv.findChessboardCorners(grayR, chessboardSize, None)
    
    # If found, add object points, image points (after refining them)
    if retL and retR == True:
        objpoints.append(objp)
        
        cornersL = cv.cornerSubPix(grayL, cornersL, (11,11), (-1,-1), criteria)
        imgpointsL.append(cornersL) # 將計算好的cornersL值依序放入imgpointsL列表中
        
        cornersR = cv.cornerSubPix(grayR, cornersR, (11,11), (-1,-1), criteria)
        imgpointsR.append(cornersR)
        
        # Draw and display the corners
        cv.drawChessboardCorners(imgL, chessboardSize, cornersL, retL)
        cv.imshow('img left', imgL)
        cv.drawChessboardCorners(imgR, chessboardSize, cornersR, retR)
        cv.imshow('img right', imgR)
        cv.waitKey(200) # 給予時間(ms)繪製記號，否則無畫面。
        
cv.destroyAllWindows()


############## CALIBRATION #######################################################
# cv.getOptimalNewCameraMatrix: 進一步調整內部參數矩陣
retL, CameraMatrixL, distL, rvecsL, tvecsL = cv.calibrateCamera(objpoints, imgpointsL, frameSize, None, None)
# heightL, widthL, channelsL = imgL.shape
# newCameraMatrixL, roi_L = cv.getOptimalNewCameraMatrix(cameraMatrixL, distL, (widthL, heightL), 1, (widthL, heightL))

retR, CameraMatrixR, distR, rvecsR, tvecsR = cv.calibrateCamera(objpoints, imgpointsR, frameSize, None, None)
# heightR, widthR, channelsR = imgR.shape
# newCameraMatrixR, roi_R = cv.getOptimalNewCameraMatrix(cameraMatrixR, distR, (widthR, heightR), 1, (widthR, heightR))


########## Stereo Vision Calibration #############################################

flags = 0
flags |= cv.CALIB_FIX_INTRINSIC
# Here we fix the intrinsic camara matrixes so that only Rot, Trns, Emat and Fmat are calculated.
# Hence intrinsic parameters are the same 

criteria_stereo = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.0001)

# This step is performed to transformation between the two cameras and calculate Essential and Fundamenatl matrix
retStereo, CameraMatrixL, distL, CameraMatrixR, distR, rot, trans, essentialMatrix, fundamentalMatrix = cv.stereoCalibrate(objpoints, imgpointsL, imgpointsR, CameraMatrixL, distL, CameraMatrixR, distR, grayL.shape[::-1], criteria_stereo, flags)


########## Stereo Rectification #################################################
# alpha: 由0~1，0表示不允許無效像素(無黑色區域)，1表示允許無效像素(以黑色填補)，-1: 自動調整黑色區塊
rectL, rectR, projMatrixL, projMatrixR, Q, roi_L, roi_R = cv.stereoRectify(CameraMatrixL, distL, CameraMatrixR, distR, grayL.shape[::-1], rot, trans, flags=cv.CALIB_ZERO_DISPARITY, alpha=-1) 

stereoMapL = cv.initUndistortRectifyMap(CameraMatrixL, distL, rectL, projMatrixL, grayL.shape[::-1], cv.CV_16SC2)
stereoMapR = cv.initUndistortRectifyMap(CameraMatrixR, distR, rectR, projMatrixR, grayR.shape[::-1], cv.CV_16SC2)

print("Saving parameters!")
cv_file = cv.FileStorage('stereoMap.xml', cv.FILE_STORAGE_WRITE)

cv_file.write('stereoMapL_x',stereoMapL[0])
cv_file.write('stereoMapL_y',stereoMapL[1])
cv_file.write('stereoMapR_x',stereoMapR[0])
cv_file.write('stereoMapR_y',stereoMapR[1])

cv_file.write('projMatrixL',projMatrixL)
cv_file.write('projMatrixR',projMatrixR)

cv_file.write('Q',Q)

# 釋放記憶體與關閉檔案
cv_file.release() 
print("Done!")

