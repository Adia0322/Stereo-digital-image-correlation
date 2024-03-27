# -*- coding: utf-8 -*-
"""
影像校正
"""
import cv2 as cv

# Camera parameters to undistort and rectify images
cv_file = cv.FileStorage()
cv_file.open('stereoMap.xml', cv.FileStorage_READ)

stereoMapL_x = cv_file.getNode('stereoMapL_x').mat()
stereoMapL_y = cv_file.getNode('stereoMapL_y').mat()
stereoMapR_x = cv_file.getNode('stereoMapR_x').mat()
stereoMapR_y = cv_file.getNode('stereoMapR_y').mat()


def undistortRectify(frameL, frameR):

    # Undistort and rectify images
    undistortedL= cv.remap(frameL, stereoMapL_x, stereoMapL_y, cv.INTER_LANCZOS4, cv.BORDER_CONSTANT, 0)
    undistortedR= cv.remap(frameR, stereoMapR_x, stereoMapR_y, cv.INTER_LANCZOS4, cv.BORDER_CONSTANT, 0)


    return undistortedL, undistortedR














