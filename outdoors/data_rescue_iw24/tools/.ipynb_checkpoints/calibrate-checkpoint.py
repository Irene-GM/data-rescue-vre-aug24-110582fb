import cv2
import numpy as np
import os
import logging
from .helpers import get_filenames
from .image import read_image

logging.basicConfig()
logger = logging.getLogger("Calibrate")
logger.setLevel(logging.DEBUG)

def calibrate(calibration_dir):

    calibration_images = get_filenames(calibration_dir, extension="CR3")

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((10*7,3), np.float32)
    objp[:,:2] = np.mgrid[0:7,0:10].T.reshape(-1,2)
    
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    
    for fname in calibration_images:

        logger.debug(f"Calibrating on image: '{fname}'")
        img = read_image(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (7,10), None)
        
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
        
        corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)
    
        # Draw and display the corners
        cv2.drawChessboardCorners(img, (7,10), corners2, ret)
        cv2.imwrite(os.path.splitext(fname)[0] + ".png", img)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        mean_error += error
    
    logger.info("Total calibration error: {}".format(mean_error/len(objpoints)))

    with open(os.path.join(calibration_dir, "calibration_parameters.npy"), "wb") as f:
        np.save(f, mtx)
        np.save(f, dist)
        np.save(f, newcameramtx)
        np.save(f, roi)

    return mtx, dist, newcameramtx, roi

def get_calibration_settings(calibration_dir):

    with open(os.path.join(calibration_dir, "calibration_parameters.npy"), "rb") as f:
        mtx = np.load(f)
        dist = np.load(f)
        newcameramtx = np.load(f)
        roi = np.load(f)

    return mtx, dist, newcameramtx, roi


def undistort_image(image, mtx, dist, newcameramtx, roi):

    dst = cv2.undistort(image, mtx, dist, None, newcameramtx)

    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]

    return dst
