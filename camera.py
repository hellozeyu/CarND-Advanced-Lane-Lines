import numpy as np
import glob
import cv2
import pickle

def camera_calibrate():
    # Camera calibration
    # prepare object points
    nx = 9 # number of inside corners in x
    ny = 6 # number of inside corners in y

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((nx*ny,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob('camera_cal/calibration*.jpg')
    img_size = None
    # Step through the list and search for chessboard corners
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        img_size = (img.shape[1], img.shape[0])
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display the corners
            cv2.drawChessboardCorners(img, (nx,ny), corners, ret)

    # Do camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    # Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
    dist_pickle = {}
    dist_pickle["mtx"] = mtx
    dist_pickle["dist"] = dist
    pickle.dump(dist_pickle, open( "camera_cal_output/dist_pickle.p", "wb" ))

    print('Calibration Done.')


def correctCameraDistortion(image):
    '''
    This function removes the camera distortion and will be part of our pipeline
    '''
    dist_pickle = pickle.load(open('camera_cal_output/dist_pickle.p', "rb" ) )
    mtx = dist_pickle["mtx"]
    dist = dist_pickle["dist"]
    img_size = (image.shape[1], image.shape[0])
    dst = cv2.undistort(image, mtx, dist, None, mtx)
    return dst
