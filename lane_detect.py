from moviepy.editor import VideoFileClip
import numpy as np
import cv2
import pickle
from collections import deque
from camera import correctCameraDistortion
import os.path
import glob

class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        #average x values of the fitted line over the last n iterations
        self.bestx = None
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        #radius of curvature of the line in some units
        self.radius_of_curvature = None
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        #x values for detected line pixels
        self.allx = None
        #y values for detected line pixels
        self.ally = None
        self.badframenum = 0
        self.badframe = False

    #
    # validate the new values, before we put them into the line
    #
    # return false if we have too many bad frames
    def validate_and_update(self,allx,ally,fitx,curverad,position):

        self.badframe = False

        #
        #  validate before changing
        #
        if self.allx is not None:

            # check to see if the curverad changes too much
            #print("=====")
            percentage = abs(((self.radius_of_curvature - curverad)/curverad))
            offset_percentage = abs(((np.mean(self.allx) - np.mean(allx))/np.mean(allx)))
            #print(self.radius_of_curvature,' ', curverad,' ',percentage)

            if ( percentage > 15 and self.badframenum < 5):
               #print("TOO FAR?")
               self.badframe = True
               self.badframenum = self.badframenum + 1
               return

            self.badframenum=0
        #
        #  update
        #
        self.allx = allx
        self.ally = ally

        self.current_fit = fitx #np.polyfit(self.allx, self.ally, 2)


        self.recent_xfitted.append(self.current_fit)
        # if we are larger than our averaging window, drop one
        if (len(self.recent_xfitted) > 5):
           self.recent_xfitted = self.recent_xfitted[1:]

        # best_fit is the average of the recent_xfitted
        if (len(self.recent_xfitted) > 1):
            #self.best_fit = np.mean(self.recent_xfitted)
            self.best_fit = (self.best_fit * 4 + self.current_fit) / 5
        else:
            self.best_fit = self.current_fit

        self.radius_of_curvature = curverad
        return


# Edit this function to create your own pipeline.
def color_gradient_pipeline(img, sx_thresh=(20, 255), s_thresh=(170, 255), l_thresh=(30, 255)):

    '''
    CONVERT FROM RGB TO HSL and just use one channel as "GRAYSCALE"
    move grayscale conversion out
    then we can use each function for grayscale, or h, or s, or l
    '''

    # Make a copy of the original image
    img = np.copy(img)

    # Convert to HSL color space and separate the h channel
    hsl = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hsl[:,:,1]
    s_channel = hsl[:,:,2]
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    # Threshold on the S channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1


    # Threshold on the L channel
    l_binary = np.zeros_like(l_channel)
    l_binary[(l_channel >= l_thresh[0]) & (l_channel <= l_thresh[1])] = 1

    # Stack each channel
    # finally, return the combined binary image
    binary = np.zeros_like(sxbinary)
    binary[((l_binary == 1) & (s_binary == 1) | (sxbinary == 1))] = 1
    # binary = 255 * np.dstack((binary, binary, binary)).astype('uint8')

    return binary


def projectImageToBird(image, inverse=False):
    '''
    Function used to define
    '''
    src = np.float32([[585, 460], [203, 720], [1127, 720], [695, 460]])
    dst = np.float32([[320, 0], [320, 720], [960,720], [960, 0]])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    img_size = (image.shape[1], image.shape[0])
    if inverse:
        binary_warped = cv2.warpPerspective(image, Minv, img_size, flags=cv2.INTER_LINEAR)
    else:
        binary_warped = cv2.warpPerspective(image, M, img_size, flags=cv2.INTER_LINEAR)
    return binary_warped

def findLanesByHistogram(binary_warped):
    # Assuming you have created a warped binary image called "binary_warped"
    # # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    #
    # # Find the peak of the left and right halves of the histogram
    # # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)

    binary_warped = np.uint8(binary_warped*255)

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()

    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 75
    # Set minimum number of pixels found to recenter window
    minpix = 25
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx,lefty,rightx,righty

def estimate_curvature(leftx, rightx, ploty):
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)

    y_eval = np.max(ploty)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.abs(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.abs(2*right_fit_cr[0])
    # Now our radius of curvature is in meters
    return left_curverad, right_curverad


def drawOutput(orig, warped, left_fitx, right_fitx, ploty):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    newwarp=projectImageToBird(color_warp, inverse=True)
    # Combine the result with the original image
    result = cv2.addWeighted(orig, 1, newwarp, 0.3, 0)
    return result


left_line = Line()
right_line = Line()
def lanePipeline(image):
    global left_line
    global right_line
    # correct the camera distortion
    orig_img = correctCameraDistortion(image)

    # threshold the image and try to pull just the lines out
    img = color_gradient_pipeline(orig_img)
    # switch to birds eye projection
    warped=projectImageToBird(img)
    # fix the image to be 0-255 instead of 0-1
    binary_warped = np.uint8(warped*255)

    # find the lanes using the historgram method
    leftx,lefty,rightx,righty = findLanesByHistogram(binary_warped)

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # find the curvature of the left and right lane marker
    left_curverad,right_curverad = estimate_curvature(left_fitx,right_fitx,ploty)
    position = findPositionInLane((img.shape[1]/2),left_fitx,right_fitx)

    line_is_parallel=False
    if (np.abs(left_fit[0]-right_fit[0]) < 0.0004 and np.abs(left_fit[1]-right_fit[1])<0.6 ):
      line_is_parallel=True

    lane_width = np.median(rightx) - np.median(leftx)
    # let's skip when they aren't parallel -- we should probably only do this for 5 frames (TODO)
    if lane_width < 770:
        if (left_line.best_fit is None or right_line.best_fit is None or line_is_parallel):
           left_line.validate_and_update(leftx,lefty,left_fitx,left_curverad,position)
           right_line.validate_and_update(rightx,righty,right_fitx,right_curverad,position)

    result = drawOutput(orig_img, warped, left_line.best_fit,right_line.best_fit,ploty)

    text = 'Radius of Curvature: {:.0f}m'.format((left_curverad+right_curverad)/2.0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(result,text,(10,25), font, 1,(255,255,255),2)

    leftright='right'
    if (position<0):
      position=position*-1
      leftright='left'

    text = 'Vehicle is {:.2f}m {:s} of center'.format(position,leftright)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(result,text,(10,50), font, 1,(255,255,255),2)

    if lane_width >=770:
        text = 'Lane is too wide'
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(result,text,(10,75), font, 1,(255,255,255),2)
    return result

def findPositionInLane(centerx, leftx,rightx):
    center = (rightx + leftx) / 2
    position = (centerx - center[719]) * 3.7 / 700
    return position
