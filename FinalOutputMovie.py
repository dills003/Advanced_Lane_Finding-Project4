#**************************************************************************
# Udacity Behavioral Cloning - Project #4
# This is my attempt at Project #4. 
# The goal of this project is to write a software pipeline 
# to identify the lane boundaries in a video from a front-facing camera 
# on a car
#**************************************************************************
############################################################################################################################################
#STEP 0: IMPORT ALL OF THE USEFUL PACKAGES
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
from IPython.display import HTML
import imageio
import os
imageio.plugins.ffmpeg.download()
#%matplotlib inline

############################################################################################################################################
#STEP 1: CREATE MY CLASS
class Lane():
    #a class that keeps track of road features/math
    def __init__(self):
        self.ploty = []
        self.left_fit = []
        self.right_fit = []
        self.left_fitx = []
        self.right_fitx = []
        self.left_curverad = 0.0
        self.right_curverad = 0.0
        self.movingAvgleft = []
        self.movingAvgright = []
        self.rightOrLeft = ""
        self.carMissedByMeterABS = 0.0
        self.leftmovingAvgA = []
        self.leftmovingAvgB = []
        self.leftmovingAvgC = []
        self.rightmovingAvgA = []
        self.rightmovingAvgB = []
        self.rigntmovingAvgC = []
        
    #fill moving average arrays up with first value
    def movingAvgSetup(self, fitl, fitr, toAvg=5):
        self.leftmovingAvgA = np.full(toAvg, fitl[0])
        self.leftmovingAvgB = np.full(toAvg, fitl[1])
        self.leftmovingAvgC = np.full(toAvg, fitl[2])
        self.rightmovingAvgA = np.full(toAvg, fitr[0])
        self.rightmovingAvgB = np.full(toAvg, fitr[1])
        self.rightmovingAvgC = np.full(toAvg, fitr[2])
    
    #takes average and weights, changes the fit coes
    def movingAvgNew(self, fitl, fitr, weights=None):
        self.leftmovingAvgA = self.leftmovingAvgA[1:] #slice the first off
        self.leftmovingAvgB = self.leftmovingAvgB[1:]
        self.leftmovingAvgC = self.leftmovingAvgC[1:]
        self.rightmovingAvgA = self.rightmovingAvgA[1:]
        self.rightmovingAvgB = self.rightmovingAvgB[1:]
        self.rightmovingAvgC = self.rightmovingAvgC[1:]
        
        self.leftmovingAvgA = np.append(self.leftmovingAvgA, fitl[0]) #append the newest reading
        self.leftmovingAvgB = np.append(self.leftmovingAvgB, fitl[1])
        self.leftmovingAvgC = np.append(self.leftmovingAvgC, fitl[2])
        self.rightmovingAvgA = np.append(self.rightmovingAvgA, fitr[0])
        self.rightmovingAvgB = np.append(self.rightmovingAvgB, fitr[1])
        self.rightmovingAvgC = np.append(self.rightmovingAvgC, fitr[2])
        
        self.left_fit[0] = np.average(self.leftmovingAvgA, weights=weights) #new fit values
        self.left_fit[1] = np.average(self.leftmovingAvgB, weights=weights)
        self.left_fit[2] = np.average(self.leftmovingAvgC, weights=weights)
        self.right_fit[0] = np.average(self.rightmovingAvgA, weights=weights)
        self.right_fit[1] = np.average(self.rightmovingAvgB, weights=weights)
        self.right_fit[2] = np.average(self.rightmovingAvgC, weights=weights)
        
    #Taken from lecture and quizes   
    def curvedLinesFirst(self, binary_warped):
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[int(binary_warped.shape[0] / 2):,:], axis=0)
        # Create an output image to draw on and visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0] / 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = np.int(binary_warped.shape[0] / nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
            cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean
            # position
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

        # Fit a second order polynomial to each
        self.left_fit = np.polyfit(lefty, leftx, 2)
        self.right_fit = np.polyfit(righty, rightx, 2)

        # Generate x and y values for plotting
        self.ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
        self.left_fitx = self.left_fit[0] * self.ploty ** 2 + self.left_fit[1] * self.ploty + self.left_fit[2]
        self.right_fitx = self.right_fit[0] * self.ploty ** 2 + self.right_fit[1] * self.ploty + self.right_fit[2]
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    
        self.movingAvgSetup(self.left_fit, self.right_fit, 8)
        
    #Taken from lecture and quizes   
    def curvedLinesNext(self, binary_warped):
        left_fitOld = self.left_fit.copy()
        right_fitOld = self.right_fit.copy()
    
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = 100
        left_lane_inds = ((nonzerox > (self.left_fit[0]*(nonzeroy**2) + self.left_fit[1]*nonzeroy + self.left_fit[2] - margin)) & (nonzerox < (self.left_fit[0]*(nonzeroy**2) + self.left_fit[1]*nonzeroy + self.left_fit[2] + margin))) 
        right_lane_inds = ((nonzerox > (self.right_fit[0]*(nonzeroy**2) + self.right_fit[1]*nonzeroy + self.right_fit[2] - margin)) & (nonzerox < (self.right_fit[0]*(nonzeroy**2) + self.right_fit[1]*nonzeroy + self.right_fit[2] + margin)))  

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        
        #fit the data
        self.left_fit = np.polyfit(lefty, leftx, 2)
        self.right_fit = np.polyfit(righty, rightx, 2)
        
        #apply the moving average
        self.movingAvgNew(self.left_fit, self.right_fit, weights=None)
        # Generate x and y values for plotting
        self.ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        self.left_fitx = self.left_fit[0]*self.ploty**2 + self.left_fit[1]*self.ploty + self.left_fit[2]
        self.right_fitx = self.right_fit[0]*self.ploty**2 + self.right_fit[1]*self.ploty + self.right_fit[2]
     
    #Gets the lane data, taken from lecture and quizes
    def getLaneData(self):
        y_eval = np.max(self.ploty)
        self.left_curverad = ((1 + (2*self.left_fit[0]*y_eval + self.left_fit[1])**2)**1.5) / np.absolute(2*self.left_fit[0])
        self.right_curverad = ((1 + (2*self.right_fit[0]*y_eval + self.right_fit[1])**2)**1.5) / np.absolute(2*self.right_fit[0])
        # Example values: 1926.74 1908.48

        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meters per pixel in x dimension

        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(self.ploty*ym_per_pix, self.left_fitx*xm_per_pix, 2) #changed to fitted line
        right_fit_cr = np.polyfit(self.ploty*ym_per_pix, self.right_fitx*xm_per_pix, 2) #changed to fitted line
        # Calculate the new radii of curvature
        self.left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        self.right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
        # Now our radius of curvature is in meters
        #print(left_curverad, 'm', right_curverad, 'm')
        # Example values: 632.1 m    626.2 m

        #center of lane stuff, assume lane should always be in center, drift is caused by lane to shift left or right
        centerOfLanePixels = 640 #1280/2 for the image
        leftLane = self.left_fitx[-1] #should be the bottom
        rightLane = self.right_fitx[-1]
        laneWidthPixels = rightLane - leftLane #to stay positive gives lane width
        carCenterPixels = (laneWidthPixels/2) + leftLane #should be center
        carMissedByPixels = centerOfLanePixels - carCenterPixels
        carMissedByMeters = carMissedByPixels * xm_per_pix

        self.rightOrLeft = "" #for the overdisplay
        if carMissedByMeters > 0: #for display purposes only
            self.rightOrLeft = "Left"
        else:
            self.rightOrLeft = "Right"
    
        self.carMissedByMetersABS = np.absolute(carMissedByMeters) #since we are giving a left or right
    
    # Create an image to draw the lines on
    # From lecture and help guide and prints info over screen, used for tolerances/filtering during debug
    def backtoRoad(self, img, Minv):
        warp_zero = np.zeros_like(img).astype(np.uint8)

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([self.left_fitx, self.ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([self.right_fitx, self.ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(warp_zero, np.int_([pts]), (0,255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(warp_zero, Minv, (img.shape[1], img.shape[0])) 
        # Combine the result with the original image
        result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)
        font = cv2.FONT_HERSHEY_COMPLEX
        cv2.putText(result,"Radius of Left Curvature: %dm" %self.left_curverad,(10,60), font, 2,(255,255,255),5,cv2.LINE_AA)
        cv2.putText(result,"Radius of Right Curvature: %dm" %self.right_curverad,(10,125), font, 2,(255,255,255),5,cv2.LINE_AA)
        cv2.putText(result,"Car is %0.3fm to the %s" %(self.carMissedByMetersABS, self.rightOrLeft),(10,190), font, 2,(255,255,255),5,cv2.LINE_AA)

        return result
    
############################################################################################################################################
#STEP 2: CALIBRATE THE CAMERA

#I will compute the camera matrix and distorsion coefficients for the front mounted camera.
#From Camera Calibration Lecture Link - direct rip off almost
#prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
checkersInX = 9 #the number of dark/white corners counted left to right
checkersInY = 6 #the number of dark/white cornerrs counted top to bottom

objp = np.zeros((checkersInX*checkersInY,3), np.float32)
objp[:,:2] = np.mgrid[0:checkersInX, 0:checkersInY].T.reshape(-1,2) #sweet new way of filling array

#Arrays to store object points and image points from all the images
objpoints = [] #3D points from what the picture is actually like in real life
imgpoints = [] #2D points from the picture

# Make a list of calibration images
images = glob.glob('camera_cal/calibration*.jpg') #grabs all of my images with calibration# name, awesome!

print("Number of calibration images: " + str(len(images))) #how many images
mytruecounter = 0 #how many times teh cv2.findCheckers worked, found not always, but 9 and 6 work the best

#Step through the list and search for chessboard corners
for idx, fname in enumerate(images):
    img = cv2.imread(fname) #read in the images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #using cv2 to read in, thus BGR to grayscale

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (checkersInX,checkersInY), None)   
    
    # If found, add object points, image points
    if ret == True:
        mytruecounter = mytruecounter + 1
        objpoints.append(objp)
        imgpoints.append(corners)

print("Number of checker finds: " + str(mytruecounter))

#From Camera Calibration Lecture Link
#Test undistortion on an image, tested on image1 which failed in checker read
img = cv2.imread('camera_cal/calibration1.jpg')
img_size = (img.shape[1], img.shape[0])

# Do camera calibration given object points and image points
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)

############################################################################################################################################
#STEP 3: USEFUL FUNCTIONS

#DISTORTION CORRECTION
#read in an image and display it undistorted, taken from lecture and quizes
def undistortMe(imgage):
    dst1 = cv2.undistort(imgage, mtx, dist, None, mtx) #undistort the images
    dst1 = cv2.cvtColor(dst1, cv2.COLOR_BGR2RGB) #flip back to rgb
    
    return dst1

#THRESHOLDING
#From the color and gradient lecture, going to use sobel x and hls, s-channel, taken from lecture an quizes
def threshold(img, s_thresh=(170, 255), sx_thresh=(20, 100)):
    img = np.copy(img)
    
    #convert to HLS color space and separate the S channel
    hsl = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    s_channel = hsl[:,:,2]
    
    #grayscale the image
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Sobel x
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    
    # Stack each channelS
    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary))
    
    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1

    return combined_binary

#PERSPECTIVE TRANSFORM
#From Undistort and Transform Quiz and 'How I Did It'
def warped(img, mtx, dist):
    #get the shape
    img_size = (img.shape[1], img.shape[0])

    # For source points I'm grabbing the outer four detected corners from MS Paint
    srcs = np.float32([[581,460],[702,460],[1110,img_size[1]],[200,img_size[1]]])
    # For destination points, 
    dest = np.float32([[185,0],[1095,0],[1095,720],[185,720]])
    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(srcs, dest)
    Minv = cv2.getPerspectiveTransform(dest, srcs)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(img, M, img_size)

    # Return the resulting image and matrix
    return warped, Minv

############################################################################################################################################
#STEP 4: PIPELINE
lane = Lane()

def decoration(func): #from a pile of examples and tuturials, sees if pipleline has bee called
    def wrapper(*args, **kwargs):
        wrapper.called += 1
        return func(*args, **kwargs)
    wrapper.called = 0
    wrapper.__name__ = func.__name__
    return wrapper
    

#Steps for Advanced lane Finding - After all Useful Packages and Calibration Takes Place
@decoration
def myPipeline(image):
    #make a copy
    #originalImage = cv2.imread(image) #uncomment if reading a single image
    originalImage = image #cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #flip back to rgb
       
    #distortion correction
    undistorted = undistortMe(image) #returns an undistorted rgb image
    
    #binary image
    binary = threshold(undistorted) #returns a binary image, stacked S(HLS) channel and sobel X
    
    #binary_warped
    binary_warped, Minv = warped(binary, mtx, dist)
        
    #perspective birds-eye view/check this
    if myPipeline.called < 2:
        lane.curvedLinesFirst(binary_warped)
    else:
        lane.curvedLinesNext(binary_warped)
    
    #get curvature and center of lane info
    lane.getLaneData()
    
    #final output
    finalresult = lane.backtoRoad(originalImage, Minv)
        
    return finalresult
 ############################################################################################################################################  
 #MOVIE MAKING
 #for the movie capture from Lab #1
myVideo = 'myVideo.mp4'
clip1 = VideoFileClip("project_video.mp4")
myClip = clip1.fl_image(myPipeline) #NOTE: this function expects color images!!
#%time myClip.write_videofile(myVideo, audio=False)