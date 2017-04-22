## **Advanced Lane Finding Project**

Without all of the lectures and quizes in this module, I would not have stood a chance. I used a couple of different files to complete this project. I used the notebook titled, "P4ScratchPad.ipynb" to do my initial work and picture grabbing. I think that I like to prototype in the notebook and do the final code in a text editor (Visual Studio). My video output file is named, "FinalOutputMovie.py". I created a Lane class for the video input, that got way unruly and needs to be broken out better, it just grew and grew. I also copied that into another notebook named, "FinalOutputMovie.ipynb".

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./writeupPics/calibration.png "Calibration"
[image2]: ./writeupPics/Undistorted.png "Undistorted"
[image3]: ./writeupPics/BinaryResult.png "Binary Result"
[image4]: ./writeupPics/warped.png "Warped Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
# Writeup / README

*1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.*  

You're reading it!

# Camera Calibration

*1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.*

The code for this step is contained in the second code cell of the Juypter notebook(P4ScratchPad.ipynb) titled "Step 1: Calibrate the Camera". Or in lines 247-292 of the FinalOutputMovie.py file. Most of the code was taken from the lectures and quizes. 

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection. Most of the checkerboards were able to be read as a 9 x 6 board. I could calibrate using 17 of the 20 images. The other three could not be read as 9 x 6. I figured 17/20 was good enough to return the values that I needed.

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

# Pipeline (single images)

*1. Provide an example of a distortion-corrected image.*
To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:

![alt text][image2]

The real heros are the people behind cv2. To undistort my image all I did was take the calibration matrix, distortion coefficients, and a test image and plugged them into cv2.undistort and I got back an undistorted image. 

The code for this step is contained in the second code cell of the Juypter notebook(P4ScratchPad.ipynb) titled "Step 2: Example of Distortion Corrected Image". Or in lines 296-302 of the FinalOutputMovie.py file. Most of the code was taken from the lectures and quizes.

*2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.*

I used a combination of color and gradient thresholds to generate a binary image. I used a combination of Sobel X and the S-channel from the HLS color model. I used these, because they worked in lecture/quizes and they looked good on my test image. Here's an example of my output for this step:

![alt text][image3]

The code for this step is contained in the second code cell of the Juypter notebook(P4ScratchPad.ipynb) titled "Step 3: Thresholding". Or in lines 304-336 of the FinalOutputMovie.py file. Most of the code was taken from the lectures and quizes.

*3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.*

The code for my perspective transform includes a function called `warped()`. The `warped()` function takes as inputs an image (`img`). The function assumes that it is recieving a binary image. The function uses cv2.getPerspectiveTransform to obtain to calculate the perspective transform matrix used to seed the cv2.warpPerspective function. I also used the getPerspectiveTransform to obtain the inverse of the perpective transform matrix. I will use this later to 'unwarp' the image. I chose the hardcode the source and destination points need for cgetPerspectiveTransform function by using MS Paint (didn't know it could be used for this until now).

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 581, 460      | 185, 0        | 
| 702, 460      | 1095, 0      |
| 1110, 720     | 1095, 720      |
| 200, 720      | 185, 720        |

I verified that my perspective transform was working as expected by drawing the `srcs` and `dsts` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image. This image also contains the polyfit lines found in the next step.

![alt text][image4]

The code for this step is contained in the second code cell of the Juypter notebook(P4ScratchPad.ipynb) titled "Step 4: Perspective Transform". Or in lines 338-356 of the FinalOutputMovie.py file. Most of the code was taken from the lectures and quizes.

*4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial.*

To find the lane-line pixel and fit their positions, I used what we did in lecture and on the quizes. My function expects to hae a binary image input into it. I first created a histogram to find the most common points of the input image. I then used a sliding window to identify the non-zero pixels in the image and created a giant list. I then used numpy's polyfit to fit a polynomial fit nice line to to the found pixels. Below is what this step looks like:

![alt text][image4]

The code for this step is contained in the second code cell of the Juypter notebook(P4ScratchPad.ipynb) titled "Step 5: Cuved Lines". Or in lines 108-183 of the FinalOutputMovie.py file. Most of the code was taken from the lectures and quizes.

*5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.*

To find the lane-line pixel and fit their positions, I used what we did in lecture and on the quizes. I first converted pixels space to real world meters. The approximate ratio was given to us. I then fit the left and right lane polynomials to x, y in real world space. That was followed by finding the radii of the curvature. To find the center of the lane, I assumed the lane center should always be in the middle of the image at pixel point 640. I then took where the left and right lane lines were found the lane width and added where the left line was versus where it should be. I then took the difference of perfect middle and where I found the middle to be.

The code for this step is contained in the second code cell of the Juypter notebook(P4ScratchPad.ipynb) titled "Step 6: Find Lane Curvature and Center". Or in lines 156-221 of the FinalOutputMovie.py file. Most of the code was taken from the lectures and quizes.

*6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.*

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image5]

---

# Pipeline (video)

*1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).*

Here's a [link to my video result](./project_video.mp4)

---

# Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

