# Advanced Lane Finding Project

#### The goals / steps of this project are the following:

1. Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
2. Apply a distortion correction to raw images.
3. Use color transforms, gradients, etc., to create a thresholded binary image.
4. Apply a perspective transform to rectify binary image ("birds-eye view").
5. Detect lane pixels and fit to find the lane boundary.
6. Determine the curvature of the lane and vehicle position with respect to center.
7. Warp the detected lane boundaries back onto the original image.
8. Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # "Image References"

[image1]: ./images/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Image"
[image2_und]: ./test_images/undistorted/undst_test1.jpg "Undistorted road image"
[image3]: ./images/binary_combo_example.png "Binary Example"
[image4]: ./images/warped_straight_lines.png "Warp Example"
[image5_win]: ./images/color_fit_window.png "Fit using sliding window method"
[image5_fit]: ./images/color_fit_fit.png "Fit from a previous fit"
[image6]: ./test_images/patched/patched_test1.jpg "Output"
[video1]: ./project_video_output.mp4 "Output video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

#### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.   

You're reading it!

### Camera Calibration

#### 1. Computing the camera matrix and distortion coefficients for camera calibration.

The code for this step is contained in the second and third code cells of the IPython notebook `P4-Advanced_Lane_Detection.ipynb`.  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. For simplicity, I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for all calibration images.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration `mtx` and distortion coefficients `dist` using the `cv2.calibrateCamera()` function.  

```ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)```

I applied this distortion correction to the calibration chessboard images using the `cv2.undistort()` function, and obtained results like this:

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

After undistortion, the image becomes like the following one. Objects near the edges now appear closer.

![alt text][image2_und]

#### 2. Used color transforms and gradients to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image, and it is implemented as:

```python
def combined_filter(img, sobel_kernel = 3, mag_th = (0, 255), dir_th=(0, np.pi/2), s_th=(0, 255)):
    sobel_binary = sobel_threshold(img, sobel_kernel=sobel_kernel, mag_th=mag_th, dir_th=dir_th)
    hls_binary = hls_threshold(img, thresh=s_th)
    color_binary = np.dstack(( np.zeros_like(sobel_binary), sobel_binary, hls_binary))
    return color_binary
```

The two threshold filter are:

1. Gradient filter `sobel_threshold(img, sobel_kernel=sobel_kernel, mag_th=mag_th, dir_th=dir_th)` which selects points with gradient magnitude within `mag_th` and direction within `dir_th`.
2. Color filter `hls_threshold(img, thresh=s_th)` which first converts the image to HSL color space, then select points whose S values are within `s_th`. I have also tried to use the H value and L value, but they are not good indicator of lane line points.

Here's an example of my output after applying both filters, where the green points are identified by the gradient filter and the blue points are identified by the color filter. Note the area of interest filter (used in Project 1 for lane detection) is not necessary here. As we will see in the next section, the perspective transformation effectively filters away all points not in our area of interest, such as the trees and the sky.

![alt text][image3]

#### 3. Performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform is located in the 5th cell, where I calculated the perspective transform matrix `M` usign the function `cv2.getPerspectiveTransform(src, dst)`.  Then any image can be transformed using the function `warped = cv2.warpPerspective(img, M, img_size)`.  I chose the hand pick the source points from a straight lane image, and destination points as a rectangle with the ratio from the left and right edges to the x center same as the original image.

The coordinates of the source and destination points are listed below:

|  Source   | Destination |
| :-------: | :---------: |
| 596, 451  |   367, 0    |
| 690, 451  |   946, 0    |
| 1048, 682 |  940, 700   |
| 276, 682  |  367, 700   |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto another straight lane image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Full pipeline to convert an input image to a warped binary output.

For each input image, we need to

1. undistort the image,
2. filter away uninteresting points using the gradient filter and the color filter,
3. perform perspective transform to get a "bird-eye" view of the lane line.

This is implemented as the following function:

```python
def image_to_binary(img, mtx, dist, M,
                   sobel_kernel=9, mag_th=(50, 255), dir_th=(0.3, 1.15), s_th=(150, 255)):
    undistorted = cv2.undistort(img, mtx, dist, None, mtx)
    filtered = combined_filter(img, sobel_kernel, mag_th, dir_th, s_th)
    filtered_binary = np.zeros(filtered.shape[:2])
    filtered_binary[(filtered[:,:,1]==1) | (filtered[:,:,2]==1)] = 1
    warped = cv2.warpPerspective(filtered_binary, M, filtered_binary.shape[::-1])
    return warped
```

#### 5. Identified lane-line pixels and fit their positions with a polynomial.

Even after the combined filter, there are still points in the binary output that are not lane line points. Two ways are used to further filter away other points.

The first method is the sliding window method `find_line_pts_sliding_window()` that divide the image height into several ranges, and only search for line points within a certain width window at each height range. The position of the bottom window is determined from a histogram of points density at the bottom part of the image, and the positions of other height ranges are determined as the mean x position of all points in the lower height range.

A 2nd order polynomial fit is calculated from all points within the windows.

![alt text][image5_win]

Alternatively, if the lane fit from a previous frame is known, we do not need to search from scratch. Assuming the difference between the new frame and previous frames is small, function `find_line_pts_from_fit()` only searches point within a small window of the previous successful fit. The following image shows the searching window of this method.

![alt text][image5_fit]

#### 6. Calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

This part is implements in function `left_curverad, right_curverad, car_pos = get_curvature_offset(left_fit, right_fit, y_eval)`. First the conversion rate between pixel and real world distance,` ym_per_pix` and `xm_per_pix` are identified using the standard straight lane image. Then the curvature is calculated from the 2nd order polynomial fits, and the distance from the lane center and image center (assumed to be car position) is calculated.

#### 7. An example image of the results plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in function `draw_lanes_on_image(img, left_fit, right_fit, M_inv)`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Image processing pipeline.

The process pipeline for video uses the `Line()` class to store lane points identified in previous frames, up to 10 frames. If a line fit is identified in previous frame, the pipeline will first try to find lane point using knowledge on the possible fit. If not enough points are found to justify the possible fit, the sliding window method will be used instead.

#### 2. Link to the final video output.

Here's a [link to my video result](./project_video_output.mp4). For diagnose purpose, here is also a [link to the perspective transformed lane view video](./project_video_bird_view.mp4).

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The line fit depends heavily on the quality of identified lane points. In the warped binary image, if too many lane points are missing or too many irrelevant points exist, the fit result will become unreliable. By using the average fit of most recent 10 frames instead of the current frame, the video pipeline is able to recover from occasional bad input images. A better filter with features more than just gradient magnitude, gradient direction, and S channel value should help to better identify candidate lane points.
