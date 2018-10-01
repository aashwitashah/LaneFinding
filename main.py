# Do relevant imports
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os
from moviepy.editor import VideoFileClip
from IPython.display import HTML

def grayscale(img):
    return cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

def canny(img, low_threshold, high_threshold):
        return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
        return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """

    #defining a blank mask to start with
    mask = np.zeros_like(img)   

    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[0, 0, 255], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  

    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1,y1,x2,y2 in line:
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img, lines

def hough_lines_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    return cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)

def average_slope_intercept(lines):
    left_lines    = [] # (slope, intercept)
    left_weights  = [] # (length,)
    right_lines   = [] # (slope, intercept)
    right_weights = [] # (length,)

    for line in lines:
        for x1, y1, x2, y2 in line:        
            if x2==x1:
                continue # ignore a vertical line

            if y2==y1:
                continue # ignore a horizontal line
        
            slope = (y2-y1)/(x2-x1)
            intercept = y1 - slope*x1
            length = np.sqrt((y2-y1)**2+(x2-x1)**2)
            if slope < 0: # y is reversed in image
                left_lines.append((slope, intercept))
                left_weights.append((length))
            else:
                right_lines.append((slope, intercept))
                right_weights.append((length))

    # add more weight to longer lines    
    left_lane  = np.dot(left_weights,  left_lines) /np.sum(left_weights)  if len(left_weights) >0 else None
    right_lane = np.dot(right_weights, right_lines)/np.sum(right_weights) if len(right_weights)>0 else None

    return left_lane, right_lane # (slope, intercept), (slope, intercept)

def make_line_points(y1, y2, line):
    """
    Convert a line represented in slope and intercept into pixel points
    """
    if line is None:
        return None

    slope, intercept = line
    # make sure everything is integer as cv2.line requires it 
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    y1 = int(y1)
    y2 = int(y2)

    return ((x1, y1), (x2, y2))

def lane_lines(image, lines):
    left_lane, right_lane = average_slope_intercept(lines)
    y1 = image.shape[0] # bottom of the image
    y2 = y1*0.58         # slightly lower than the middle

    left_line  = make_line_points(y1, y2, left_lane)
    right_line = make_line_points(y1, y2, right_lane)

    return left_line, right_line

def draw_lane_lines(image, lines, color=[0, 255, 0], thickness=10):
    # make a separate image to draw lines and combine with the orignal later
    line_image = np.zeros_like(image)
    for line in lines:
        if line is not None:
                cv2.line(line_image, *line,  color, thickness)
    # image1 * α + image2 * β + λ
    # image1 and image2 must be the same shape.
    return cv2.addWeighted(image, 0.8, line_image, 1, 0.0)

def show_image(img, cmap=None):
    cols = 2
    rows = (len(img)+1)//cols
    
    plt.figure(figsize=(10, 11))
    for i, image in enumerate(img):
        plt.subplot(rows, cols, i+1)
        # use gray scale color map if there is only one channel
        cmap = 'gray' if len(image.shape)==2 else cmap
        plt.imshow(image, cmap=cmap)
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout(pad=0, h_pad=0, w_pad=0)
    plt.show()
    #cv2.waitKey(0)

def process_img(image):
    # region masking
    # parameter: image to be region masked for lane finding, vertices for region polygon
    ysize = image.shape[0]
    xsize = image.shape[1]
    bottom_left = [0.1*xsize, 0.95*ysize]
    top_left = [0.4*xsize, 0.6*ysize]
    bottom_right = [0.9*xsize, 0.95*ysize]
    top_right = [0.6*xsize, 0.6*ysize]

    vertices = np.array([[bottom_left, bottom_right, top_right, top_left]])

    # convert to grayscale
    # parameter: image to convert to grayscale
    gray = grayscale(image)
    cv2.imwrite("gray.jpg", gray)    

    # Define a kernel size and apply Gaussian smoothing
    # parameter: image to apply gaussian smoothing, kernel_size
    blur_gray = gaussian_blur(gray, 15)
    cv2.imwrite("gaussian.jpg", blur_gray)    

    # Define our parameters for Canny and apply
    # parameter: image to apply canny, low_threshold, high_threshold
    masked_edges = canny(blur_gray, 50, 150)
    cv2.imwrite("canny.jpg", masked_edges)    

    # region of interest image
    roi_image = region_of_interest(masked_edges, np.int32([vertices]))
    cv2.imwrite("roi.jpg", roi_image)    

    # Define the Hough transform parameters
    # Make a blank the same size as our image to draw on
    # parameter: image to be transformed in hough space, rho, theta, threshold, minimum line lenght, maximum line gap
    line_image, lines = hough_lines(roi_image, 1, np.pi/180, 20, 20, 300)
    cv2.imwrite("hough.jpg", line_image)    
    
    list_of_lines = lines
    lane_images = []
    for lines in zip(list_of_lines):
        lane_images.append(draw_lane_lines(image, lane_lines(image, list_of_lines)))

    return lane_images[0]


def process_video(video_input, video_output):
    clip = VideoFileClip(os.path.join('test_videos', video_input))
    processed = clip.fl_image(process_img)
    processed.write_videofile(os.path.join('test_videos_output', video_output), audio=False)

files = os.listdir("test_images/")
for file in files:
    print(file)
    image = mpimg.imread('test_images/' + file)
    lane_image = process_img(image)
    cv2.imwrite('test_images_output/' + file, lane_image)    

process_video('challenge.mp4', 'challenge.mp4')
videos = os.listdir("test_videos/")
for video in videos:
    print(video)
    process_video(video, video)
    
