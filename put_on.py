import matplotlib.pyplot as plt
from scipy.misc import imrotate
from IPython import embed
import cv2  # OpenCV Library
# import the necessary packages
from imutils import face_utils
from collections import OrderedDict
import numpy as np
import argparse
import imutils
import dlib

# based on example from https://www.pyimagesearch.com/2017/04/10/detect-eyes-nose-lips-jaw-dlib-opencv-python/
#-----------------------------------------------------------------------------
#       Load and configure mustache (.png with alpha transparency)
#-----------------------------------------------------------------------------
 
# Load our overlay image: mustache.png
imgMustache = cv2.imread('mustache.png',-1)
 
# Create the mask for the mustache
orig_mask = imgMustache[:,:,3]
# Create the inverted mask for the mustache
orig_mask_inv = cv2.bitwise_not(orig_mask)
 
# Convert mustache image to BGR
# and save the original image size (used later when re-sizing the image)
imgMustache = imgMustache[:,:,0:3]
origMustacheHeight, origMustacheWidth = imgMustache.shape[:2]
 
#-----------------------------------------------------------------------------
#       Main program loop
#-----------------------------------------------------------------------------
def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
 
    # return a tuple of (x, y, w, h)
    return (x, y, w, h) 


def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)
 
    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
 
    # return the list of (x, y)-coordinates
    return coords



## initialize dlib's face detector (HOG-based) and then create
## the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
#predictor = dlib.shape_predictor(args["shape_predictor"])
## load the input image, resize it, and convert it to grayscale
#image = cv2.imread(args["image"])
 
## show the output image with the face detections + facial landmarks
#cv2.imshow("Output", image)
#cv2.waitKey(0)

FACIAL_LANDMARKS_IDXS = OrderedDict([
    ("mouth", (48, 68)),
    ("right_eyebrow", (17, 22)),
    ("left_eyebrow", (22, 27)),
    ("right_eye", (36, 42)),
    ("left_eye", (42, 48)),
    ("nose", (27, 35)),
    ("jaw", (0, 17))
])


colors = [(19, 199, 109), (79, 76, 240), (230, 159, 23),
            (168, 100, 168), (158, 163, 32),
            (163, 38, 32), (180, 42, 220)]

def add_mustache(frame):
    # Create greyscale image from the video feed
    
    image = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ih, iw, _ = image.shape
    
    # detect faces in the grayscale image
    rects = detector(gray, 1)
    #roi_color = frame #frame[y:y+h, x:x+w]
    print(len(rects), "RECTS") 
    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
     
        # convert dlib's rectangle to a OpenCV-style bounding box
        # [i.e., (x, y, w, h)], then draw the face bounding box
        (x, y, w, h) = face_utils.rect_to_bb(rect)
    
        #roi_gray = gray[y:y+h, x:x+w]
        #roi_color = frame[y:y+h, x:x+w]
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
     
        # show the face number
        cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
     
        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        for (x, y) in shape:
            cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
    
        #name = 'nose'
        #i,j = face_utils.FACIAL_LANDMARKS_IDXS[name]
        #for (i, name) in enumerate(FACIAL_LANDMARKS_IDXS.keys()):
        i = 0
        name = 'mouth'
        if 1:
            # grab the (x, y)-coordinates associated with the
            # face landmark
            (j, k) = FACIAL_LANDMARKS_IDXS[name]
            pts = shape[j:k]
    
            # check if are supposed to draw the jawline
            if name == "jaw":
                # since the jawline is a non-enclosed facial region,
                # just draw lines between the (x, y)-coordinates
                for l in range(1, len(pts)):
                    ptA = tuple(pts[l - 1])
                    ptB = tuple(pts[l])
                    #cv2.line(overlay, ptA, ptB, colors[i], 2)
                    cv2.line(image, ptA, ptB, colors[i], 2)
    
            # otherwise, compute the convex hull of the facial
            # landmark coordinates points and display it
            else:
                hull = cv2.convexHull(pts)
                #cv2.drawContours(overlay, [hull], -1, colors[i], -1) 
                #cv2.drawContours(image, [hull], -1, colors[i], -1) 
                if name == 'mouth':
                    nx = np.min(pts[:,0])
                    ny = np.min(pts[:,1])
                    nw = np.max(pts[:,0])-np.min(pts[:,0])+1
                    nh = np.max(pts[:,1])-np.min(pts[:,1])+1
              
                    #cv2.rectangle(image,(nx,ny),(nx+nw,ny+nh),(255,0,0),2)
                    cv2.rectangle(image,(nx,ny),(nx+nw,ny+nh),colors[i])
    
                    mustacheWidth =  15 * nw
                    mustacheHeight = mustacheWidth * origMustacheHeight / origMustacheWidth
    
                    # Center the mustache on the bottom of the nose
                    x1 = nx - int(mustacheWidth/4.) 
                    x2 = nx + nw + int(mustacheWidth/4.) 
                    y1 = ny + nh - int(mustacheHeight/4.) 
                    y2 = ny + nh + int((mustacheHeight/2.)) 
    
                    rx2,ry2 = pts[np.argmax(pts[:,0]),:]
                    rx1,ry1 = pts[np.argmin(pts[:,0]),:]
                    
                    opp = ry2-ry1
                    adj = rx2-rx1
                angle = -np.rad2deg(np.arctan(opp/float(adj)))
                
                # Check for clipping
                if x1 < 0:
                    x1 = 0
                if y1 < 0:
                    y1 = 0
                if x2 > iw:
                    x2 = iw
                if y2 > ih:
                    y2 = ih
 
                # Re-calculate the width and height of the mustache image
                mustacheWidth = x2 - x1
                mustacheHeight = y2 - y1
 
                # Re-size the original image and the masks to the mustache sizes
                # calcualted above
                mustache = cv2.resize(imgMustache, (mustacheWidth,mustacheHeight), interpolation = cv2.INTER_AREA)
                mustache_rot = imrotate(mustache, angle)
                orig_mask_rot = imrotate(orig_mask, angle)
                orig_mask_inv_rot = cv2.bitwise_not(orig_mask_rot)
                mask = cv2.resize(orig_mask_rot, (mustacheWidth,mustacheHeight), interpolation = cv2.INTER_AREA)
                mask_inv = cv2.resize(orig_mask_inv_rot, (mustacheWidth,mustacheHeight), interpolation = cv2.INTER_AREA)
 
                # take ROI for mustache from background equal to size of mustache image
                roi = image[y1:y2, x1:x2]
 
                ### roi_bg contains the original image only where the mustache is not
                ## in the region that is the size of the mustache.

                roi_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
                ## roi_fg contains the image of the mustache only where the mustache is
                roi_fg = cv2.bitwise_and(mustache,mustache,mask = mask)
 
                ## join the roi_bg and roi_fg
                dst = cv2.add(roi_bg,roi_fg)
 
                ## place the joined image, saved to dst back over the original image
                image[y1:y2, x1:x2] = dst
    return image
 


if __name__ == '__main__':
     

## collect video input from first webcam on system
    video_capture = cv2.VideoCapture(0)
     
    while True:
        # Capture video feed
        ret, frame = video_capture.read()
    
        image = add_mustache(frame)
     
                
          
        # Display the resulting frame
        cv2.imshow('Video', image)
     
        print("press any key to exit")
        # NOTE;  x86 systems may need to remove: " 0xFF == ord('q')"
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
# 
# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()

