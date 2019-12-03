# import required packages
import cv2
import argparse
import numpy as np
import header_detection as header

# Reference Image
capture_reference = 'newFull.jpg'
header.downloadFile(capture_reference)

# Scale for both images
scale = 0.00392

# read class names from text file
classes = None
with open('coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# read pre-trained model and config file
net = cv2.dnn.readNet('fish2760.weights', 'yolov3-voc-version2.cfg')

# Call function for ref images
# Array contains nms bounding boxes with x y coordinates

# Get cropped images
cropped_Images = header.cropImages(capture_reference)
left_image = cropped_Images[0]
right_image = cropped_Images[1]

left_image_width = left_image.shape[1]

# create input blob for reference image 
left_Blob = cv2.dnn.blobFromImage(left_image, scale, (608,608), (0,0,0), True, crop=False)
right_Blob = cv2.dnn.blobFromImage(right_image, scale, (608,608), (0,0,0), True, crop=False)

# Cropped Image Scan
left_Image_Boxes = header.scan_Image(net, left_image, left_Blob)
right_Image_Boxes = header.scan_Image(net, right_image, right_Blob)

#Update co-ordinates of right image boxes
header.updateCoordinate(right_Image_Boxes, left_image_width)

# Combine lists
referenceImage_Boxes = left_Image_Boxes+right_Image_Boxes

# Create file and write boxes co-ordinates to text file
filename = "ref_File.txt"
ref_File = open(filename, "w+")

for box in referenceImage_Boxes:
    ref_File.writelines("%s " %value for value in box)
    ref_File.write("\n")

ref_File.close()