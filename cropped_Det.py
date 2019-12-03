# import required packages
import cv2
import argparse
import numpy as np
import header_detection as header

# image to be displayed
capture = 'newEmpty_V2.jpg'
referenceFile = 'ref_File.txt'
header.downloadFile(capture)

#Output image
output_image = cv2.imread(capture)
#Crop sky because mehmet doesnt like it
output_image = output_image[414:output_image.shape[0],0:output_image.shape[1]]

# Screen resolution for scaling cv2 display window
screen_res = 1920, 1080

# Scale for both images
scale = 0.00392

# read class names from text file
classes = None
with open('coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# read pre-trained model and config file
net = cv2.dnn.readNet('fish2760.weights', 'yolov3-voc-version2.cfg')

# Call function for ref and input images
#Crop images into two parts
cropped_images = header.cropImages(capture)
left_image = cropped_images[0]
right_image = cropped_images[1]

#Blob for left and right images
leftImage_Blob = cv2.dnn.blobFromImage(left_image, scale, (608,608), (0,0,0), True, crop=False)
rightImage_Blob = cv2.dnn.blobFromImage(right_image, scale, (608,608), (0,0,0), True, crop=False)

# Call function to get bounding boxes of detected cars
left_image_boxes = header.scan_Image(net, left_image, leftImage_Blob)
right_image_boxes = header.scan_Image(net, right_image, rightImage_Blob)

#Update co-ordinates of right image boxes to scale to whole image
header.updateCoordinate(right_image_boxes,left_image.shape[1])
# Free parking space tracking variable
freeSpots = 0

# Concatenate the two boxes
InputImage_Boxes = left_image_boxes+right_image_boxes

# Get co-ordinates from refFile.txt of reference image detections
referenceImage_Boxes=[]

with open(referenceFile,'r') as filehandler:
    filecontents = filehandler.readlines()

    for line in filecontents:
        # Ignore new line character at end
        curr_line = line[:-1]
        # Parse string by its spaces
        box = header.parse_line(curr_line)
        referenceImage_Boxes.append(box)

# Iterate over the reference image boxes array and compute iou for each box detected in the input image
# If iou is less than (some) threshold at given location, draw green box on that location
# indicating that location has no car (i.e: is free for parking)
for box in referenceImage_Boxes:
    # Call function to calculate iou for each in inputBox array
    iou_list = header.compute_overlap(box, InputImage_Boxes)
   
    if not iou_list:
        # If list is empty, slot at curr_box co-ords in input image is empty -> draw green box
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
    
        header.draw_bounding_box(output_image, round(x), round(y), round(x+w), round(y+h), 1)
        freeSpots+=1
       

print("Number of Free Parking Spots: ", freeSpots)
# Upload to dynamoDb
#header.addToDynamo(freeSpots)

# display output image 
# Set window size for displaying image
scale_width = screen_res[0] / output_image.shape[1]
scale_height = screen_res[1] / output_image.shape[0]
window_Scale = min(scale_width, scale_height)
window_width = int(output_image.shape[1] * window_Scale)
window_Height = int(output_image.shape[0] * window_Scale)
cv2.namedWindow('object detection', cv2.WINDOW_NORMAL)
cv2.resizeWindow('object detection', window_width, window_Height)   

cv2.imshow("object detection", output_image)
cv2.imwrite("object detection.jpg", output_image)
# wait until any key is pressed
cv2.waitKey()

# release resources
cv2.destroyAllWindows()
