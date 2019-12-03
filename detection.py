# import required packages
import cv2
import argparse
import numpy as np

# read reference image
ref_image = cv2.imread('newFull.jpg')
# read input image
input_image = cv2.imread('newEmpty_V2.jpg')
#Output image
output_image = cv2.imread('newEmpty_V2.jpg')

# Screen resolution
screen_res = 1920, 1080

# Scale for both images
scale = 0.00392

# read class names from text file
classes = None
with open('coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# read pre-trained model and config file
net = cv2.dnn.readNet('fish2760.weights', 'yolov3-voc-version2.cfg')
#net = cv2.dnn.readNet('yolov3-voc.weights', 'yolov3-voc.cfg')

# create input blob for reference image 
ref_Blob = cv2.dnn.blobFromImage(ref_image, scale, (608,608), (0,0,0), True, crop=False)
# create input blob for input image
inputImage_Blob = cv2.dnn.blobFromImage(input_image, scale, (608,608), (0,0,0), True, crop=False)

def scan_Image(net, image, blob):
    # set input blob for the network
    net.setInput(blob)

    # run inference through the network
    # and gather predictions from output layers
    outs = net.forward(get_output_layers(net))

    # initialization
    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.2
    nms_threshold = 0.4

    # for each detection from each output layer 
    # get the confidence, class id, bounding box params
    # and ignore weak detections (confidence < 0.5)
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.1:
                center_x = int(detection[0] * image.shape[1])
                center_y = int(detection[1] * image.shape[0])
                w = int(detection[2] * image.shape[1])
                h = int(detection[3] * image.shape[0])
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    # apply non-max suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    bounding_boxes=[]
    # go through the detections remaining
    # after nms 

    for i in indices:
        i = i[0]
        box = boxes[i]
        bounding_boxes.append(box)
       
    return bounding_boxes

# end of scanImage function

# function to get the output layer names 
# in the architecture
def get_output_layers(net):
    
    layer_names = net.getLayerNames()
    
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers

# function to draw bounding box on the detected object with class name
def draw_bounding_box(img, x, y, x_plus_w, y_plus_h, RG_indicator):
    if RG_indicator == 1:
        cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h),(0, 255, 0), 3)
    else:
        cv2.rectangle(img,(x,y), (x_plus_w,y_plus_h),(0, 0, 255), 1)


def bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle

    # boxA coordinatess
    boxA_x = boxA[0]
    boxA_y = boxA[1]
    boxA_w = boxA[2]
    boxA_h = boxA[3] 

    # boxB coordinates
    boxB_x = boxB[0]
    boxB_y = boxB[1]
    boxB_w = boxB[2]
    boxB_h = boxB[3]
    
    xA = max(boxA_x, boxB_x)
    yA = max(boxA_y, boxB_y)
    xB = min(boxA_x+boxA_w, boxB_x+boxB_w)
    yB = min(boxA_y+boxA_h, boxB_y+boxB_h)

    #Computer the area of intersection of rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
 
	# compute the area of both the prediction and ground-truth
	# rectangles
    boxAArea = (boxA_w + 1) * (boxA_h + 1)
    boxBArea = (boxB_w + 1) * (boxB_h + 1)
 
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
 
	# return the intersection over union value
    return iou
# end of IOU calculating function

# Find the intersection over union
def compute_overlap(box, InputImage_Boxes):
    iou_list_for_box=[]
    
    for index in InputImage_Boxes:
        curr_Box = index
        iou = bb_intersection_over_union(box, curr_Box)
        iou = abs(iou)
        if iou > 0.5:
            iou_list_for_box.append(iou)

    return iou_list_for_box


# Call function for ref and input images
# Array contains nms bounding boxes with x y coordinates
referenceImage_Boxes = scan_Image(net, ref_image, ref_Blob)
#print("Reference Boxes")
#print(referenceImage_Boxes)
InputImage_Boxes = scan_Image(net, input_image, inputImage_Blob)
# Free parking space tracking variable
freeSpots = 0

# Iterate over the inputImage_Boxes array and compute iou for each box in reference Image
# If iou is less than (some) threshold, draw green box around that location
for index in referenceImage_Boxes:
    box = index
    # Call function to calculate iou for each in inputBox array
    iou_list=compute_overlap(box, InputImage_Boxes)
   
    if not iou_list:
        # If list is empty, slot at curr_box coords in input image is empty -> draw green box
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
    
        draw_bounding_box(output_image, round(x), round(y), round(x+w), round(y+h), 1)
        freeSpots+=1
       

print("Number of Free Parking Spots: ", freeSpots)

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
