import numpy as np
import tensorflow as tf
import cv2
from collections import defaultdict
from io import StringIO
from PIL import Image

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import ops as utils_ops

import argparse

if tf.__version__ < '1.4.0':
  raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')


def parseArguments():
    # Create argument parser
    parser = argparse.ArgumentParser()
    # Optional arguments
    parser.add_argument("-d", "--device", help="video device to use, defaults to 0", type=int, default=0)
    parser.add_argument("-W", "--width", help="width of the camera shown, defaults to 480", type=int, default=480)
    parser.add_argument("-H", "--height", help="height of the camera shown, defaults to 360", type=int, default=360)
    parser.add_argument("-c", "--class_id", help="class to filter, defaults to 1 (person)", type=int, default=1)
    parser.add_argument("-s", "--score", help="score threshold to use, defaults to 0.5", type=float, default=0.5)
    # Parse arguments
    args = parser.parse_args()
    return args

def filter_class_as_lists(classes, scores, b_boxes, class_filter, score_filter):
    #import ipdb; ipdb.set_trace() 
    return_classes = []
    return_scores = []
    return_b_boxes = []
    for i in range(classes.shape[0]):
        if classes[i]==class_filter and scores[i]>score_filter:
            return_classes.append(classes[i])
            return_scores.append(scores[i])
            return_b_boxes.append(b_boxes[i])
    return(return_classes,return_scores,return_b_boxes)

def add_icon_position(original_image, icon_logo,roi_coords):
    
    rows,cols,channels = original_image.shape
    print("original image shape: ",  original_image.shape)
    print("icon logo shape: ", icon_logo.shape)
    print("input roi: ", roi_coords)

    ## Normalized coordinates
    (left, right, top, bottom) = (roi_coords[0] * rows, roi_coords[2] * rows, 
                                  roi_coords[1] * cols, roi_coords[3] * cols)

    (left, right, top, bottom) = tuple(map(int,(left, right, top, bottom)))
    print("converted coords from normalized: ", (left, right, top, bottom))

    new_width = right - left
    new_height = bottom - top
    
    print("w: ", new_width)
    print("h: ", new_height)
    
    icon_logo = cv2.resize(icon_logo, (new_height,new_width))
    print("icon logo size: ", icon_logo.shape)
    roi =  original_image[left:right,top:bottom]
    
    print("new roi shape: ", roi.shape)

    # Now create a mask of logo and create its inverse mask also
    # in this case, is already B/W
    mask = cv2.cvtColor(icon_logo,cv2.COLOR_BGR2GRAY)
    print("mask shape: ", mask.shape)
    mask_inv = cv2.bitwise_not(mask)
    print("inv mask: shape", mask_inv)

    # Now black-out the area of logo in ROI
    original_image_bg = cv2.bitwise_and(roi,roi,mask = mask)

    # Take only region of logo from logo image.
    icon_logo_fg = cv2.bitwise_and(icon_logo,icon_logo,mask = mask_inv)

    # Put logo in ROI and modify the main image
    dst = cv2.add(original_image_bg,icon_logo_fg)
    original_image[left:right,top:bottom] = dst
    
    return(original_image)


def run_detection_webcam(device,width,height,class_id,score):
    #TODO CHANGE PRINTS TO LOG, STUPID MAN
    last_cat_box=[]
    cap = cv2.VideoCapture(device)
    imstack = cv2.imread("cat_icon.png")
    # Running the tensorflow session
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
            ret = True
            while (ret):
                ret,image_np = cap.read()
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                # Each box represents a part of the image where a particular object was detected.
                boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                # Each score represent how level of confidence for each of the objects.
                # Score is shown on the result image, together with the class label.
                scores = detection_graph.get_tensor_by_name('detection_scores:0')
                classes = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                # Actual detection.
                (boxes, scores, classes, num_detections) = sess.run(
                    [boxes, scores, classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})
                classes,scores,boxes = filter_class_as_lists(np.squeeze(classes),
                                                            np.squeeze(scores),
                                                            np.squeeze(boxes),
                                                            class_id,
                                                            score)
                boxes = np.array(boxes)
                classes = np.array(classes).astype(np.int32)
                num_detections = len(classes)
                # Visualization of the results of a detection.
                vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    boxes,
                    classes,
                    scores,
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=8)
                # Found a cat
                if any(classes):
                    last_cat_box = boxes[0]
                # No cat, print the cat icon if the last known position of the cat exist
                elif any(last_cat_box):
                    img_cv = add_icon_position(image_np,imstack,last_cat_box)                    
                        
                img_cv = cv2.resize(image_np,(width,height))
    #            if not classes and frame_alarm_counter >= frame_counter:
    #                img_cv = cv2.addWeighted(img_cv,0.7,imstack,0.3,0)
    #            else:
    #                frame_counter=0
                cv2.imshow('image',img_cv)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    cap.release()
                    break

if __name__ == "__main__":

    args = parseArguments()
    print(args)
    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    PATH_TO_CKPT = 'model/frozen_inference_graph.pb'
    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS = 'data/mscoco_label_map.pbtxt'
    #number of classes
    NUM_CLASSES = 90

    #load graph
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    #load labels
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    run_detection_webcam(args.device,args.width,args.height,args.class_id,args.score)