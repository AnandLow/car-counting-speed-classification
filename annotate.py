import os
import numpy as np
import tensorflow as tf
import six.moves.urllib as urllib 
import tarfile 
from PIL import Image
from tqdm import tqdm
from time import gmtime, strftime
import json
import cv2

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

class DetectionObj(object):
    def __init__(self, model = "ssd_mobilenet_v1_coco_11_06_2017'):
        self.CURRENT_PATH = cs.getcwd()
        
        self.TARGET_PATH = self.CURRENT_PATH
        
        self.MODELS = {"ssd_mobilenet_v2_coco_11_06_2017")
        
        self.THRESHOLD = 0.25
        
        self.CKPT_FILE = os.path.join(self.CURRENT_PATH, 'object_detection', self.MODEL_NAME, 'frozen_inference_graph.pb')
        
        # Load detection model
        self.DETECTION_GRAPH = self.load_frozen_model()
        
        # Load the labels of the class recognized by the detection model
        
        self.NUM_CLASSES = 90
        path_to_labels = os.path.join(self.CURRENT_PATH, 'object_detection', 'data', 'mscoco_label_map.pbtxt')
        
        label_mapping = label_map_util.load_labelmap(path_to_labels)
        
        extracted_categories = label_map_util.convert_lbel_map_to_categories(label_mapping, max_num_classes = self.NUM_CLASSES, use_display_name=True)
        
        self.LABELS = {item['id']:item['name'] for item in extracted_categories}
        
        self.CATEGORY_INDEX = label_map_util.create_category_index(extracted_categories)
        
        self.TF_SESSION = tf.Session(graph=self.DETECTION_GRAPH)
        
        
        
        
        

