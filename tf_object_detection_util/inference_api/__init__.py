import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from pathlib import Path

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

def tf_open_image(path:'str or pathlib.Path - path to the image to be opened') -> 'np.ndarray':
    '''
    reads the image in "path" as an RGB image and returns the image as a numpy ndarray of shape (height, width, 3)
    '''
    image = Image.open(path)
    if image.mode != 'RGB':
        image = image.convert("RGB")
    image_np = load_image_into_numpy_array(image)
    return image_np

def load_image_into_numpy_array(image):
    '''
    converts a PIL Image (return provided by Image.open()) into a numpy array of 
    8 bit unsigned ints of size (height, width, 3)
    '''
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

def load_graph(path_to_frozen_graph):
    '''
    loads tensorflow frozen inference graph in the passed in path and returns the graph
    '''
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(path_to_frozen_graph, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
            
    return detection_graph

def run_inference_for_single_image(image:'np.ndarray of shape (height, width, 3)', 
                                graph:'TensorFlow Graph', sess:'tensorflow Session object'
                                ) -> 'dict - dictionary of outputs':
  with graph.as_default():
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: np.expand_dims(image, 0)})

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.int64)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict

def translate_detection_class(detection_classes:'list(int)', idx_to_label:'dict(int -> str)'
                ) -> 'list(str) - list of class names for the list of class indicies passed in':
    return [idx_to_label[i]['name'] for i in detection_classes ]


def keep_detected_boxes(output_dict):
    '''
    Sometimes there are more entries in output fields than there are number of detections.
    This function removes the extra outputs.

    WARNING - performs change inplace
    '''
    num_detections = output_dict['num_detections']
    for k in ['detection_boxes', 'detection_scores', 'detection_classes', 'detection_classes_translated']:
        output_dict[k] = output_dict[k][:num_detections]

    return output_dict

def np_if_not(arr):
    '''
    converts arr to a numpy array if it isn't already one. 
    If arr is already a numpy array, returns the same object.
    '''
    if arr is None:
        return None
    return arr if type(arr) == np.ndarray else np.array(arr)

class TFInference:
    def __init__(self, path_to_frozen_graph:str, path_to_pbtxt:str):
        '''
        path_to_frozen_graph - path to the frozen inference graph generated by the tensorflow object detection api.
        path_to_pbtxt - path to the .pxtxt file. this file contains the mapping of object names to unique IDs.
        '''
        # TODO make path_to_pbtxt optional. Return raw predictions if path_to_pbtxt is None
        self.path_to_frozen_graph, self.path_to_pbtxt = path_to_frozen_graph, path_to_pbtxt
        self.graph = load_graph(str(self.path_to_frozen_graph))
        # TODO - replace usage of util with your own implementation. Reduces dependency on library
        # and the user having to add the library to their PYTHON_PATH. Kinda optional though. Since we 
        # are anyway depending on the Tensorflow Object Detection API.
        self.idx_to_labels = label_map_util.create_category_index_from_labelmap(str(self.path_to_pbtxt), 
                                    use_display_name=True)
        self.sess = tf.Session(graph=self.graph)

    def visualize_pred(self, pred:'the prediction obtained from "predict" method',
        img:'numpy array (image) or tuple of (height, width). If tuple, black image of specified height and width is generated.'=None) -> np.ndarray:
        '''
        Visualizes the bounding boxes and object names of the prediction on the passed in image and returns the visualization as a numpy array
            of size (height, width, 3)
        '''
        if img is None:
            img = np.zeros((pred['img_height'], pred['img_width'], 3), dtype=np.uint8)
        else:
            img = img.copy()

        vis_util.visualize_boxes_and_labels_on_image_array(
              img,
              np_if_not(pred['detection_boxes']) ,
              np_if_not(pred['detection_classes']) ,
              np_if_not(pred['detection_scores']) ,
              self.idx_to_labels,
              instance_masks=np_if_not(pred.get('detection_masks')) ,
              use_normalized_coordinates=True,
              line_thickness=8)
        
        return img
    
    def predict(self, image, visualize=False):
        '''
        image : string, pathlib.Path or np.ndarray - If string or Path, image is read from the specified location
            if np.ndarray, the image is expected to be an RGB image of shape (height, width, 3). The values should be in the range 0-255
            dtype should be np.uint8
        visualize : boolean - if True, visualizes the bounding box and returns the image as a numpy array

        returns 
        output_dict : dict - results obtained from the model
        image : np.ndarray - a copy of the image passed in along with the visualization of the bounding boxes on the image
        '''
        if (type(image) == str) or (type(image) == Path):
            image = tf_open_image(image)
        elif visualize == True:
            # copy so as to not overwrite the original image when visualizing
            image = np.copy(image)
        
        output_dict = run_inference_for_single_image(image, self.graph, self.sess)
        output_dict['detection_classes_translated'] = translate_detection_class(
                    output_dict['detection_classes'], self.idx_to_labels)
        output_dict['img_height'] = image.shape[0]
        output_dict['img_width'] = image.shape[1]

        if visualize:
            image = self.visualize_pred(output_dict, image)
        
        output_dict = keep_detected_boxes(output_dict)

        return output_dict, image

    def close(self):
        '''
        Releases the resources allocated on the GPU/main memory. Always call this once you're done using the TFInference  object.
        '''
        self.sess.close()


