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

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

def tf_open_image(path):
    image = Image.open(path)
    image_np = load_image_into_numpy_array(image)
    return image_np

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

def load_graph(path_to_frozen_graph):
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(path_to_frozen_graph, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
            
    return detection_graph

def run_inference_for_single_image(image, graph):
  with graph.as_default():
    with tf.Session() as sess:
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

def translate_detection_class(detection_classes, idx_to_label):
    return [idx_to_label[i]['name'] for i in detection_classes ]


def keep_detected_boxes(output_dict):
    num_detections = output_dict['num_detections']
    for k in ['detection_boxes', 'detection_scores', 'detection_classes', 'detection_classes_translated']:
        output_dict[k] = output_dict[k][:num_detections]

    return output_dict

class TFInference:
    def __init__(self, path_to_frozen_graph, path_to_pbtxt):
        # TODO make path_to_pbtxt optional. Return raw predictions if path_to_pbtxt is None
        self.path_to_frozen_graph, self.path_to_pbtxt = path_to_frozen_graph, path_to_pbtxt
        self.graph = load_graph(str(self.path_to_frozen_graph))
        # TODO - replace usage of util with your own implementation. Reduces dependency on library
        # and the user having to add the library to their PYTHON_PATH. Kinda optional though. Since we 
        # are anyway depending on the Tensorflow Object Detection API.
        self.idx_to_labels = label_map_util.create_category_index_from_labelmap(str(self.path_to_pbtxt), 
                                    use_display_name=True)
    
    def predict(self, img_path, visualize=False):
        image = tf_open_image(img_path)
        output_dict = run_inference_for_single_image(image, self.graph)
        output_dict['detection_classes_translated'] = translate_detection_class(
                    output_dict['detection_classes'], self.idx_to_labels)

        if visualize:
            vis_util.visualize_boxes_and_labels_on_image_array(
              image,
              output_dict['detection_boxes'],
              output_dict['detection_classes'],
              output_dict['detection_scores'],
              self.idx_to_labels,
              instance_masks=output_dict.get('detection_masks'),
              use_normalized_coordinates=True,
              line_thickness=8)
        
        output_dict = keep_detected_boxes(output_dict)

        return output_dict, image


