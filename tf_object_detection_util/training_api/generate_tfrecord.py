"""
Usage:
  # From tensorflow/models/
  # Create train data:
  python generate_tfrecord.py --csv_input=data/train_labels.csv  --output_path=train.record

  python generate_tfrecord.py --csv_input=TFRConv/train.csv --output_path=TFRConv/train.record --image_dir=TFRConv/train/

  # Create test data:
  python generate_tfrecord.py --csv_input=data/test_labels.csv  --output_path=test.record

  python generate_tfrecord.py --csv_input=TFRConv/valid.csv --output_path=TFRConv/valid.record --image_dir=TFRConv/valid/

  
  
  python train.py --logtostderr --train_dir=/home/prasannals/Downloads/handsup/data --pipeline_config_path=/home/prasannals/Downloads/handsup/data/TFRConv/ssd_mobilenet_v1_pets.config

  python export_inference_graph.py --input_type image_tensor --pipeline_config_path /home/prasannals/Downloads/handsup/data/TFRConv/ssd_mobilenet_v1_pets.config --trained_checkpoint_prefix /home/prasannals/Downloads/handsup/data/model.ckpt-4733 --output_directory /home/prasannals/Downloads/handsup/data/TFRConv/inference_graph
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import io
import pandas as pd
import tensorflow as tf

from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict

def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path, class_text_to_int):
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example

def genTfr(outputPath, imageDir, csvInput, catDict):
    writer = tf.python_io.TFRecordWriter(outputPath)
    path = os.path.join(imageDir)
    examples = pd.read_csv(csvInput)
    grouped = split(examples, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group, path, lambda label : catDict.get(label, None) )
        writer.write(tf_example.SerializeToString())

    writer.close()
    print('Successfully created the TFRecords: {}'.format(outputPath))
