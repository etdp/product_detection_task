#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 19:03:32 2019

@author: thomas_yang
"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

"""
Usage:
# Create train data:
python generate_tfrecord.py --label=<LABEL> --csv_input=<PATH_TO_ANNOTATIONS_FOLDER>/train_labels.csv
--output_path=<PATH_TO_ANNOTATIONS_FOLDER>/train.record
# Create test data:
python generate_tfrecord.py --label=<LABEL> --csv_input=<PATH_TO_ANNOTATIONS_FOLDER>/test_labels.csv
--output_path=<PATH_TO_ANNOTATIONS_FOLDER>/test.record
"""

"""
runfile('/home/thomas_yang/ML/TensorFlow/scripts/preprocessing/generate_tfrecord.py', args='--label0=4003906 --label1=748675116052 --label2=4710043001433 --label3=4710095046208 --label4=4710105030326 --label5=4710126035003 --label6=4710126041004 --label7=4710126045460 --label8=4710126100923 --label9=4710128020106 --label10=4710174114095 --label11=4710298161234 --label12=4710423051096 --label13=4710543006693 --label14=4710594924427 --label15=4710626186519 --label16=4710757030200 --label17=4711162821520 --label18=4711202224892 --label19=4711402892921 --label20=4713507024627 --label21=4714431053110 --label22=4719264904219 --label23=4719264904233 --label24=4902777062013 --label25=7610700600863 --label26=8801111390064 --label27=8886467102400 --label28=8888077101101 --label29=8888077102092 --csv_input=/home/thomas_yang/ML/TensorFlow/workspace/training_demo/annotations/train_labels.csv --img_path=/home/thomas_yang/ML/TensorFlow/workspace/training_demo/images/train --output_path=/home/thomas_yang/ML/TensorFlow/workspace/training_demo/annotations/train.record', wdir='/home/thomas_yang/ML/TensorFlow/scripts/preprocessing')
runfile('/home/thomas_yang/ML/TensorFlow/scripts/preprocessing/generate_tfrecord.py', args='--label0=lefthand --label1=righthand --csv_input=/home/thomas_yang/ML/TensorFlow/workspace/training_demo/annotations/test_labels.csv --img_path=/home/thomas_yang/ML/TensorFlow/workspace/training_demo/images/test --output_path=/home/thomas_yang/ML/TensorFlow/workspace/training_demo/annotations/test.record', wdir='/home/thomas_yang/ML/TensorFlow/scripts/preprocessing')
"""


import os
import io
import pandas as pd
import tensorflow as tf
import sys
sys.path.append("../../models/research")
from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict

flags = tf.app.flags
flags.DEFINE_string('csv_input', '', 'Path to the CSV input')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
#flags.DEFINE_string('label', '', 'Name of class label')
# if your image has more labels input them as
flags.DEFINE_string('label0', '', 'Name of class[0] label')
flags.DEFINE_string('label1', '', 'Name of class[1] label')
flags.DEFINE_string('label2', '', 'Name of class[1] label')
flags.DEFINE_string('label3', '', 'Name of class[1] label')
flags.DEFINE_string('label4', '', 'Name of class[1] label')
flags.DEFINE_string('label5', '', 'Name of class[1] label')
flags.DEFINE_string('label6', '', 'Name of class[1] label')
flags.DEFINE_string('label7', '', 'Name of class[1] label')
flags.DEFINE_string('label8', '', 'Name of class[1] label')
flags.DEFINE_string('label9', '', 'Name of class[1] label')
flags.DEFINE_string('label10', '', 'Name of class[1] label')
flags.DEFINE_string('label11', '', 'Name of class[1] label')
flags.DEFINE_string('label12', '', 'Name of class[1] label')
flags.DEFINE_string('label13', '', 'Name of class[1] label')
flags.DEFINE_string('label14', '', 'Name of class[1] label')
flags.DEFINE_string('label15', '', 'Name of class[1] label')
flags.DEFINE_string('label16', '', 'Name of class[1] label')
flags.DEFINE_string('label17', '', 'Name of class[1] label')
flags.DEFINE_string('label18', '', 'Name of class[1] label')
flags.DEFINE_string('label19', '', 'Name of class[1] label')
flags.DEFINE_string('label20', '', 'Name of class[1] label')
flags.DEFINE_string('label21', '', 'Name of class[1] label')
flags.DEFINE_string('label22', '', 'Name of class[1] label')
flags.DEFINE_string('label23', '', 'Name of class[1] label')
flags.DEFINE_string('label24', '', 'Name of class[1] label')
flags.DEFINE_string('label25', '', 'Name of class[1] label')
flags.DEFINE_string('label26', '', 'Name of class[1] label')
flags.DEFINE_string('label27', '', 'Name of class[1] label')
flags.DEFINE_string('label28', '', 'Name of class[1] label')
flags.DEFINE_string('label29', '', 'Name of class[1] label')
# and so on.
flags.DEFINE_string('img_path', '', 'Path to images')
FLAGS = flags.FLAGS

def class_text_to_int(row_label):
#    if row_label == FLAGS.label: # 'ship':
#        return 1
    #comment upper if statement and uncomment these statements for multiple labelling
    if row_label == FLAGS.label0:
        return 1
    elif row_label == FLAGS.label1:
        return 2
    elif row_label == FLAGS.label2:
        return 3
    elif row_label == FLAGS.label3:
        return 4
    elif row_label == FLAGS.label4:
        return 5
    elif row_label == FLAGS.label5:
        return 6
    elif row_label == FLAGS.label6:
        return 7
    elif row_label == FLAGS.label7:
        return 8
    elif row_label == FLAGS.label8:
        return 9
    elif row_label == FLAGS.label9:
        return 10
    elif row_label == FLAGS.label10:
        return 11
    elif row_label == FLAGS.label11:
        return 12
    elif row_label == FLAGS.label12:
        return 13
    elif row_label == FLAGS.label13:
        return 14
    elif row_label == FLAGS.label14:
        return 15
    elif row_label == FLAGS.label15:
        return 16
    elif row_label == FLAGS.label16:
        return 17
    elif row_label == FLAGS.label17:
        return 18
    elif row_label == FLAGS.label18:
        return 19
    elif row_label == FLAGS.label19:
        return 20
    elif row_label == FLAGS.label20:
        return 21
    elif row_label == FLAGS.label21:
        return 22
    elif row_label == FLAGS.label22:
        return 23
    elif row_label == FLAGS.label23:
        return 24
    elif row_label == FLAGS.label24:
        return 25
    elif row_label == FLAGS.label25:
        return 26
    elif row_label == FLAGS.label26:
        return 27
    elif row_label == FLAGS.label27:
        return 28
    elif row_label == FLAGS.label28:
        return 29   
    elif row_label == FLAGS.label29:
        return 30
    else:
        None

def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]

def create_tf_example(group, path):
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
        encoded_jpg_io = io.BytesIO(encoded_jpg)
        image = Image.open(encoded_jpg_io)
        width, height = image.size
        filename = group.filename.encode('utf8')
        image_format = b'jpg'
        # check if the image format is matching with your images.
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
#            print(row['class'])
#            classes_text.append(row['class'].encode('utf8'))
            classes_text.append(str(row['class']).encode('utf8'))
#            print(type(FLAGS.label0))
#            print(type(row['class']))
#            print(FLAGS.label0 == row['class'])
            classes.append(class_text_to_int(str(row['class'])))
#            print(classes)
#            classes.append(class_text_to_int(str(row['class'])))
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
        'image/object/class/label': dataset_util.int64_list_feature(classes),}))
    return tf_example

def main(_):
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
    path = os.path.join(os.getcwd(), FLAGS.img_path)
    examples = pd.read_csv(FLAGS.csv_input)
    grouped = split(examples, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group, path)
        writer.write(tf_example.SerializeToString())
        
    writer.close()
    output_path = os.path.join(os.getcwd(), FLAGS.output_path)
    print('Successfully created the TFRecords: {}'.format(output_path))
    
if __name__ == '__main__':
    tf.app.run()





