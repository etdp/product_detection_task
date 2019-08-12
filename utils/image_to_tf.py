#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 11:36:34 2019
test test 
@author: thomas_yang
"""

# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""Convert raw PASCAL dataset to TFRecord for object_detection.

Example usage:
    python object_detection/dataset_tools/create_pascal_tf_record.py \
        --data_dir=/home/user/VOCdevkit \
        --year=VOC2012 \
        --output_path=/home/user/pascal.record
"""
#from __future__ import absolute_import
#from __future__ import division
#from __future__ import print_function

import hashlib
import io
import logging
import os
import re
import cv2
import random

from lxml import etree
import PIL.Image
import tensorflow as tf
from os.path import basename
import numpy as np
from xml.dom import minidom
import IPython.display as display

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

tf.enable_eager_execution()

path = '/home/thomas_yang/Downloads/ai_competition_unzip_data/'

label_dict = {'04003906':0, 
              '748675116052':1,
              '4710043001433':2,
              '4710095046208':3,
              '4710105030326':4,
              '4710126035003':5,
              '4710126041004':6,
              '4710126045460':7,
              '4710126100923':8,
              '4710128020106':9,
              '4710174114095':10,
              '4710298161234':11,
              '4710423051096':12,
              '4710543006693':13,
              '4710594924427':14,
              '4710626186519':15,
              '4710757030200':16,
              '4711162821520':17,
              '4711202224892':18,
              '4711402892921':19,
              '4713507024627':20,
              '4714431053110':21,
              '4719264904219':22,
              '4719264904233':23,
              '4902777062013':24,
              '7610700600863':25,
              '8801111390064':26,
              '8886467102400':27,
              '8888077101101':28,
              '8888077102092':29
        }

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def image_example(image_string, filepath):

#    lot_info = filepath.split('/')[-1].split('_')
    labels = label_dict[filepath.split('/')[-4]]
#    print(labels)
    
    filename, file_extension = os.path.splitext(filepath)
    if file_extension == '.jpg':
        xmlFilepath = filepath.split('.jpg')[0] + '.xml'
        image_format = b'jpg'
    elif file_extension == '.png':    
        xmlFilepath = filepath.split('.png')[0] + '.xml'
        image_format = b'png'
    img = cv2.imread(filepath)

    imgShape = img.shape
    height = imgShape[0]
    width = imgShape[1]

    labelXML = minidom.parse(xmlFilepath)
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []
#    countLabels = 0

    classes.append(labels)
    tmpArrays = labelXML.getElementsByTagName("filename")
    for elem in tmpArrays:
        filenameImage = elem.firstChild.data

    tmpArrays = labelXML.getElementsByTagName("name")
    for elem in tmpArrays:
        classes_text.append(str(elem.firstChild.data).encode('utf8'))

    tmpArrays = labelXML.getElementsByTagName("xmin")
    for elem in tmpArrays:
        xmins.append(int(elem.firstChild.data) / width)

    tmpArrays = labelXML.getElementsByTagName("xmax")
    for elem in tmpArrays:
        xmaxs.append(int(elem.firstChild.data) / width)

    tmpArrays = labelXML.getElementsByTagName("ymin")
    for elem in tmpArrays:
        ymins.append(int(elem.firstChild.data) / height)

    tmpArrays = labelXML.getElementsByTagName("ymax")
    for elem in tmpArrays:
        ymaxs.append(int(elem.firstChild.data) / height)

    encoded_jpg = image_string        
    tf_example = tf.train.Example(features=tf.train.Features(feature={
    'image/height': dataset_util.int64_feature(height),
    'image/width': dataset_util.int64_feature(width),
    'image/filename':  dataset_util.bytes_feature(filenameImage.encode('utf8')),
    'image/source_id': dataset_util.bytes_feature(filenameImage.encode('utf8')),
    'image/encoded': dataset_util.bytes_feature(encoded_jpg),
    'image/format': dataset_util.bytes_feature(image_format),
    'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
    'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
    'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
    'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
    'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
    'image/object/class/label': dataset_util.int64_list_feature(classes)}))
    return tf_example        

def go_pbtxt(): 
    filename = path + '/object_detection.pbtxt'
    if not os.path.exists(filename):
        print("-----------make object_detection.pbtxt -----------")
        print("writeing to {}".format(filename))

        for dirPath, dirNames, fileNames in os.walk(path):
            class_labels = dirNames
            break
        class_values = list(range(len(class_labels)))
        classdict = dict(zip(class_labels,class_values))
        classdict = label_dict
        print(classdict)
        inv_classList = {v: k for k, v in classdict.items()}
        print(inv_classList)

        with open(filename, 'a') as the_file:
            for i in range(1, len(classdict)+1):
                the_file.write("item {" + '\n')
                the_file.write("  id: " + str(i) + '\n')
                the_file.write("  name: '" + inv_classList[i-1] + "'" + '\n')
                the_file.write("}" + '\n\n')

        the_file.close()
    else:
        print('pbtxt already exist')

image_feature_description = {

    'image/height': tf.compat.v1.FixedLenFeature([], tf.int64),
    'width': tf.compat.v1.FixedLenFeature([], tf.int64),
    'labelXmin': tf.compat.v1.FixedLenFeature([], tf.int64),
    'labelYmin': tf.compat.v1.FixedLenFeature([], tf.int64),
    'labelXmax': tf.compat.v1.FixedLenFeature([], tf.int64),
    'labelYmax':tf.compat.v1.FixedLenFeature([], tf.int64),
    'label': tf.compat.v1.FixedLenFeature([], tf.string),
    'image_raw': tf.compat.v1.FixedLenFeature([], tf.string),
}  

def _parse_image_function(example_proto):  
      # Parse the input tf.Example proto using the dictionary above.
    return tf.compat.v1.parse_single_example(example_proto, image_feature_description)

raw_image_dataset = tf.data.TFRecordDataset(path + 'train_images.record')
parsed_image_dataset = raw_image_dataset.map(_parse_image_function)

def show_image(count):    
    # Create a dictionary describing the features.
    print(parsed_image_dataset)
    c = 0 
    for image_features in parsed_image_dataset:
        image_raw = image_features['image_raw'].numpy()
        display.display(display.Image(data=image_raw))
        c+=1
        if c==count:
            break

paths = os.walk(path)
folders = []

for folder,_,_ in paths:
    folders.append(folder)
folders = folders[1:]

walk_folders = []
for i in folders:
    try:
        aa = re.search( path + r'.*/Camera[0-9]/[0-9]', i)
        aa = aa.group()
        walk_folders.append(aa)
    except:
        pass

for i in walk_folders:
    for file in os.listdir(i):
        filename, file_extension = os.path.splitext(file)
        file_extension = file_extension.lower()
        if(file_extension == ".jpg" or file_extension==".jpeg" or file_extension==".png" or file_extension==".bmp"):
            aa = re.search(path + r'.*/Camera[0-9]/[0-9]/.*', i + '/' + file)
            aa = aa.group()
            new_name_list = (aa.split('/')[-4:])
            new_name = new_name_list[0] + '_' + new_name_list[1] + '_' + new_name_list[2] + '_' + new_name_list[3].split('.png')[0]
            imgFile = basename(filename) + file_extension
#            if file_extension == ".jpg":
#                print(imgFile, imgFile.split('_'))
#            if not imgFile.split('_'):
#                os.rename(i + '/' + imgFile, i + '/' + new_name + '.png')
#            xmlFile = basename(filename) + ".xml"
#            if not xmlFile.split('_'):
#                os.rename(i + '/' + xmlFile, i + '/' + new_name + '.xml')

        # 寫檔：
train =[]
val = []
test = []
for i in walk_folders:
    for file in os.listdir(i):
        filename, file_extension = os.path.splitext(file)
        file_extension = file_extension.lower()
        if(file_extension == ".jpg" or file_extension==".jpeg" or file_extension==".png" or file_extension==".bmp"):
            filepath = i + '/' + file
            train_val_test = np.random.choice(['train', 'val', 'test'], p=[0.8, 0.1, 0.1])
            if train_val_test == 'train':
                train.append(filepath)

            elif  train_val_test == 'val':  
                val.append(filepath)

            elif  train_val_test == 'test':     
                 test.append(filepath)            

random.shuffle(train)
random.shuffle(val)
random.shuffle(test)

with tf.compat.v1.python_io.TFRecordWriter(path + 'train_images.record') as writer:
    for filepath in train:
        image_string = open(filepath, 'rb').read()
        tf_example = image_example(image_string, filepath)
        writer.write(tf_example.SerializeToString())

with tf.compat.v1.python_io.TFRecordWriter(path + 'val_images.record') as writer:
    for filepath in val:
        image_string = open(filepath, 'rb').read()
        tf_example = image_example(image_string, filepath)
        writer.write(tf_example.SerializeToString())

with tf.compat.v1.python_io.TFRecordWriter(path + 'test_images.record') as writer:
    for filepath in test:
        image_string = open(filepath, 'rb').read()
        tf_example = image_example(image_string, filepath)
        writer.write(tf_example.SerializeToString())



