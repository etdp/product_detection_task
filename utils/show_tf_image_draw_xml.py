#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 15:47:07 2019

@author: thomas_yang
"""

import hashlib
import io
import logging
import os
import re
import cv2
import random
from PIL import Image

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

path = '/home/thomas_yang/ML/TensorFlow/workspace/training_demo/annotations'

label_dict = {'04003906':'MM-Chocolate', 
              '748675116052':'Raisin',
              '4710043001433':'Mantuo',
              '4710095046208':'Fruit Tea',
              '4710105030326':'Carambola juice',
              '4710126035003':'Gummy Chocolate',
              '4710126041004':'Fruit candy',
              '4710126045460':'Lemon',
              '4710126100923':'Juice',
              '4710128020106':'Apple Soda',
              '4710174114095':'Straw Berry Pie',
              '4710298161234':'Jelly',
              '4710423051096':'Puni candy',
              '4710543006693':'Lays',
              '4710594924427':'cranberry juice',
              '4710626186519':'TEA',
              '4710757030200':'KAISER Chocolate',
              '4711162821520':'Corn log',
              '4711202224892':'LINLONGO',
              '4711402892921':'Hot spicy docan',
              '4713507024627':'Milk log',
              '4714431053110':'Straw Berry',
              '4719264904219':'Korean yellow',
              '4719264904233':'Korean purple',
              '4902777062013':'Grean Chocolate',
              '7610700600863':'Yellow',
              '8801111390064':'Red Chocolate PIE',
              '8886467102400':'Pink',
              '8888077101101':'Chocolate Log',
              '8888077102092':'Pink PANDA Ball'
        }


image_feature_description = {

    'image/height':tf.compat.v1.FixedLenFeature([], tf.int64),
    'image/width': tf.compat.v1.FixedLenFeature([], tf.int64),
    'image/filename':  tf.compat.v1.FixedLenFeature([], tf.string),
    'image/source_id': tf.compat.v1.FixedLenFeature([], tf.string),
    'image/encoded': tf.compat.v1.FixedLenFeature([], tf.string),
    'image/format': tf.compat.v1.FixedLenFeature([], tf.string),
    'image/object/bbox/xmin': tf.compat.v1.FixedLenFeature([], tf.float32),
    'image/object/bbox/xmax': tf.compat.v1.FixedLenFeature([], tf.float32),
    'image/object/bbox/ymin': tf.compat.v1.FixedLenFeature([], tf.float32),
    'image/object/bbox/ymax': tf.compat.v1.FixedLenFeature([], tf.float32),
    'image/object/class/text': tf.compat.v1.FixedLenFeature([], tf.string),
    'image/object/class/label': tf.compat.v1.FixedLenFeature([], tf.int64),
}  

def _parse_image_function(example_proto):  
      # Parse the input tf.Example proto using the dictionary above.
    return tf.compat.v1.parse_single_example(example_proto, image_feature_description)

raw_image_dataset = tf.data.TFRecordDataset(path + '/train.record')
parsed_image_dataset = raw_image_dataset.map(_parse_image_function)

def show_image(count):    
    # Create a dictionary describing the features.
#    print(parsed_image_dataset)
    
    c = 0 
    for index, image_features in enumerate(parsed_image_dataset):
        image_raw = image_features['image/encoded'].numpy()
        
        encoded_jpg_io = io.BytesIO(image_raw)
        image = Image.open(encoded_jpg_io)  
        pix = np.array(image)
        
        width = pix.shape[1]
        height = pix.shape[0]        
        xmin = int((image_features['image/object/bbox/xmin'].numpy()) * width)
        xmax = int((image_features['image/object/bbox/xmax'].numpy()) * width)
        ymin = int((image_features['image/object/bbox/ymin'].numpy()) * height)
        ymax = int((image_features['image/object/bbox/ymax'].numpy()) * height)
        text = (image_features['image/object/class/text'].numpy().decode("utf-8") )
        
        cv2.rectangle(pix, (xmin,ymin), (xmax,ymax), (255,0,0), 3)
        cv2.putText(pix, str(int((text))), (xmin, ymin), 1, 2, (0,0,255), 2)
        cv2.putText(pix, label_dict[(((text)))], (xmin, ymax + 20), 1, 2, (0,255,0), 2)
        cv2.putText(pix, 'Image Number: ' + str(c), (5, 20), 1, 2, (255,0,255), 2)
        cv2.imshow('TFrecord_Image', cv2.cvtColor(pix, cv2.COLOR_RGB2BGR))
#        print(xmin, xmax, ymin, ymax, np.int64(text))        
#        cv2.waitKey(0)

        c+=1
#        print(c)
        if c==count:
            break
        
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
        
show_image(1000)



"""
label_dict = {'04003906':'MM巧顆粒', 
              '748675116052':'葡萄乾',
              '4710043001433':'慢駝珠',
              '4710095046208':'冰鎮水果茶',
              '4710105030326':'楊桃汁',
              '4710126035003':'Gummy巧顆粒',
              '4710126041004':'知心水果軟糖',
              '4710126045460':'寧模薄荷',
              '4710126100923':'純果汁',
              '4710128020106':'蘋果西打',
              '4710174114095':'草莓法國蘇',
              '4710298161234':'百香果汁果凍',
              '4710423051096':'Puni軟糖',
              '4710543006693':'樂事洋芋片',
              '4710594924427':'蔓越莓果汁',
              '4710626186519':'分解茶',
              '4710757030200':'KAISER巧顆粒',
              '4711162821520':'玉米棒',
              '4711202224892':'LINLONGO',
              '4711402892921':'辣味豆乾',
              '4713507024627':'牛奶棒',
              '4714431053110':'草莓蒟蒻',
              '4719264904219':'韓國黃黃零食',
              '4719264904233':'韓國紫紫零食',
              '4902777062013':'綠綠抹茶巧顆粒',
              '7610700600863':'黃黃喉糖',
              '8801111390064':'紅色巧顆粒派',
              '8886467102400':'品客',
              '8888077101101':'紅色巧顆粒棒棒',
              '8888077102092':'粉紅PANDA球'
        }
"""
