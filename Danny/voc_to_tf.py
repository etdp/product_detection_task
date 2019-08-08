from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

import tensorflow as tf
tf.__version__

import cv2
import numpy as np
import IPython.display as display
from xml.dom import minidom
import os
import re
from os.path import basename


class To_Tfrecords:
    def __init__(self, dataset_folder_name):
        savePath = os.getcwd() + '/' + dataset_folder_name + '/'
        self.path = savePath
    
    def _bytes_feature(self,value):
      """Returns a bytes_list from a string / byte."""
      return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _float_feature(self,value):
      """Returns a float_list from a float / double."""
      return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    def _int64_feature(self,value):
      """Returns an int64_list from a bool / enum / int / uint."""
      return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


    def image_example(self,image_string, filepath):

        lot_info = filepath.split('/')[-1].split('_')
        labels = lot_info[0]

        xmlFilepath = filepath.split('.png')[0] + '.xml'
        img = cv2.imread(filepath)

        size_ratio_w = 1
        size_ratio_h = 1

        imgShape = img.shape
        img_h = imgShape[0]
        img_w = imgShape[1]


        labelXML = minidom.parse(xmlFilepath)
        labelName = []
        labelXmin = []
        labelYmin = []
        labelXmax = []
        labelYmax = []
        countLabels = 0

        tmpArrays = labelXML.getElementsByTagName("filename")
        for elem in tmpArrays:
            filenameImage = elem.firstChild.data

        tmpArrays = labelXML.getElementsByTagName("name")
        for elem in tmpArrays:
            labelName.append(str(elem.firstChild.data))

        tmpArrays = labelXML.getElementsByTagName("xmin")
        for elem in tmpArrays:
            labelXmin.append(int(int(elem.firstChild.data) * size_ratio_w))

        tmpArrays = labelXML.getElementsByTagName("ymin")
        for elem in tmpArrays:
            labelYmin.append(int(int(elem.firstChild.data) * size_ratio_h))

        tmpArrays = labelXML.getElementsByTagName("xmax")
        for elem in tmpArrays:
            labelXmax.append(int(int(elem.firstChild.data) * size_ratio_w))

        tmpArrays = labelXML.getElementsByTagName("ymax")
        for elem in tmpArrays:
            labelYmax.append(int(int(elem.firstChild.data) * size_ratio_h))

        feature = {
          'height': self._int64_feature(img_h),
          'width': self._int64_feature(img_w),
          'labelXmin': self._int64_feature(labelXmin[0]),
          'labelYmin': self._int64_feature(labelYmin[0]),
          'labelXmax': self._int64_feature(labelXmax[0]),
          'labelYmax': self._int64_feature(labelYmax[0]),
          'label': self._bytes_feature(labels.encode('utf-8')),
          'image_raw': self._bytes_feature(image_string), # 紀錄位置
        }
        return tf.train.Example(features=tf.train.Features(feature=feature))

    
    def _parse_image_function(self,example_proto):
          # Parse the input tf.Example proto using the dictionary above.
        return tf.compat.v1.parse_single_example(example_proto, image_feature_description)

    
    def go(self):
        # rename:
        paths = os.walk(self.path)
        folders = []

        for folder,_,_ in paths:
            folders.append(folder)
        folders = folders[1:]
        walk_folders = []
        for i in folders:
            try:
                aa = re.search( self.path + r'.*/Camera[0-9]/[0-9]', i)
                aa = aa.group()
                walk_folders.append(aa)
            except:
                pass
            
        for i in walk_folders:
            for file in os.listdir(i):
                filename, file_extension = os.path.splitext(file)
                file_extension = file_extension.lower()
                if(file_extension == ".jpg" or file_extension==".jpeg" or file_extension==".png" or file_extension==".bmp"):
                    aa = re.search(self.path + r'.*/Camera[0-9]/[0-9]/.*.png', i + '/' + file)
                    aa = aa.group()
                    new_name_list = (aa.split('/')[-4:])
                    new_name = new_name_list[0] + '_' + new_name_list[1] + '_' + new_name_list[2] + '_' + new_name_list[3].split('.png')[0]
                    imgFile = basename(filename) + file_extension
                    if not imgFile.split('_'):
                        os.rename(i + '/' + imgFile, i + '/' + new_name + '.png')

                    xmlFile = basename(filename) + ".xml"
                    if not xmlFile.split('_'):
                        os.rename(i + '/' + xmlFile, i + '/' + new_name + '.xml')

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
                    train_val_test = np.random.choice(['train', 'val', 'test'], p=[0.6, 0.2, 0.2])
                    if train_val_test == 'train':
                        train.append(filepath)

                    elif  train_val_test == 'val':  
                        val.append(filepath)

                    elif  train_val_test == 'test':     
                         test.append(filepath)


        with tf.compat.v1.python_io.TFRecordWriter(self.path + 'train_images.tfrecords') as writer:
            for filepath in train:
                image_string = open(filepath, 'rb').read()
                tf_example = self.image_example(image_string, filepath)
                writer.write(tf_example.SerializeToString())

        with tf.compat.v1.python_io.TFRecordWriter(self.path + 'val_images.tfrecords') as writer:
            for filepath in val:
                image_string = open(filepath, 'rb').read()
                tf_example = self.image_example(image_string, filepath)
                writer.write(tf_example.SerializeToString())

        with tf.compat.v1.python_io.TFRecordWriter(self.path + 'test_images.tfrecords') as writer:
            for filepath in test:
                image_string = open(filepath, 'rb').read()
                tf_example = self.image_example(image_string, filepath)
                writer.write(tf_example.SerializeToString())

        return print('Write Done')  
    
    
    def show_image(self,count):
        raw_image_dataset = tf.data.TFRecordDataset(self.path + 'train_images.tfrecords')
        # Create a dictionary describing the features.
        image_feature_description = {

            'height': tf.compat.v1.FixedLenFeature([], tf.int64),
            'width': tf.compat.v1.FixedLenFeature([], tf.int64),

            'labelXmin': tf.compat.v1.FixedLenFeature([], tf.int64),
            'labelYmin': tf.compat.v1.FixedLenFeature([], tf.int64),
            'labelXmax': tf.compat.v1.FixedLenFeature([], tf.int64),
            'labelYmax':tf.compat.v1.FixedLenFeature([], tf.int64),

            'label': tf.compat.v1.FixedLenFeature([], tf.string),
            'image_raw': tf.compat.v1.FixedLenFeature([], tf.string),
        }

        parsed_image_dataset = raw_image_dataset.map(self._parse_image_function)

        c = 0 
        for image_features in parsed_image_dataset:
            image_raw = image_features['image_raw'].numpy()
            display.display(display.Image(data=image_raw))
            c+=1
            if c==count:
                break
                
    def go_pbtxt(self): 
        filename = self.path + '/object_detection.pbtxt'
        if not os.path.exists(filename):
            print("-----------make object_detection.pbtxt -----------")
            print("writeing to {}".format(filename))

            for dirPath, dirNames, fileNames in os.walk(self.path):
                class_labels = dirNames
                break
            class_values = list(range(len(class_labels)))
            classdict = dict(zip(class_labels,class_values))

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
        
