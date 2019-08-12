#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 17:06:30 2019

@author: thomas_yang
"""

from xml.etree.ElementTree import parse, Element
import os

def list_files(in_path):
    files = []
    for (dirpath, dirnames, filenames) in os.walk(in_path):
        files.extend(filenames)
        break
    return files

def _get_files(img_dir):
    files = list_files(img_dir)
    return [os.path.join(img_dir,x) for x in files]

data_file = _get_files('/home/thomas_yang/ML/TensorFlow/workspace/training_demo/images/train')
data_file.sort()

for file in data_file:
    if file.endswith('.xml'):        
        doc = parse(file)
        root = doc.getroot()
        
        index = root.getchildren().index(root.find('filename'))
        root.remove(root.find('filename'))
        
        e = Element('filename')
        
        if os.path.isfile(file.replace('xml', 'png')):
            e.text = file.split('/')[-1].replace('xml', 'png')
        else :
            e.text = file.split('/')[-1].replace('xml', 'jpg')
            
        root.insert(index, e)
        doc.write(file, xml_declaration=True)
        