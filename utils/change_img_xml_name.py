#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 15:04:19 2019

@author: thomas_yang
"""

import os
import scipy
import numpy as np
import shutil

def list_files(in_path):
    files = []
    for (dirpath, dirnames, filenames) in os.walk(in_path):
        files.extend(filenames)
        break

    return files

def _get_files(img_dir):
    files = list_files(img_dir)
    return [os.path.join(img_dir,x) for x in files]


ai_competition_unzip_data_path = '/home/thomas_yang/Downloads/ai_competition_unzip_data/'
ai_competition_rename_data = '/home/thomas_yang/Downloads/ai_competition_rename_data/'

class_folders = os.listdir(ai_competition_unzip_data_path)
for class_folder in class_folders:
    class_folders_Cameras = os.path.join(ai_competition_unzip_data_path,class_folder)
    Cameras = os.listdir(class_folders_Cameras)
    Cameras.sort()
    for Camera in Cameras:
        Cameras_views = os.path.join(class_folders_Cameras, Camera)
        views = os.listdir(Cameras_views)
        views.sort()
        for view in views:
            view_name = (ai_competition_unzip_data_path + '{Class}/{Camera}/{View}'.format(Class=class_folder, Camera=Camera, View=view))
            data_file = _get_files(view_name)
            data_file.sort()    
            for i in np.arange(0, len(data_file)):
                print(data_file[i])
                new_name = os.path.join('/'+ data_file[i].split('/')[1],
                                        data_file[i].split('/')[2],
                                        data_file[i].split('/')[3],
                                        data_file[i].split('/')[4],
                                        data_file[i].split('/')[5],
                                        data_file[i].split('/')[6],
                                        data_file[i].split('/')[7],
                                        data_file[i].split('/')[5]+
                                        data_file[i].split('/')[6]+
                                        data_file[i].split('/')[7]+
                                        data_file[i].split('/')[8])
                os.rename(data_file[i], new_name)
                shutil.move(new_name, ai_competition_rename_data)



