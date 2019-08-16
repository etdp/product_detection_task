
from __future__ import print_function

import numpy as np
import cv2 as cv
import time
import json
from os import listdir
from os.path import isfile, isdir, join
import glob, os
from PIL import Image

class App():
    BLUE = [255,0,0]        # rectangle color
    RED = [0,0,255]         # PR BG
    GREEN = [0,255,0]       # PR FG

    DRAW_BG = {'color' : GREEN, 'val' : 2}
    DRAW_FG = {'color' : RED, 'val' : 1}

    # setting up flags
    rect = (0,0,1,1)
    drawing = False         # flag for drawing curves
    rectangle = False       # flag for drawing rect
    rect_over = False       # flag to check if rect drawn
    rect_or_mask = 100      # flag for selecting rect or mask mode
    value = DRAW_FG         # drawing initialized to FG
    thickness = 10           # brush thickness

    def onmouse(self, event, x, y, flags, param):

        # draw touchup curves

        if event == cv.EVENT_LBUTTONDOWN:
#            if self.rect_over == False:
            print("first draw rectangle \n")
#            else:
            self.drawing = True
            cv.circle(self.img, (x,y), self.thickness, self.value['color'], -1)
            cv.circle(self.mask, (x,y), self.thickness, self.value['val'], -1)

        elif event == cv.EVENT_MOUSEMOVE:
            if self.drawing == True:
                cv.circle(self.img, (x, y), self.thickness, self.value['color'], -1)
                cv.circle(self.mask, (x, y), self.thickness, self.value['val'], -1)

        elif event == cv.EVENT_LBUTTONUP:
            if self.drawing == True:
                self.drawing = False
                cv.circle(self.img, (x, y), self.thickness, self.value['color'], -1)
                cv.circle(self.mask, (x, y), self.thickness, self.value['val'], -1)

    def run(self):
        data_dir = ("/10T/database/ai_competition_crop/4710757030200/")
        save_dir = ("/10T/database/ai_competition_removeBG/4710757030200/")
        for camera_idx, camera_file in enumerate(sorted(os.listdir(data_dir))):
#            for camera_file in (os.listdir(data_dir+product_file)):
            for position_file in (os.listdir(data_dir+camera_file)):
                path_ = os.path.join(data_dir,camera_file,position_file,'*.png')
                image_path = glob.glob(path_)
                if os.path.isfile(image_path[0]):
                    image_path = image_path
                else:
                    path_ = os.path.join(data_dir,camera_file,position_file,'*.jpg')
                    image_path = glob.glob(path_)   
                image_path.sort()
                
                
                for image in range(len(image_path)): 
                    # Loading images
                    self.img_ = Image.open(image_path[image])
                    width, height = self.img_.size
                    self.rect = (180,180,width-200-180,height-200-180) 
                    cv.namedWindow('input')
                    self.img = cv.imread(image_path[image],cv.IMREAD_ANYCOLOR)
            #        self.img = cv.resize(self.img,(640,480),interpolation=cv.INTER_CUBIC) 
                    self.img2 = self.img.copy()                               # a copy of original image
                    self.mask = np.zeros(self.img.shape[:2], dtype = np.uint8) # mask initialized to PR_BG
                    self.output = np.zeros(self.img.shape, np.uint8)           # output image to be shown
            
                    # input and output windows
                    cv.namedWindow('output')
                    cv.moveWindow('output', 500,500)
                    cv.setMouseCallback('input', self.onmouse)
                    cv.moveWindow('input', 1200,800)
                    
#                        if (self.rect_or_mask == 0):         # grabcut with rect      
                    tStart_1 = time.time()#計時開始
                    bgdmodel = np.zeros((1, 65), np.float64)
                    fgdmodel = np.zeros((1, 65), np.float64)
                    cv.grabCut(self.img2, self.mask, self.rect, bgdmodel, fgdmodel, 5, cv.GC_INIT_WITH_RECT)
                    tEnd_1 = time.time()#計時結束
                    print('time_1 cost %f sec' % (tEnd_1 - tStart_1))
                    self.rect_or_mask = 1
            
                    print(" Instructions: \n")
                    print(" Draw a rectangle around the object using right mouse button \n")
            
                    while(1):
            
                        cv.imshow('output', self.output)
                        cv.imshow('input', self.img)
                        k = cv.waitKey(1)
            
                        # key bindings
                        if k == 27:         # esc to exit
                            cv.destroyAllWindows()
                            break
                        elif k == ord('2'): # BG drawing
                            print(" mark background regions with left mouse button \n")
                            self.value = self.DRAW_BG
                        elif k == ord('1'): # FG drawing
                            print(" mark foreground regions with left mouse button \n")
                            self.value = self.DRAW_FG
                        elif k == ord('s'): # save image
            #                bar = np.zeros((self.img.shape[0], 5, 3), np.uint8)
            #                res = np.hstack((self.img2, bar, self.img, bar, self.output))
                            save_dir_1 = os.path.join(save_dir)
                            
                            if os.path.isdir(save_dir_1):
                                save_dir_2 = os.path.join(save_dir,camera_file)
                            else:
                                os.makedirs(save_dir_1)
                                save_dir_2 = os.path.join(save_dir,camera_file)
                            if os.path.isdir(save_dir_2):
                                save_dir_3 = os.path.join(save_dir,camera_file,position_file)
                            else:
                                os.makedirs(save_dir_2)
                                save_dir_3 = os.path.join(save_dir,camera_file,position_file)
                            if os.path.isdir(save_dir_3):
                                name = image_path[image].split('/')[-1]
                                cv.imwrite(save_dir_3 + '/' + name, self.output)
                                print(" Result saved as image \n")
                                break
                            else:
                                os.makedirs(save_dir_3)
                                name = image_path[image].split('/')[-1]
                                cv.imwrite(save_dir_3 + '/' + name, self.output)
                                print(" Result saved as image \n")
                                break
                        elif k == ord('n'): # segment the image
                            if self.rect_or_mask == 1:         # grabcut with mask        
                                tStart_2 = time.time()#計時開始
                                bgdmodel = np.zeros((1, 65), np.float64)
                                fgdmodel = np.zeros((1, 65), np.float64)
                                cv.grabCut(self.img2, self.mask, self.rect, bgdmodel, fgdmodel, 1, cv.GC_INIT_WITH_MASK)
                                tEnd_2 = time.time()#計時結束
                                print('time_2 cost %f sec' % (tEnd_2 - tStart_2))
            
                        mask2 = np.where((self.mask==1) + (self.mask==3), 255, 0).astype('uint8')
                        self.output = cv.bitwise_and(self.img2, self.img2, mask=mask2)
            
                    print('Done')
                    output_img = self.output
#                        return output_img


def videocap():
    #获得视频的格式
    cap = cv.VideoCapture('D:/spyder/multi-view-angle/VIDEO0005.mp4')
    c = 1  
    
    if cap.isOpened(): #判断是否正常打开
        success, frame = cap.read()
    else:
        success = False
    
    while success :  
        
        success, frame = cap.read() 
        if(c == 100):
            original_img = cv.resize(frame, (640, 480), interpolation=cv.INTER_CUBIC)
            cv.imwrite("D:/spyder/multi-view-angle/img/VIDEO0005.bmp", original_img)
        elif(c > 100):
            break
        
        c = c + 1
        
    cap.release()

    
if __name__ == '__main__':
    print(__doc__)
    output_img = App().run()
