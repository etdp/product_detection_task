import sys
import os
from datetime import datetime
import numpy as np
from PIL import Image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf


label_map = {1: '04003906',
             2: '748675116052',
             3: '4710043001433',
             4: '4710095046208',
             5: '4710105030326',
             6: '4710126035003',
             7: '4710126041004',
             8: '4710126045460',
             9: '4710126100923',
             10: '4710128020106',
             11: '4710174114095',
             12: '4710298161234',
             13: '4710423051096',
             14: '4710543006693',
             15: '4710594924427',
             16: '4710626186519',
             17: '4710757030200',
             18: '4711162821520',
             19: '4711202224892',
             20: '4711402892921',
             21: '4713507024627',
             22: '4714431053110',
             23: '4719264904219',
             24: '4719264904233',
             25: '4902777062013',
             26: '7610700600863',
             27: '8801111390064',
             28: '8886467102400',
             29: '8888077101101',
             30: '8888077102092'}

PATH_TO_TEST_IMAGES_LIST = 'input.txt'

if hasattr(sys, "_MEIPASS"):
   PATH_TO_FROZEN_GRAPH = os.path.join(sys._MEIPASS, 'inference_graph.pb')
else:
   PATH_TO_FROZEN_GRAPH = 'inference_graph.pb'


detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


with detection_graph.as_default(), tf.compat.v1.Session() as sess, \
		open('output.txt', 'w') as fout:
    image_tensor = tf.compat.v1.get_default_graph().get_tensor_by_name('image_tensor:0')

    tensor_output = {}
    for key in ['num_detections', 'detection_boxes',
                'detection_scores', 'detection_classes']:
        tensor_output[key] = tf.compat.v1.get_default_graph().get_tensor_by_name(key + ':0')

    with open(PATH_TO_TEST_IMAGES_LIST, 'r') as f:
        fileList = f.read().splitlines()

    for imgFile in fileList:
        image = Image.open(imgFile)
        img_input = np.array(image)

        print(datetime.now(), file=fout)
        print(imgFile, file=fout)

        image_np_expanded = np.expand_dims(img_input, axis=0)

        # Run inference
        output_dict = sess.run(tensor_output,
                               feed_dict={image_tensor: image_np_expanded})

        output_dict['num_detections'] = output_dict['num_detections'][0]
        output_dict['detection_classes'] = output_dict['detection_classes'][0]
        output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
        output_dict['detection_scores'] = output_dict['detection_scores'][0]

        index = 1
        for classID, bbox in zip(output_dict['detection_classes'], output_dict['detection_boxes']):
            print(index, label_map[int(classID)], *bbox, sep=',', file=fout)
            index += 1

            if index > 3:
                break

        print(datetime.now(), file=fout)
