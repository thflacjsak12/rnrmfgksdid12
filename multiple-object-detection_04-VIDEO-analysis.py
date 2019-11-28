# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 00:13:47 2019

@author: Kim
"""

from imageai.Detection import VideoObjectDetection
import os
from matplotlib import pyplot as plt
import cv2

execution_path = os.getcwd()

color_index = {'bus': 'red', 'handbag': 'steelblue', 'giraffe': 'orange', 'spoon': 'gray', 
               'cup': 'yellow', 'chair': 'green', 'elephant': 'pink', 'truck': 'indigo', 
               'motorcycle': 'azure', 'refrigerator': 'gold', 'keyboard': 'violet', 
               'cow': 'magenta', 'mouse': 'crimson', 'sports ball': 'raspberry', 
               'horse': 'maroon', 'cat': 'orchid', 'boat': 'slateblue', 'hot dog': 'navy', 
               'apple': 'cobalt', 'parking meter': 'aliceblue', 'sandwich': 'skyblue', 
               'skis': 'deepskyblue', 'microwave': 'peacock', 'knife': 'cadetblue', 
               'baseball bat': 'cyan', 'oven': 'lightcyan', 'carrot': 'coldgrey', 
               'scissors': 'seagreen', 'sheep': 'deepgreen', 'toothbrush': 'cobaltgreen', 
               'fire hydrant': 'limegreen', 'remote': 'forestgreen', 'bicycle': 'olivedrab', 
               'toilet': 'ivory', 'tv': 'khaki', 'skateboard': 'palegoldenrod', 
               'train': 'cornsilk', 'zebra': 'wheat', 'tie': 'burlywood', 'orange': 'melon', 
               'bird': 'bisque', 'dining table': 'chocolate', 'hair drier': 'sandybrown',
               'cell phone': 'sienna', 'sink': 'coral', 'bench': 'salmon', 'bottle': 'brown', 
               'car': 'silver', 'bowl': 'maroon', 'tennis racket': 'palevilotered', 
               'airplane': 'lavenderblush', 'pizza': 'hotpink', 'umbrella': 'deeppink', 
               'bear': 'plum', 'fork': 'purple', 'laptop': 'indigo', 'vase': 'mediumpurple', 
               'baseball glove': 'slateblue', 'traffic light': 'mediumblue', 'bed': 'navy', 
               'broccoli': 'royalblue', 'backpack': 'slategray', 'snowboard': 'skyblue',
               'kite': 'cadetblue', 'teddy bear': 'peacock', 'clock': 'lightcyan', 
               'wine glass': 'teal', 'frisbee': 'aquamarine', 'donut': 'mincream', 
               'suitcase': 'seagreen', 'dog': 'springgreen', 'banana': 'emeraldgreen', 
               'person': 'honeydew', 'surfboard': 'palegreen', 'cake': 'sapgreen', 
               'book': 'lawngreen', 'potted plant': 'greenyellow', 'toaster': 'ivory', 
               'stop sign': 'beige', 'couch': 'khaki'}


resized = False

def forFrame(frame_number, output_array, output_count, returned_frame):

    plt.clf()

    this_colors = []
    labels = []
    sizes = []

    counter = 0
    print(output_array)

    for eachItem in output_count:
        print(eachItem, output_count[eachItem])
        counter += 1
        labels.append(eachItem + " = " + str(output_count[eachItem]))
        sizes.append(output_count[eachItem])
        this_colors.append(color_index[eachItem])

    global resized

#     if (resized == False):
#         manager = plt.get_current_fig_manager()
#         manager.resize(width=1000, height=500)
#         resized = True

    plt.subplot(1, 2, 1)
    plt.title("Frame : " + str(frame_number))
    plt.axis("off")
    plt.imshow(returned_frame, interpolation="none")

    plt.subplot(1, 2, 2)
    plt.title("Analysis: " + str(frame_number))
    plt.pie(sizes, labels=labels, colors=this_colors, shadow=True, startangle=140, autopct="%1.1f%%")

    plt.pause(0.01)

def forFrame2(frame_number, output_array, output_count, returned_frame):
    plt.clf()
    
    x = []
    y = []
    
    for eachItem in output_count:
        x.append(eachItem)
        y.append(output_count[eachItem])        
        
    plt.figure(figsize=(10,6))    
    plt.subplot(1,2,1)
    plt.title('Frame : '+ str(frame_number))
    plt.axis('off')
    plt.imshow(cv2.cvtColor(returned_frame, cv2.COLOR_BGR2RGB))
    
    plt.subplot(1,2,2)
    plt.title('Analysis : ' + str(frame_number))
    plt.bar(x, y)
    
    for i, v in enumerate(y):
        plt.text(i, v-2, str(v), color='white', fontweight='bold', fontsize=10)
    
    plt.pause(0.01)


    
    

video_detector = VideoObjectDetection()
video_detector.setModelTypeAsYOLOv3()
video_detector.setModelPath(os.path.join(execution_path, "models/yolo.h5"))
video_detector.loadModel()

plt.figure(figsize=(10,6))
plt.show()

video_detector.detectObjectsFromVideo(input_file_path=os.path.join(execution_path, "videos/drone_01.mp4"), 
                                      output_file_path=os.path.join(execution_path, "video_frame_analysis") ,
                                      frames_per_second=20, per_frame_function=forFrame2, 
                                      minimum_percentage_probability=30, 
                                      return_detected_frame=True)