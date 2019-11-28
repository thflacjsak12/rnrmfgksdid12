# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 00:07:55 2019

@author: Kim
"""

from imageai.Detection import VideoObjectDetection
import imageai
import os

execution_path = os.getcwd()

# detector = VideoObjectDetection()
# detector.setModelTypeAsRetinaNet()
# detector.setModelPath( os.path.join(execution_path , "models/resnet50_coco_best_v2.0.1.h5"))
detector = VideoObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath( os.path.join(execution_path , "models/yolo.h5"))

detector.loadModel(detection_speed="fast") #fast, faster, fastest, flash

video_path = detector.detectObjectsFromVideo(
    input_file_path=os.path.join(execution_path, "videos/seoul_02_0.mp4"),
    output_file_path=os.path.join(execution_path, "video_out2"),
    frames_per_second=20, log_progress=True)
print(video_path)