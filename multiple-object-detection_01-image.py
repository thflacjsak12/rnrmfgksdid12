# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 23:50:48 2019

@author: Kim
"""

from imageai.Detection import ObjectDetection
import os

execution_path = os.getcwd()
# print(execution_path)

# 모델 설정하기
detector = ObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath( os.path.join(execution_path , "models/yolo.h5"))

# 모델 성능 설정하기
detector.loadModel(detection_speed="fast") #fast, faster, fastest, flash

# 모델에 이미지 입력 및 출력 설정하기
detections = detector.detectObjectsFromImage(
    input_image=os.path.join(execution_path , "images/street.jpg"), 
    output_image_path=os.path.join(execution_path , "image_out.jpg"), 
    minimum_percentage_probability=50)

# 실행결과 출력하기
for eachObject in detections:
    print(eachObject["name"] , " : ", eachObject["percentage_probability"], " : ", eachObject["box_points"] )
    print("--------------------------------")
