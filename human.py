from imageai.Detection import ObjectDetection
import cv2 
import numpy as np
detector = ObjectDetection()

detector.setModelTypeAsYOLOv3()
from imageai.Detection import ObjectDetection

detector = ObjectDetection()
model_path = "./models/yolov5n-seg.pt"
detector.setModelTypeAsYOLOv3()
detector.setModelPath(model_path)
detector.loadModel()


vid = cv2.VideoCapture(0)

while(True):
    ret, frame = vid.read()
    a = detector.detectObjectsFromImage(input_image=frame, output_type= 'array')
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
vid.release()

cv2.destroyAllWindows()
        
    
    