import cv2
import argparse
import os
from fashion_detection_model import SSD, Predictor
from utils.utils import draw_boxes
import glob

class_names = ['BACKGROUND', 'sunglass', 'hat', 'jacket', 'shirt', 'pants', 'shorts', 'skirt', 'dress', 'bag', 'shoe']
model_path = './models/vgg16-ssd-Epoch-125-Loss-2.8042236075681797.pth'


net = SSD(len(class_names), is_test=True)
net.load(model_path)
predictor = Predictor(net, candidate_size=200)
data_path = glob.glob('./image/plain*.jpg')
for image_path in data_path:
    img = cv2.imread(image_path)
    boxes, labels, probs = predictor.predict(img, 30, 0.5)
    img = draw_boxes(img, boxes, labels, probs, class_names)
    cv2.imwrite(image_path.replace('image', 'output'), img)
