import os
import cv2
import numpy as np

script_dir = os.path.dirname(__file__)
dataset_dir = os.path.join(script_dir, 'dataset\\')
train_dir = dataset_dir + 'train\\'
test_dir = dataset_dir + 'test\\'
model_path = os.path.join(script_dir,'model\\classifier_model.h5')

CLASSES, gems = [], [] # names of classes, count of images for each class
img_w, img_h = 220, 220    # width and height of image

from common import known_classes
from image_helper import edge_and_cut, prepare_image
from tensorflow.keras.models import Sequential, load_model

def initModel():
    return load_model(model_path)

def predict_image(model: Sequential, img_path: str):
    image = prepare_image(img_path)
    image = edge_and_cut(image)
    pred_image = np.array([image])

    # print(pred_image)
    predict_x = model.predict(pred_image)
    pred_class = np.argmax(predict_x,axis=1)[0]

    # print(pred_class)
    # print(known_classes[pred_class])

    return known_classes[pred_class]

# TODO uncomment following lines to test prediction
# model = initModel()
# predict_image(model,train_dir + 'Garnet Red\garnet red_2.jpg')