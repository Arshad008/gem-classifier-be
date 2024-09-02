import os
import cv2
import numpy as np
from common import known_classes

script_dir = os.path.dirname(__file__)
dataset_dir = os.path.join(script_dir, 'dataset\\')
train_dir = dataset_dir + 'train\\'
test_dir = dataset_dir + 'test\\'
model_path = os.path.join(script_dir,'model\\classifier_model.keras')

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
    # print("x value is")
    # print(predict_x)

    # print("ng value")
    # print(np.argmax(predict_x,axis=1))

    # Get the indices of the top 5 values sorted by max along the specified axis
    pred_class = np.argmax(predict_x,axis=1)[0]

    # get closer matches
    organized_matches = {}
    similar_indices = np.argsort(predict_x, axis=1)[:, -5:][:, ::-1]
    similar_values = np.take_along_axis(predict_x, similar_indices, axis=1)

    similar_indices = similar_indices[0]#.ravel()
    similar_values = similar_values[0]#.ravel()

    # print("Indixes")
    # print(similar_indices)
    # print("Values")
    # print(similar_values)

    # Organize values
    for indx in range(0, len(similar_indices)):
        organized_matches.update({known_classes[similar_indices[indx]]: similar_values[indx]})

    # print(pred_class)
    # print(known_classes[pred_class])

    return [known_classes[pred_class], str(organized_matches)]

# TODO uncomment following lines to test prediction
# model = initModel()
# predict_image(model,train_dir + 'Garnet Red\garnet red_2.jpg')