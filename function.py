from flaskexample import app
from flask import Flask, render_template, request, redirect,url_for
import matplotlib.pyplot as plt
import io
import base64
import pickle
import cv2
import os
from flask_wtf import FlaskForm
from wtforms import Form, BooleanField, StringField, SelectField, validators, SubmitField
from flask_wtf.file import FileField, FileRequired,FileAllowed
from werkzeug.utils import secure_filename
from scipy import spatial
import operator
import pandas as pd
import numpy as np
import keras
import matplotlib.image as mpimg
import h5py
from keras.layers import Flatten, Dense, Input,concatenate
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout
from keras.models import Model
from keras.models import Sequential
import tensorflow as tf
import glob
from flask_table import Table, Col, LinkCol
import pdb
from keras import backend as K
from PIL import Image



def get_vector(filename): 
    """ takes filename, returns vector"""
    K.clear_session()
    vgg16 = keras.applications.VGG16(weights='imagenet', include_top=True, pooling='max', input_shape=(224, 224, 3))
    basemodel = Model(inputs=vgg16.input, outputs=vgg16.get_layer('fc2').output)
    img= cv2.imread(filename)
    img1 = cv2.resize(img, (224, 224))
    feature_vector = basemodel.predict(img1.reshape(1, 224, 224, 3))
    vector = feature_vector[0]
    return vector

def cosine_similarity(x, y):
    """x and y are vectors"""
    return 1 - spatial.distance.cosine(x, y)

#function to get cluster number from input brand 
def get_brand_cluster(df, input_brand):
    try:
        return int(df[df.brand == input_brand.lower()].cluster)
    except:
        return 0

### function from YOLO to input an image and return bounding box and class ids

###load YOLO weights, config, and class###
# 'path to yolo config file' 
CONFIG='flaskexample/YOLO/yolov3.cfg'
# 'path to text file containing class names'
CLASSES='flaskexample/YOLO/yolov3.txt'
# 'path to yolo pre-trained weights' 
WEIGHTS='flaskexample/YOLO/yolov3.weights'

### YOLO shit, don't mess with it##
# read class names from text file
classes = None
with open(CLASSES, 'r') as f:
     classes = [line.strip() for line in f.readlines()]
        
scale = 0.00392
conf_threshold = 0.5
nms_threshold = 0.4

# generate different colors for different classes 
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

# function to get the output layer names 
# in the architecture
def get_output_layers(net): 
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers

# function to draw bounding box on the detected object with class name
def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])
    color = COLORS[class_id]
    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)
    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

### still YOLO shit, don't mess with it
def get_bounding_class(img): 
    image = cv2.imread(img)
    Width = image.shape[1]
    Height = image.shape[0]
    scale = 0.00392

    classes = None

    with open(CLASSES, 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

    net = cv2.dnn.readNet(WEIGHTS, CONFIG)

    blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)

    net.setInput(blob)

    outs = net.forward(get_output_layers(net))

    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4


    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])


    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        draw_bounding_box(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))

    return { "filename":img, "class_ids":class_ids, "boxes":boxes}
    ### YOLO shit end ##

def CropImage(input_dictionary):
    filename = input_dictionary['filename']
    image = mpimg.imread(filename)
    simple_file_name = os.path.splitext(filename.strip('flaskexample/static/uploads/'))[0]
    
    # Select boxes corresponding to class_id = 0
    all_class_ids = input_dictionary['class_ids']
    all_boxes = input_dictionary['boxes']
    
    selected_class_ids = []
    selected_boxes = []
    for class_id, box in zip(all_class_ids, all_boxes):
        if class_id == 0:
            selected_class_ids.append(class_id)
            selected_boxes.append(box)
        
    # Loop over all selected boxes, and crops image to boxes
    for i,box in enumerate(selected_boxes):
        x = int(box[0])
        y = int(box[1])
        w = int(box[2])
        h = int(box[3])

        cropped = image[y:y+h, x:x+w]
        np_im = np.array(cropped)
        shape = np_im.shape
        if shape[0]>50 or shape[1]>30:
            new_im = Image.fromarray(cropped)
            fill_new_img = make_square(new_im)
            name= 'flaskexample/YOLO/resize_fill_photos/'+simple_file_name+'_resize.png'
            fill_new_img.save(name)
    return fill_new_img, name

##### function to resize cropped images and fill-in white color
def make_square(im, min_size=225, fill_color=(255, 255, 255, 255)):
    x, y = im.size
    size = max(min_size, x, y)
    new_im = Image.new('RGBA', (size, size), fill_color)
    new_im.paste(im, (int((size - x) / 2), int((size - y) / 2)))
    return new_im

##