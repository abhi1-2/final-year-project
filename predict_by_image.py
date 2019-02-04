
# coding: utf-8

# In[23]:

import dlib
#import glob
#import json
import os
#from skimage import io
#import sys
import cvlib as cv
import numpy as np
import cv2
import tensorflow as tf
current_dir=os.getcwd()+'/'
from keras.models import model_from_json
from keras import backend as k
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from keras.utils import get_file
import imutils
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
from imutils import face_utils
face_detector = dlib.get_frontal_face_detector()
predictor_model=current_dir+ "shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(predictor_model)
facerec=dlib.face_recognition_model_v1(current_dir+'dlib_face_recognition_resnet_model_v1.dat')
json_file = open(current_dir+'models/model_age_1.json', 'r')
loaded_model_json = json_file.read()
json_file.close()


def gender_predictor(image):    
    #image      = 'test_images/male_2.jpeg'
    img = cv2.imread(image)
    pos_db=['0','1']
    alpha=['male','female']
    
    #fa=FaceAligner(predictor, desiredFaceWidth=256)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detected_faces = face_detector(gray, 1)

    if(detected_faces):
        for i,face_rect in enumerate(detected_faces):
                shape=predictor(gray,face_rect)
                face_descriptor = facerec.compute_face_descriptor(img,shape)
                face_descriptor=np.array(face_descriptor)
                face_descriptor=face_descriptor.reshape((-1,1*128)).astype(np.float32)
                saver = tf.train.import_meta_graph('models/test_model3_gender.ckpt.meta')
                graph = tf.get_default_graph()   
                with tf.Session(graph=graph) as session:
                    saver.restore(session,'models/test_model3_gender.ckpt')
                    weights_l1=tf.trainable_variables()[0]
                    biases_l1=tf.trainable_variables()[1]
                    weights_l2=tf.trainable_variables()[2]
                    biases_l2=tf.trainable_variables()[3]
                    weights_l3=tf.trainable_variables()[4]
                    biases_l3=tf.trainable_variables()[5] 
                    logits_1=tf.matmul(face_descriptor,weights_l1)+biases_l1
                    logits_1=tf.nn.relu(logits_1)
                    keep_prob1=tf.placeholder(tf.float32)
                    drop_out=tf.nn.dropout(logits_1,keep_prob1)
                    logits_2=tf.matmul(logits_1,weights_l2)+biases_l2
                    # keep_prob=tf.placeholder(tf.float32)
                    #drop_out=tf.nn.dropout(logits_2,keep_prob)
                    logits_2=tf.nn.relu(logits_2)
                    logits_3=tf.matmul(logits_2,weights_l3)+biases_l3 
                    output=tf.nn.softmax(logits_3).eval()
                    index=np.argmax(output)
                    return alpha[int(pos_db[index])]


    else:
        return 0
def age_predictor( path):
    alpha=['baby','teen','young','mid-age','old','very old']
    
    
    #print("Loaded model from disk")




    image=path
    image=cv2.imread(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detected_faces = face_detector(gray, 1)
    if(detected_faces):
        for i,face_rect in enumerate(detected_faces):
            shape=predictor(gray,face_rect)
            face_descriptor = facerec.compute_face_descriptor(image,shape)
            face_descriptor=np.array(face_descriptor)
            face_descriptor=np.array(face_descriptor.reshape((-1,1*128)).astype(np.float32))
            
            with tf.Graph().as_default():
                with tf.Session() as sess:
                    k.set_session(sess)
                    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
                    loaded_model.load_weights(current_dir+"models/model_age_1.h5")
                    index=int(loaded_model.predict(face_descriptor).argmax())
                    return alpha[index]
    else :
        return 0                

# In[24]:
def gender_predictor_keras(img_path):
    dwnld_link = "https://s3.ap-south-1.amazonaws.com/arunponnusamy/pre-trained-weights/gender_detection.model"
    model_path = get_file("gender_detection.model", dwnld_link,
                         cache_subdir="pre-trained", cache_dir=os.getcwd())
                         
    # load model
    #model = load_model(model_path)

    # read input image
    #img_path=os.getcwd()+'/t.jpg'
    image = cv2.imread(img_path)



    # load pre-trained model


    if image is None:
        print("Could not read input image")
        exit()

    # load pre-trained model


    # detect faces in the image
    face, confidence = cv.detect_face(image)

    classes = ['man','woman']

    # loop through detected faces
    for idx, f in enumerate(face):

         # get corner points of face rectangle       
        (startX, startY) = f[0], f[1]
        (endX, endY) = f[2], f[3]

        # draw rectangle over face
        #cv2.rectangle(image, (startX,startY), (endX,endY), (0,255,0), 2)

        # crop the detected face region
        face_crop = np.copy(image[startY:endY,startX:endX])

        # preprocessing for gender detection model
        try:
            face_crop = cv2.resize(face_crop, (96,96))
        except:
            pass
        face_crop = face_crop.astype("float") / 255.0
        face_crop = img_to_array(face_crop)
        face_crop = np.expand_dims(face_crop, axis=0)

        # apply gender detection on face
        with tf.Graph().as_default():
            with tf.Session() as sess:
                k.set_session(sess)
                model = load_model(model_path)
                conf = model.predict(face_crop)[0]
                return classes[conf.argmax()]




def ethinicity_predictor():
    return 0

# In[ ]:




# In[ ]:



