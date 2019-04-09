import cv2

import numpy as np
import dlib
import os

face_detector = dlib.get_frontal_face_detector()

predictor_path = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)


class find_specs:
    # return the list of (x, y)-coor
    def __init__(self,path):
        self.isSpecs=False
        image_path=path
        img=cv2.imread(image_path)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces = face_detector(gray, 1)
        if(faces):
            for face_rect in (faces):
                #print(face_rect)
                x=face_rect.left()
                y=face_rect.top()
                w=face_rect.right()-x
                h=face_rect.bottom()-y
                
                landmarks = predictor(gray, face_rect)
                
                #landmarks = landmarks_to_np(landmarks)
                
                LEFT_EYE_CENTER,RIGHT_EYE_CENTER=self.get_eye_centres(landmarks)
                aligned_face = self.get_aligned_face(gray, LEFT_EYE_CENTER, RIGHT_EYE_CENTER)
        #cv2.imshow("aligned_face #{}".format(i + 1), aligned_face)

        

                judge = self.judge_eyeglass(aligned_face)
                self.isSpecs=judge
        else:
            pass

    def get_eye_centres(self,landmarks):
        coords = np.zeros((landmarks.num_parts, 2), dtype='int')
        for i in range(landmarks.num_parts):
            coords[i]=(landmarks.part(i).x,landmarks.part(i).y)
        landmarks=coords
        #cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        '''for j,(x, y) in enumerate(landmarks):
            cv2.circle(img, (x, y), 1, (0, 0, 255), -1)
            cv2.putText(img,str(j),(x+4, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255))'''
        EYE_LEFT_OUTTER = landmarks[36]
        EYE_LEFT_INNER = landmarks[39]
        EYE_RIGHT_OUTTER = landmarks[45]
        EYE_RIGHT_INNER = landmarks[42]

        '''x = ((landmarks[0:4]).T)[0]
        y = ((landmarks[0:4]).T)[1]
        A = np.vstack([x, np.ones(len(x))]).T
        k, b = np.linalg.lstsq(A, y, rcond=None)[0]'''
        
        x_left = (EYE_LEFT_OUTTER[0]+EYE_LEFT_INNER[0])/2
        y_left=(EYE_LEFT_OUTTER[1]+EYE_LEFT_INNER[1])/2
        x_right = (EYE_RIGHT_OUTTER[0]+EYE_RIGHT_INNER[0])/2
        y_right=(EYE_RIGHT_OUTTER[1]+EYE_RIGHT_INNER[1])/2
        LEFT_EYE_CENTER =  np.array([np.int32(x_left), np.int32(y_left)])
        RIGHT_EYE_CENTER =  np.array([np.int32(x_right), np.int32(y_right)])
        return LEFT_EYE_CENTER,RIGHT_EYE_CENTER
        #cv2.circle(img, (LEFT_EYE_CENTER[0],LEFT_EYE_CENTER[1]), 1, (0, 0, 255), -1)
        #cv2.circle(img, (RIGHT_EYE_CENTER[0],RIGHT_EYE_CENTER[1]), 1, (0, 0, 255), -1)
    def get_aligned_face(self,img, left, right):
        desired_w = 256
        desired_h = 256
        desired_dist = desired_w * 0.5
        
        eyescenter = ((left[0]+right[0])*0.5 , (left[1]+right[1])*0.5)# 眉心
        dx = right[0] - left[0]
        dy = right[1] - left[1]
        dist = np.sqrt(dx*dx + dy*dy)# 瞳距
        scale = desired_dist / dist # 缩放比例
        angle = np.degrees(np.arctan2(dy,dx)) # 旋转角度
        M = cv2.getRotationMatrix2D(eyescenter,angle,scale)# 计算旋转矩阵

        # update the translation component of the matrix
        tX = desired_w * 0.5
        tY = desired_h * 0.5
        M[0, 2] += (tX - eyescenter[0])
        M[1, 2] += (tY - eyescenter[1])

        aligned_face = cv2.warpAffine(img,M,(desired_w,desired_h))
        
        return aligned_face
    def judge_eyeglass(self,img):
        img = cv2.GaussianBlur(img, (11,11), 0) 

        sobel_y = cv2.Sobel(img, cv2.CV_64F, 0 ,1 , ksize=-1) 
        sobel_y = cv2.convertScaleAbs(sobel_y) 
        #cv2.imshow('sobel_y',sobel_y)

        edgeness = sobel_y 
        
        
        retVal,thresh = cv2.threshold(edgeness,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        
        
        d = len(thresh) * 0.5
        x = np.int32(d * 6/7)
        y = np.int32(d * 1/2)
        w = np.int32(d * 2/7)
        h = np.int32(d * 3/4)
        
        roi = thresh[y:y+h, x:x+w] 
        
        measure = sum(sum(roi/255)) / (np.shape(roi)[0] * np.shape(roi)[1])
        
        
        #print(measure)
        
        
        if measure > 0.07:
            judge = True
        else:
            judge = False
        #print(judge)
        return judge

    


