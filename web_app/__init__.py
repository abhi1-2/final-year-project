from flask import Flask ,render_template,request,url_for,Response,redirect,session,jsonify
from predict_by_image import gender_predictor,age_predictor
from predict_by_image import spectacles_predictor
import base64
import os
import cv2
import dlib
import json
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from flask_migrate import Migrate
	
face_detector = dlib.get_frontal_face_detector()
vc=cv2.VideoCapture(0)
current_dir=os.getcwd()
app=Flask(__name__,static_url_path = "/test_images", static_folder = "test_images")
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///test_new.db'
db=SQLAlchemy(app)
migrate = Migrate(app, db)
CORS(app,resources=r"/data/*")


from web_app import models


@app.route('/data/<age>/<gender>')

def get_all_data_(age,gender):
	from web_app.models import Products
	pro=Products().query.filter_by(age_group=age,gender=gender).all()

	
	items=[]
	for item in pro:
		item_dict={
		'img_url':item.picture_url,
		'name':item.name,
		'price':item.price,
		'description':item.description,
		'age':age,
		'gender':gender,

		}
		#print(items)
		items.append(item_dict)
	item=Products().query.filter_by(glasses=True).first()
	item_dict={
		'img_url':item.picture_url,
		'name':item.name,
		'price':item.price,
		'description':item.description,
		'age':age,
		'gender':gender,

		}
	items.append(item_dict)	
	print(items)
	return Response(json.dumps(items),  mimetype='application/javascript')



@app.route('/test')
def home():
	
	
	
	print(current_dir)
	img_path=current_dir+'/test_images/srija_test.jpeg'
	gender=gender_predictor(img_path)
	age=age_predictor(img_path)
	
	'''if gender!='male':
			beard=beard_classifier(img_path)
	specs=specs_classifier(img_path)
			
		
	'''
	#data=get_all_data_('0-11','male')

	print(age)
	full_filename='/test_images/srija_test.jpeg'
	return render_template('home.html')

@app.route('/')

def index():
    """Video streaming home page."""

    return render_template('index.html')

#onclick of button
@app.route('/predict')

def predict():
	#gender=gender_predictor(current_dir+'/t.jpg')
	#vc.release()
	image=current_dir+'/t.jpg'
	#image=current_dir+'/test_images/female_1.jpeg'
	
	#session['gender']=gender
	gender=gender_predictor(image)
	print(gender)
	
	age=age_predictor(image)
	print(age)
	glasses=spectacles_predictor(image)
	print(glasses)
	return render_template('output.html',gender=gender,age=age,glasses=glasses,ethnicity="Asian")
def gen():
    """Video streaming generator function."""
    while True:
        rval, frame = vc.read()
        '''gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected_faces = face_detector(gray,1)
        for i,face_rect in enumerate(detected_faces):
        	for (x, y, w, h) in face_rect:
        		cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)'''
        cv2.imwrite('t.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + open('t.jpg', 'rb').read() + b'\r\n')


@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


