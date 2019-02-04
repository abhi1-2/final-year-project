from flask import Flask ,render_template,request,url_for,Response,redirect,session
from predict_by_image import gender_predictor,age_predictor
import base64
import os
import cv2
vc=cv2.VideoCapture(0)
current_dir=os.getcwd()
app=Flask(__name__,static_url_path = "/test_images", static_folder = "test_images")



@app.route('/test')
def home():
	print('opening home')
	print(current_dir)
	img_path=current_dir+'/test_images/srija_test.jpeg'
	gender=gender_predictor_keras(img_path)#gender_predictor(img_path)
	full_filename='/test_images/srija_test.jpeg'
	return render_template('home.html',gender=gender,image=full_filename)

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
	
	#session['gender']=gender
	gender=gender_predictor(image)
	print(gender)
	
	age=age_predictor(image)
	print(age)
	return render_template('output.html',gender=gender,age=age)
def gen():
    """Video streaming generator function."""
    while True:
        rval, frame = vc.read()
        cv2.imwrite('t.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + open('t.jpg', 'rb').read() + b'\r\n')


@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


