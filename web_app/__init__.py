from flask import Flask ,render_template
from predict_by_image import gender_predictor
import os
current_dir=os.getcwd()
app=Flask(__name__)

@app.route('/home')
def home():
	print('opening home')
	#print(current_dir)
	
	gender=gender_predictor(current_dir+'/test_images/srija_test.jpeg')
	return render_template('home.html',gender=gender)




