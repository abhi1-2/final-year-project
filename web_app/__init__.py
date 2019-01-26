from flask import Flask ,render_template
from predict_by_image import gender_predictor
import os
current_dir=os.getcwd()
app=Flask(__name__,static_url_path = "/test_images", static_folder = "test_images")



@app.route('/home')
def home():
	print('opening home')
	print(current_dir)
	img_path=current_dir+'/test_images/srija_test.jpeg'
	gender=gender_predictor(img_path)
	full_filename='/test_images/srija_test.jpeg'
	return render_template('home.html',gender=gender,image=full_filename)




