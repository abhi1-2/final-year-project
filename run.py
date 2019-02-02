from web_app import app 


if __name__=='__main__':
	app.secret_key='123456'
	app.run(debug=True,host='0.0.0.0',port=8083,threaded=True)