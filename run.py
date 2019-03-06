from web_app import app 
from web_app import db


if __name__=='__main__':
	db.create_all()
	app.secret_key='123456'
	app.run(debug=True,host='0.0.0.0',port=8083,threaded=True)