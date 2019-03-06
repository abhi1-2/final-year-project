from web_app import db

class Products(db.Model):
	id=db.Column(db.Integer, primary_key=True)
	name=db.Column(db.String(80), unique=True, nullable=False)
	price=db.Column(db.Float, unique=True, nullable=False)
	description=db.Column(db.String(300),nullable=True)
	picture_url=db.Column(db.String(200),nullable=False,default='test_images/default.jpeg')
	age_group=db.Column(db.String(20),nullable=False)
	gender=db.Column(db.String(20),nullable=False)
	ethinicity=db.Column(db.String(20),nullable=False,default='asian')
	

	def __repr__(self):
		return f"Product('{self.id}','{self.name}','{self.price}')"

