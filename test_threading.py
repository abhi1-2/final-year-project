from threading import Thread
from predict_by_image import gender_predictor,age_predictor
import time
image="t.jpg"
'''class ThreadWithReturnValue(Thread):
	def __init__(self, group=None, target=None, name=None,
				 args=(), kwargs={}, Verbose=None):
		Thread.__init__(self, group, target, name, args, kwargs)
		self._return = None
	def run(self):
		print(type(self._target))
		if self._target is not None:
			self._return = self._target(*self._args,
												**self._kwargs)
	def join(self, *args):
		Thread.join(self, *args)
		return self._return
try:
	start=time.time()
	thread1 = ThreadWithReturnValue(target=gender_predictor, args=(image,))
	thread2 = ThreadWithReturnValue(target=age_predictor, args=(image,))
	thread1.start()
	thread2.start()

	print(thread1.join())
	print(thread2.join())
	end=time.time()
	print(end-start)'''
try:
	
	import multiprocessing
	pool = multiprocessing.Pool(processes = 2)
	start=time.time()
	p1 = multiprocessing.Process(target=gender_predictor,args=[image])
	p1.start()
	p2 = multiprocessing.Process(target=age_predictor,args=[image])
	p2.start()
	p1.join()
	p2.join()
	end=time.time()
	print(pool.map(p1,[0,1]))
	print(end-start)

except Exception as e:
	print(e)

try:
	start=time.time()
	print(gender_predictor(image))
	print(age_predictor(image))
	end=time.time()
	print(end-start)
except Exception as e:
	print(e)