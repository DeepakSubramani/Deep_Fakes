import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np
import skimage

class Deep:

	def __init__(self):
		self.trump = self.load_data('trump')
		self.cage = self.load_data('cage')

	def load_data(self, name):
		"""
		Loads images.
		"""
		images = []
		labels = []
		for root, dirs, files in os.walk('data/' + name):
			for name in files:
				filepath = os.path.join(root, name)
				if filepath.endswith('.jpg'):
					image = plt.imread(filepath)/255
					images.append(image)
		return np.array(images)

	def plot_progress(self,fig, predictions):
		"""
		Plots decoded predictions
		"""
		for i in range(1,17):
			fig.add_subplot(4,4,i)
			plt.imshow(predictions[i])
			plt.axis('off')
		plt.pause(0.01)

	def train_model(self):
		"""
		Autoencoder
		"""
		fig=plt.figure(figsize=(8, 8))
		x = tf.placeholder(dtype = tf.float32, shape = [None, 256, 256 ,3])
		xflat = tf.contrib.layers.flatten(x)
		latent_space = tf.contrib.layers.fully_connected(xflat, 20)
		logits = tf.contrib.layers.fully_connected(latent_space, 256*256*3)
		logits = tf.reshape(logits, [-1, 256, 256, 3])
		loss = tf.losses.mean_squared_error(x, logits)
		train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			for i in range(80):
				_, im, l = sess.run([train_op, logits, loss], feed_dict = {x: self.cage})
				print('loss:',l)
				self.plot_progress(fig,im)

if __name__ == '__main__':
	a = Deep()
	a.train_model()
	pass
