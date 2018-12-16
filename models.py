import tensorflow as tf
import numpy as np

class Model:
	def __init__(self):
		pass

	@staticmethod
	def encoder(image):
		"""
		Encoder model
		"""
		with tf.name_scope('Encoder'):
			with tf.name_scope('Input_Image'):
				images = tf.placeholder(tf.float32, [None, 64, 64, 3])
			
			with tf.name_scope('Convolutions'):
				conv1 = tf.layers.conv2d(images, 64, (3,3), strides = (1,1), padding = 'same', activation = 'relu')
				conv2 = tf.layers.conv2d(conv1, 128, (3,3), strides = (2,2), padding = 'same', activation = 'relu')
				conv3 = tf.layers.conv2d(conv2, 256, (3,3), strides = (2,2), padding = 'same', activation = 'relu')
				conv4 = tf.layers.conv2d(conv3, 512, (3,3), strides = (2,2), padding = 'same', activation = 'relu')
				conv5 = tf.layers.conv2d(conv4, 1024, (3,3), strides = (2,2), padding = 'same', activation = 'relu')

			with tf.name_scope('Fully_Connected'):
				flat1 = tf.contrib.layers.flatten(conv5)
				flat2 = tf.contrib.layers.fully_connected(flat1, 1024)
				flat3 = tf.contrib.layers.fully_connected(flat2, 4*4*1024)

			with tf.name_scope('Upscaling'):
				up1 = tf.reshape(flat3, [-1, 4, 4, 1024])
				up2 = tf.layers.conv2d(up1, 512, (3,3), strides = (1,1), padding = 'same', activation = 'relu')
				up2 = tf.image.resize_images(up2, size=(8,8), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
			
		return up2

	@staticmethod
	def decoder(encoded_images):
		"""
		Decoder Model
		"""
		with tf.name_scope('Decoder'):
			with tf.name_scope('Upscaling'):
				up1 = tf.layers.conv2d(encoded_images, 256, (4,4), strides = (1,1), padding = 'same')
				up1 = tf.nn.leaky_relu(up1)
				up1 = tf.image.resize_images(up1, size=(16,16), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
				
				up2 = tf.layers.conv2d(up1, 128, (4,4), strides = (1,1), padding = 'same')
				up2 = tf.nn.leaky_relu(up2)
				up2 = tf.image.resize_images(up2, size=(32,32), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
			
				up3 = tf.layers.conv2d(up2, 64, (4,4), strides = (1,1), padding = 'same')
				up3 = tf.nn.leaky_relu(up3)
				up3 = tf.image.resize_images(up3, size=(64,64), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
			
			with tf.name_scope('Convolutions'):
				conv1 = tf.layers.conv2d(up3, 64, (3,3), strides = (1,1), padding = 'same')
				conv1 = tf.nn.leaky_relu(conv1)
				conv2 = tf.layers.conv2d(conv1, 64, (3,3), strides = (1,1), padding = 'same')
				conv2 = conv2 + up3

			with tf.name_scope('Masks'):
				m1 = tf.layers.conv2d(conv2, 3, (5,5), strides = (1,1), padding = 'same', activation = 'tanh')
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			writer = tf.summary.FileWriter("graph", sess.graph)
			writer.close()










a = Model()
encoded_images = a.encoder(1)
a.decoder(encoded_images)
