import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np
import skimage
from tqdm import tqdm
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

class Deep:

    def __init__(self, batch_size = 16, epochs = 7):
        self.batch_size = batch_size
        self.epochs = epochs
        self.dataset = self.make_tf_dataset()
        self.create_cnn_model()
        self.saver = tf.train.Saver()


    def create_latent_tensors(self, model_file_name):
        """
        Obtain latent space tensors used for training
        the different decoders.
        """
        trump = self.load_data('trump')
        cage = self.load_data('cage')
        latent_trump = np.empty((0,32,32,5))
        latent_cage = np.empty((0,32,32,5))
        with tf.Session() as sess:
            tf_saver = tf.train.Saver()
            tf_saver.restore(sess, './model/encoder')
            sess.run(self.iterator.initializer, feed_dict = {self.place_holder:trump})
            try:
                while True:
                    latent = sess.run([self.latent_space])
                    latent_trump = np.vstack((latent_trump,latent[0]))
            except tf.errors.OutOfRangeError:
                pass

            sess.run(self.iterator.initializer, feed_dict = {self.place_holder:cage})
            try:
                while True:
                    latent = sess.run([self.latent_space])
                    latent_trump = np.vstack((latent_cage,latent[0]))
            except tf.errors.OutOfRangeError:
                pass

            
        self.latent_trump = latent_trump
        self.latent_cage = latent_cage

    def create_model(self):
        """
        Autoencoder model
        With initializable iterator.
        No need to dataset.repeat(no_epochs) just initialize to 
        beginning of dataset every epoch.
        """
        self.place_holder = tf.placeholder(tf.float32, [None, 256, 256, 3])
        dataset = tf.data.Dataset.from_tensor_slices(self.place_holder).batch(self.batch_size)
        self.iterator = dataset.make_initializable_iterator()
        x = self.iterator.get_next()
        xflat = tf.contrib.layers.flatten(x)
        latent_space_1 = tf.layers.dense(xflat, 90, activation = 'relu')
        latent_space_2 = tf.layers.dense(latent_space_1, 60, activation = 'relu')
        latent_space = tf.layers.dense(latent_space_2, 90, activation = 'relu')
        logits = tf.layers.dense(latent_space, 256*256*3)
        logits = tf.reshape(logits, [-1, 256, 256, 3])
        self.predictions = logits
        self.loss = tf.losses.mean_squared_error(x, logits)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.loss)

    def create_cnn_model(self):
        self.place_holder = tf.placeholder(tf.float32, [None, 256, 256, 3])
        dataset = tf.data.Dataset.from_tensor_slices(self.place_holder).batch(self.batch_size)
        self.iterator = dataset.make_initializable_iterator()
        x = self.iterator.get_next()
        down_1 = tf.layers.conv2d(x, 15, (5,5), strides = (2,2), padding = 'same', activation = 'relu')
        down_2 = tf.layers.conv2d(down_1, 10, (5,5), strides = (2,2), padding = 'same', activation = 'relu')
        latent_space = tf.layers.conv2d(down_2, 5, (5,5), strides = (2,2), padding = 'same', activation = 'relu')
        self.latent_space = latent_space
        up_1_conv = tf.layers.conv2d(latent_space, 10, (5,5), strides = (1,1), padding = 'same', activation = 'relu')
        up_1 = tf.image.resize_images(up_1_conv, size=(64,64), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        up_2_conv = tf.layers.conv2d(up_1, 25, (5,5), strides = (1,1), padding = 'same', activation = 'relu')
        up_2 = tf.image.resize_images(up_2_conv, size=(128,128), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        logits = tf.layers.conv2d(up_2, 3, (5,5), strides = (1,1), padding = 'same')
        logits = tf.image.resize_images(logits, size=(256,256), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        self.predictions = logits
        self.loss = tf.losses.mean_squared_error(x, logits)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.loss)


    def train(self, save_model = True):
        fig=plt.figure(figsize=(6, 6))
        iterations = self.epochs*self.dataset.shape[0]
        with tf.Session() as sess, tqdm(total = iterations) as pbar:
            sess.run(tf.global_variables_initializer())
            for epochs in range(self.epochs):
                sess.run(self.iterator.initializer, feed_dict = {self.place_holder:self.dataset})
                try:
                    while True:
                        loss, im, _ = sess.run([self.loss, self.predictions, self.optimizer])
                        #print(loss)
                        pbar.update(self.batch_size)
                except tf.errors.OutOfRangeError:
                    pass
                self.plot_progress(fig,im)
            if save_model:
                self.saver.save(sess, './model/encoder')

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
        return np.array(images).astype('float32')

    def make_tf_dataset(self):
        trump_images = self.load_data('trump')
        cage_images = self.load_data('cage')
        all_images = np.vstack((trump_images, cage_images))
        np.random.shuffle(all_images)
        return all_images
        #return tf.data.Dataset.from_tensor_slices(all_images).repeat(self.epochs).batch(self.batch_size)

    def plot_progress(self,fig, predictions):
        """
        Plots decoded predictions
        """
        predictions = predictions/np.max(predictions)
        for i in range(1,predictions.shape[0]):
            fig.add_subplot(4,4,i)
            plt.imshow(predictions[i-1])
            plt.axis('off')
        plt.pause(0.01)

    def show_saved_variables(self, filename):
        print_tensors_in_checkpoint_file(file_name=filename, tensor_name='', all_tensors = False)        


a = Deep()
#a.show_saved_variables('./model/encoder')
a.create_latent_tensors('./model/encoder')
"""
with tf.Session() as sess:
    tf_saver = tf.train.Saver()
    tf_saver.restore(sess, './model/encoder')
    sess.run(a.iterator.initializer, feed_dict = {a.place_holder:a.dataset})
    loss, im, _ = sess.run([a.loss, a.predictions, a.optimizer])
    plt.imshow(im[0])
    plt.show()
"""
"""
if __name__ == '__main__':
    a = Deep()
    a.train()
"""

