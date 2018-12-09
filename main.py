import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np
import skimage
from tqdm import tqdm
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

class Deep:

    def __init__(self, batch_size = 16, epochs = 10):
        self.batch_size = batch_size
        self.epochs = epochs
        self.dataset = self.make_tf_dataset()
        self.create_model()
        self.saver = tf.train.Saver()

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
        latent_space_1 = tf.contrib.layers.fully_connected(xflat, 60)
        latent_space_2 = tf.contrib.layers.fully_connected(latent_space_1, 60)
        latent_space = tf.contrib.layers.fully_connected(latent_space_2, 60)
        logits = tf.contrib.layers.fully_connected(latent_space, 256*256*3)
        logits = tf.reshape(logits, [-1, 256, 256, 3])
        self.predictions = logits
        self.loss = tf.losses.mean_squared_error(x, logits)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.loss)

    def train(self, save_model = False):
        fig=plt.figure(figsize=(6, 6))
        iterations = self.epochs*self.dataset.shape[0]
        with tf.Session() as sess, tqdm(total = iterations) as pbar:
            sess.run(tf.global_variables_initializer())
            for epochs in range(self.epochs):
                sess.run(self.iterator.initializer, feed_dict = {self.place_holder:self.dataset})
                try:
                    while True:
                        loss, im, _ = sess.run([self.loss, self.predictions, self.optimizer])
                        print(loss)
                        pbar.update(self.batch_size)
                except tf.errors.OutOfRangeError:
                    pass
                self.plot_progress(fig,im)
            if save_model:
                self.saver.save(sess, './one')

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
        for i in range(1,predictions.shape[0]):
            fig.add_subplot(4,4,i)
            plt.imshow(predictions[i-1])
            plt.axis('off')
        plt.pause(0.01)

    def show_saved_variables(self, filename):
        print_tensors_in_checkpoint_file(file_name=filename, tensor_name='', all_tensors = False)        

if __name__ == '__main__':
    a = Deep()
    a.train()


