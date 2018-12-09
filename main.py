import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np
import skimage
from tqdm import tqdm

class Deep:

    def __init__(self, batch_size = 16, epochs = 20):
        self.batch_size = batch_size
        self.epochs = epochs
        self.dataset = self.make_tf_dataset()
        self.model()

    def model(self):
        """
        Autoencoder model
        """
        #iterator = self.dataset.make_one_shot_iterator()
        
        self.place_holder = tf.placeholder(tf.float32, [None, 256, 256, 3])
        dataset = tf.data.Dataset.from_tensor_slices(self.place_holder).batch(self.batch_size)
        self.iterator = dataset.make_initializable_iterator()
        x = self.iterator.get_next()
        xflat = tf.contrib.layers.flatten(x)
        latent_space = tf.contrib.layers.fully_connected(xflat, 20)
        logits = tf.contrib.layers.fully_connected(latent_space, 256*256*3)
        logits = tf.reshape(logits, [-1, 256, 256, 3])
        self.predictions = logits
        self.loss = tf.losses.mean_squared_error(x, logits)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.loss)

    def train(self):
        fig=plt.figure(figsize=(6, 6))
        iterations = self.epochs*self.dataset.shape[0]
        with tf.Session() as sess, tqdm(total = iterations) as pbar:
            sess.run(tf.global_variables_initializer())
            for epochs in range(self.epochs):
                sess.run(self.iterator.initializer, feed_dict = {self.place_holder:self.dataset})
                try:
                    while True:
                        loss, im, _ = sess.run([self.loss, self.predictions, self.optimizer])
                        self.plot_progress(fig,im)
                        pbar.update(self.batch_size)
                except tf.errors.OutOfRangeError:
                    pass

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

if __name__ == '__main__':
    a = Deep()
    a.train()

