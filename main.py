import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np
import skimage
from tqdm import tqdm
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

class Deep:

    def __init__(self, batch_size = 16, epochs = 5):
        self.batch_size = batch_size
        self.epochs = epochs
        self.dataset = self.make_tf_dataset()
        #self.create_cnn_model()
        #self.saver = tf.train.Saver()

    def decoder_model(self, name):
        """
        Unique decoder model.
        One is made for each dataset.
        """
        #self.create_latent_tensors('./model/encoder')
        self.decoder = {}
        name = name + '_'
        X_train = tf.placeholder(tf.float32, [None, 32, 32, 5])
        self.decoder[name + 'X_train'] = X_train
        y_train = tf.placeholder(tf.float32, [None, 256, 256, 3])
        self.decoder[name + 'y_train'] = y_train


        dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(self.batch_size)
        iterator = dataset.make_initializable_iterator()
        self.decoder[name + 'iterator'] = iterator
        latent_space, true_images = iterator.get_next()
        up_1_conv = tf.layers.conv2d(latent_space, 10, (5,5), strides = (1,1), padding = 'same', activation = 'relu')
        up_1 = tf.image.resize_images(up_1_conv, size=(64,64), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        up_2_conv = tf.layers.conv2d(up_1, 25, (5,5), strides = (1,1), padding = 'same', activation = 'relu')
        up_2 = tf.image.resize_images(up_2_conv, size=(128,128), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        predictions = tf.layers.conv2d(up_2, 3, (5,5), strides = (1,1), padding = 'same')
        predictions = tf.image.resize_images(predictions, size=(256,256), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        self.decoder[name + 'predictions'] = predictions
        loss = tf.losses.mean_squared_error(true_images, predictions)
        self.decoder[name + 'loss'] = loss
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
        self.decoder[name + 'optimizer'] = optimizer

        #self.decoder = {name + 'X_train':X_train, name + 'y_train':y_train, name + 'predictions': predictions,
        #                name + 'loss': loss, name + 'optimizer': optimizer, name + 'iterator': iterator}

    def train_decoders(self):
        """
        Trains both Trump's and Cage's decoders.
        """
        self.decoder_model('trump')
        print(self.decoder)
        #self.decoder_model('cage')
        trump = self.load_data('trump')
        #cage = self.load_data('cage')
        fig=plt.figure(figsize=(6, 6))
        iterations = self.epochs*trump.shape[0]
        with tf.Session() as sess, tqdm(total = iterations) as pbar:
            sess.run(tf.global_variables_initializer())
            for epochs in range(self.epochs):
                sess.run(self.decoder['trump_iterator'].initializer,
                feed_dict = {self.decoder['trump_X_train']:self.latent_trump, self.decoder['trump_y_train']: trump})
                try:
                    while True:
                        _,loss, im = sess.run([self.decoder['trump_optimizer'],self.decoder['trump_loss'], self.decoder['trump_predictions']])
                        pbar.update(self.batch_size)
                except tf.errors.OutOfRangeError:
                    pass
                
                self.plot_progress(fig,im)
            saver = tf.train.Saver()
            saver.save(sess, './decoder/trump_decoder')
        #    if save_model:
        #        self.saver.save(sess, './model/encoder')
    

    def plot_deep_fakes(self, decoder_model_file = './decoder/trump_decoder'):
        """
        Replace Cage face with Trump face.
        """

        fig=plt.figure(figsize=(6, 6))
        #fig1=plt.figure(figsize=(6, 6))
        cage = self.load_data('cage')
        #trump = self.load_data('trump')
        #tf.reset_default_graph()
        #tf.reset_default_graph()
        self.decoder_model('trump')
        var_name_list = [v.name for v in tf.trainable_variables()]
        #print(var_name_list)
        #print(tf.get_default_graph().as_graph_def())
        with tf.Session() as sess:
            #sess.run(tf.global_variables_initializer())
            #new_saver = tf.train.import_meta_graph('decoder/trump_decoder.meta',clear_devices=True)
            #new_saver.restore(sess, 'my-save-dir/my-model-10000')
            tf_saver = tf.train.Saver()
            #tf.reset_default_graph()
            tf_saver.restore(sess, decoder_model_file)
            sess.run(self.decoder['trump_iterator'].initializer,
                feed_dict = {self.decoder['trump_X_train']:self.latent_cage, self.decoder['trump_y_train']: cage})
            deep_fakes = sess.run(self.decoder['trump_predictions'])
        print(deep_fakes.shape)
        #self.plot_progress(fig1, cage[0:16])
        self.plot_progress(fig, deep_fakes)




    def create_latent_tensors(self, model_file_name):
        """
        Obtain latent space tensors used for training
        the different decoders.
        """
        trump = self.load_data('trump')
        cage = self.load_data('cage')
        latent_trump = np.empty((0,32,32,5))
        latent_cage = np.empty((0,32,32,5))
        self.create_cnn_model()
        with tf.Session() as sess:
            tf_saver = tf.train.Saver()
            tf_saver.restore(sess, './model/encoder')
            sess.run(self.cnn['iterator'].initializer, feed_dict = {self.cnn['place_holder']:trump})
            try:
                while True:
                    latent = sess.run([self.cnn['latent_space']])
                    latent_trump = np.vstack((latent_trump,latent[0]))
            except tf.errors.OutOfRangeError:
                pass
            sess.run(self.cnn['iterator'].initializer, feed_dict = {self.cnn['place_holder']:cage})
            
            try:
                while True:
                    latent = sess.run([self.cnn['latent_space']])
                    latent_cage = np.vstack((latent_cage,latent[0]))
            except tf.errors.OutOfRangeError:
                pass

        self.latent_trump = latent_trump
        self.latent_cage = latent_cage

    def create_cnn_model(self):
        """
        Convolutional encoder model.
        """ 
        place_holder = tf.placeholder(tf.float32, [None, 256, 256, 3])
        dataset = tf.data.Dataset.from_tensor_slices(place_holder).batch(self.batch_size)
        iterator = dataset.make_initializable_iterator()
        x = iterator.get_next()
        down_1 = tf.layers.conv2d(x, 15, (5,5), strides = (2,2), padding = 'same', activation = 'relu')
        down_2 = tf.layers.conv2d(down_1, 10, (5,5), strides = (2,2), padding = 'same', activation = 'relu')
        latent_space = tf.layers.conv2d(down_2, 5, (5,5), strides = (2,2), padding = 'same', activation = 'relu')
        up_1_conv = tf.layers.conv2d(latent_space, 10, (5,5), strides = (1,1), padding = 'same', activation = 'relu')
        up_1 = tf.image.resize_images(up_1_conv, size=(64,64), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        up_2_conv = tf.layers.conv2d(up_1, 25, (5,5), strides = (1,1), padding = 'same', activation = 'relu')
        up_2 = tf.image.resize_images(up_2_conv, size=(128,128), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        logits = tf.layers.conv2d(up_2, 3, (5,5), strides = (1,1), padding = 'same')
        logits = tf.image.resize_images(logits, size=(256,256), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        loss = tf.losses.mean_squared_error(x, logits)
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
        self.cnn = {'place_holder': place_holder, 'iterator': iterator, 'latent_space': latent_space,
                    'predictions': logits, 'loss': loss, 'optimizer': optimizer}


    def train(self, save_model = True):
        """
        Encoder training. Encoder is shared by both datasets.
        """
        fig=plt.figure(figsize=(6, 6))
        iterations = self.epochs*self.dataset.shape[0]
        with tf.Session() as sess, tqdm(total = iterations) as pbar:
            sess.run(tf.global_variables_initializer())
            for epochs in range(self.epochs):
                sess.run(self.iterator.initializer, feed_dict = {self.cnn['place_holder']:self.dataset})
                try:
                    while True:
                        loss, im, _ = sess.run([self.cnn['loss'], self.cnn['predictions'], self.cnn['optimizer']])
                        #im_cage = sess.run([])
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

    def plot_progress(self,fig, predictions):
        """
        Plots decoded predictions
        """
        predictions = predictions/np.max(predictions)
        for i in range(1,predictions.shape[0]):
            fig.add_subplot(4,4,i)
            plt.imshow(predictions[i-1])
            plt.axis('off')
        plt.show()
        #plt.pause(0.01)

    def show_saved_variables(self, filename):
        print_tensors_in_checkpoint_file(file_name=filename, tensor_name='', all_tensors = False)        

if __name__ == '__main__':
    a = Deep()
    #a.show_saved_variables('./model/trump_decoder')
    a.create_latent_tensors('./model/encoder')
    tf.reset_default_graph()
    #tf.reset_default_graph()
    #a.train_decoders()
    a.plot_deep_fakes()
    #a.decoder_model('trump')
    #print(a.decoder)
    #var_name_list = [v.name for v in tf.trainable_variables()]
    #print(var_name_list)

