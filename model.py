import tensorflow as tf
from keras.datasets import mnist
import numpy as np
import matplotlib.gridspec as gridspec
import os
import matplotlib.pyplot as plt
from PIL import Image


class Model:
    def __init__(self):
        self.x = tf.placeholder(tf.float32, shape=(None, 784))
        self.z = tf.placeholder(tf.float32, shape=(None, 100))
        # self.D_W1 = tf.get_variable(name='D_W1', shape=[784, 128], initializer=tf.contrib.layers.xavier_initializer())
        # self.D_b1 = tf.get_variable(name='D_b1', shape=[128], initializer=tf.constant_initializer(0))
        #
        # self.D_W2 = tf.get_variable(name='D_W2', shape=[128, 1], initializer=tf.contrib.layers.xavier_initializer())
        # self.D_b2 = tf.get_variable(name='D_b2', shape=[1], initializer=tf.constant_initializer(0))
        #
        # self.G_W1 = tf.get_variable(name='G_W1', shape=[100, 128], initializer=tf.contrib.layers.xavier_initializer())
        # self.G_b1 = tf.get_variable(name='G_b1', shape=[128], initializer=tf.constant_initializer(0))
        #
        # self.G_W2 = tf.get_variable(name='G_W2', shape=[128, 784], initializer=tf.contrib.layers.xavier_initializer())
        # self.G_b2 = tf.get_variable(name='G_b2', shape=[784], initializer=tf.constant_initializer(0))
        # self.theta_D = [self.D_W1, self.D_W2, self.D_b1, self.D_b2]
        # self.theta_G = [self.G_W1, self.G_W2, self.G_b1, self.G_b2]
        self.is_training = tf.placeholder(dtype=tf.bool)
        self.sample_image = self.generator(self.z)
        self.d_x_logit, self.d_x_pre = self.discriminator(self.x, reuse=False)
        self.d_z_logit, self.d_z_pre = self.discriminator(self.sample_image, reuse=True)
        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_z_logit,
                                                                             labels=tf.ones_like(self.d_z_logit)))
        self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_x_logit,
                                                                                  labels=tf.ones_like(self.d_x_logit)))
        self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_z_logit,
                                                                                  labels=tf.zeros_like(self.d_z_logit)))
        self.d_loss = self.d_loss_real + self.d_loss_fake
        self.tvars = tf.trainable_variables()
        for t in self.tvars:
            print(t.name)
        self.d_vars = [var for var in self.tvars if 'dis' in var.name]
        self.g_vars = [var for var in self.tvars if 'gen' in var.name]

        self.trainerD = tf.train.AdamOptimizer().minimize(self.d_loss, var_list=self.d_vars)
        self.trainerG = tf.train.AdamOptimizer().minimize(self.g_loss, var_list=self.g_vars)

    def generator(self, z):
        with tf.variable_scope('gen'), tf.name_scope('gen'):
            d1 = tf.layers.dense(inputs=z, units=128, activation=tf.nn.relu,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer())
            # d1 = tf.nn.relu(tf.matmul(z, self.G_W1)+self.G_b1)
            # d1 = tf.layers.batch_normalization(d1, training=self.is_training)
            # d2 = tf.layers.dense(inputs=d1, units=254, activation=tf.nn.relu)
            # d3 = tf.layers.dense(inputs=d2, units=512, activation=tf.nn.relu)
            output = tf.layers.dense(inputs=d1, units=784, activation=tf.nn.sigmoid,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer())
            # output = tf.nn.sigmoid(tf.matmul(d1, self.G_W2)+self.G_b2)
            return output

    def discriminator(self, inputs, reuse):
        with tf.variable_scope('dis', reuse=reuse):
            # d1 = tf.layers.dense(inputs=inputs, units=512, activation=tf.nn.relu)
            # d2 = tf.layers.dense(inputs=d1, units=256, activation=tf.nn.relu)
            d3 = tf.layers.dense(inputs=inputs, units=128, activation=tf.nn.leaky_relu,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer())
            # d3 = tf.nn.relu(tf.matmul(inputs, self.D_W1)+self.D_b1)
            # d_logit = tf.matmul(d3, self.D_W2)+self.D_b2
            d_logit = tf.layers.dense(inputs=d3, units=1,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer())
            d_pre = tf.nn.sigmoid(d_logit)
            return d_logit, d_pre

    def plot(self, samples):
        fig = plt.figure(figsize=(4, 4))
        gs = gridspec.GridSpec(4, 4)
        gs.update(wspace=0.05, hspace=0.05)

        for i, sample in enumerate(samples):  # [i,samples[i]] imax=16
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_aspect('equal')
            plt.imshow(sample.reshape(28, 28), cmap='Greys_r')
        return fig

    def train(self, batch_size, epoch):
        x_train = self.load_mnist()
        path = './img'
        if not os.path.exists(path):
            os.makedirs(path)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(epoch):
                x_batch = x_train[np.random.randint(0, x_train.shape[0], size=[batch_size])]
                z_batch = np.random.uniform(-1, 1, [batch_size, 100])
                if i % 1000 == 0:
                    test_batch = np.random.uniform(-1, 1, [16, 100])
                    g_img, t_vars = sess.run([self.sample_image, self.tvars], feed_dict={self.z: test_batch,
                                                                                          self.is_training: False})

                    fig = self.plot(g_img)
                    plt.savefig('img/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
                    plt.close(fig)

                _, dloss = sess.run([self.trainerD, self.d_loss], feed_dict={self.x: x_batch,
                                                                             self.z: z_batch,
                                                                             self.is_training: True})
                _, gloss = sess.run([self.trainerG, self.g_loss], feed_dict={self.z: z_batch,
                                                                             self.is_training: True})

                if i % 1000 == 0:
                    print('epoch:{} dloss:{} gloss:{}'.format(i, dloss, gloss))

    def load_mnist(self):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.reshape(60000, 784)
        x_train = x_train.astype('float32')
        x_train /= 255
        return x_train


if __name__ == '__main__':
    m = Model()
    m.train(128, 1000000)


