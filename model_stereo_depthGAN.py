# Code for
# Unsupervised Adversarial Depth Estimation using Cycled Generative Networks
# Andrea Pilzer, Dan Xu, Mihai Puscas, Elisa Ricci, Nicu Sebe
#
# 3DV 2018 Conference, Verona, Italy
#
# parts of the code from https://github.com/mrharicot/monodepth
#

from __future__ import absolute_import, division, print_function
from collections import namedtuple

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from bilinear_sampler import *
from module import *

monodepth_parameters = namedtuple('parameters',
                        'encoder, '
                        'height, width, '
                        'batch_size, '
                        'num_threads, '
                        'num_epochs, '
                        'do_stereo, '
                        'wrap_mode, '
                        'use_deconv, '
                        'lr_loss_weight, '
                        'full_summary, '
                        'num_gpus')

class stereo_depthGAN_Model(object):
    """unsupervised stereo depthGAN model"""

    def __init__(self, params, branch, mode, left, right, reuse_variables=None, model_index=0):
        self.params = params
        self.branch = branch
        self.mode = mode
        self.left = left
        self.right = right
        self.model_collection = ['model_' + str(model_index)]

        self.batch_size = params.batch_size/params.num_gpus
        self.size_W = params.width
        self.size_H = params.height
        self.input_c_dim = 3
        self.output_c_dim = 1
        self.identity = tf.Variable(tf.ones([self.batch_size, self.size_H, self.size_W, 1]), trainable=False)

        self.reuse_variables = reuse_variables
        self.reuse = reuse_variables
        self.discriminator = self.discr
        self.encoder = self.build_resnet50_enc
        self.decoder = self.build_resnet50_dec
        self.fusion = self.fusion_func

        self.criterionGAN = mae_criterion

        OPTIONS = namedtuple('OPTIONS', 'batch_size image_size \
                              gf_dim df_dim output_c_dim is_training')
        self.options = OPTIONS._make((params.batch_size, params.height,
                                      64, 64, self.output_c_dim,
                                      mode == 'train'))

        self.build()
        #self.build_outputs()

        if self.mode == 'test':
            self.disp_out = self.disp_output()
            return

        self.build_losses()
        self.build_summaries()

    def gradient_x(self, img):
        gx = img[:,:,:-1,:] - img[:,:,1:,:]
        return gx

    def gradient_y(self, img):
        gy = img[:,:-1,:,:] - img[:,1:,:,:]
        return gy

    def upsample_nn(self, x, ratio):
        s = x.get_shape().as_list()
        h = s[1]
        w = s[2]
        return tf.image.resize_nearest_neighbor(x, [h * ratio, w * ratio])

    def scale_pyramid(self, img, num_scales):
        scaled_imgs = [img]
        s = tf.shape(img)
        h = s[1]
        w = s[2]
        for i in range(num_scales - 1):
            ratio = 2 ** (i + 1)
            nh = h // ratio
            nw = w // ratio
            scaled_imgs.append(tf.image.resize_area(img, [nh, nw]))
        return scaled_imgs

    def generate_image_left(self, img, disp):
        return bilinear_sampler_1d_h(img, -disp)

    def generate_image_right(self, img, disp):
        return bilinear_sampler_1d_h(img, disp)

    def get_disp_original(self, x):
        disp = 0.3 * self.conv(x, 2, 3, 1, tf.nn.sigmoid)
        return disp

    def get_disp(self, x):
        disp = 0.3 * self.conv(x, 1, 3, 1, tf.nn.sigmoid)
        return disp

    def conv(self, x, num_out_layers, kernel_size, stride, activation_fn=tf.nn.elu):
        p = np.floor((kernel_size - 1) / 2).astype(np.int32)
        p_x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]])
        return slim.conv2d(p_x, num_out_layers, kernel_size, stride, 'VALID', activation_fn=activation_fn)

    def conv_block(self, x, num_out_layers, kernel_size):
        conv1 = self.conv(x,     num_out_layers, kernel_size, 1)
        conv2 = self.conv(conv1, num_out_layers, kernel_size, 2)
        return conv2

    def maxpool(self, x, kernel_size):
        p = np.floor((kernel_size - 1) / 2).astype(np.int32)
        p_x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]])
        return slim.max_pool2d(p_x, kernel_size)

    def resconv(self, x, num_layers, stride):
        do_proj = tf.shape(x)[3] != num_layers or stride == 2
        shortcut = []
        conv1 = self.conv(x,         num_layers, 1, 1)
        conv2 = self.conv(conv1,     num_layers, 3, stride)
        conv3 = self.conv(conv2, 4 * num_layers, 1, 1, None)
        if do_proj:
            shortcut = self.conv(x, 4 * num_layers, 1, stride, None)
        else:
            shortcut = x
        return tf.nn.elu(conv3 + shortcut)

    def resblock(self, x, num_layers, num_blocks):
        out = x
        for i in range(num_blocks - 1):
            out = self.resconv(out, num_layers, 1)
        out = self.resconv(out, num_layers, 2)
        return out

    def upconv(self, x, num_out_layers, kernel_size, scale):
        upsample = self.upsample_nn(x, scale)
        conv = self.conv(upsample, num_out_layers, kernel_size, 1)
        return conv

    def deconv(self, x, num_out_layers, kernel_size, scale):
        p_x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]])
        conv = slim.conv2d_transpose(p_x, num_out_layers, kernel_size, scale, 'SAME')
        return conv[:,3:-1,3:-1,:]

    def build_resnet50_enc(self, net_input, name, direction='', reuse=False):
        with tf.variable_scope(name):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            #set convenience functions
            conv   = self.conv
            if self.params.use_deconv:
                upconv = self.deconv
            else:
                upconv = self.upconv

            with tf.variable_scope('encoder'):
                conv1 = conv(net_input, 64, 7, 2) # H/2  -   64D
                pool1 = self.maxpool(conv1,           3) # H/4  -   64D
                conv2 = self.resblock(pool1,      64, 3) # H/8  -  256D
                conv3 = self.resblock(conv2,     128, 4) # H/16 -  512D
                conv4 = self.resblock(conv3,     256, 6) # H/32 - 1024D
                conv5 = self.resblock(conv4,     512, 3) # H/64 - 2048D

            with tf.variable_scope('skips'):
                skip1 = conv1
                skip2 = pool1
                skip3 = conv2
                skip4 = conv3
                skip5 = conv4

            return skip1, skip2, skip3, skip4, skip5, conv5

    def build_resnet50_dec(self, skip1, skip2, skip3, skip4, skip5, conv5, name, direction='', reuse=False):
        with tf.variable_scope(name):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            #set convenience functions
            conv   = self.conv
            if self.params.use_deconv:
                upconv = self.deconv
            else:
                upconv = self.upconv
            # DECODING
            with tf.variable_scope('decoder'):
                upconv6 = upconv(conv5,   512, 3, 2) #H/32
                concat6 = tf.concat([upconv6, skip5], 3)
                iconv6  = conv(concat6,   512, 3, 1)

                upconv5 = upconv(iconv6, 256, 3, 2) #H/16
                concat5 = tf.concat([upconv5, skip4], 3)
                iconv5  = conv(concat5,   256, 3, 1)

                upconv4 = upconv(iconv5,  128, 3, 2) #H/8
                concat4 = tf.concat([upconv4, skip3], 3)
                iconv4  = conv(concat4,   128, 3, 1)
                disp4 = self.get_disp(iconv4)
                udisp4  = self.upsample_nn(disp4, 2)

                upconv3 = upconv(iconv4,   64, 3, 2) #H/4
                concat3 = tf.concat([upconv3, skip2, udisp4], 3)
                iconv3  = conv(concat3,    64, 3, 1)
                disp3 = self.get_disp(iconv3)
                udisp3  = self.upsample_nn(disp3, 2)

                upconv2 = upconv(iconv3,   32, 3, 2) #H/2
                concat2 = tf.concat([upconv2, skip1, udisp3], 3)
                iconv2  = conv(concat2,    32, 3, 1)
                disp2 = self.get_disp(iconv2)
                udisp2  = self.upsample_nn(disp2, 2)

                upconv1 = upconv(iconv2,  16, 3, 2) #H
                concat1 = tf.concat([upconv1, udisp2], 3)
                iconv1  = conv(concat1,   16, 3, 1)
                disp1 = self.get_disp(iconv1)

            return disp4, disp3, disp2, disp1

    def discr(self,image, options, reuse=False, name="discriminator"):
        with tf.variable_scope(name):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            elif reuse == False and self.reuse == True:
                tf.get_variable_scope().reuse_variables()
            elif reuse == True and self.reuse == False:
                tf.get_variable_scope().reuse_variables()
            elif reuse == False and self.reuse == False:
                assert tf.get_variable_scope().reuse is False

            h0 = lrelu(conv2d(image, options.df_dim, name='d_h0_conv'))
            # h0 is (128 x 128 x self.df_dim)
            h1 = lrelu(instance_norm(conv2d(h0, options.df_dim*2, name='d_h1_conv'), 'd_bn1'))
            # h1 is (64 x 64 x self.df_dim*2)
            h2 = lrelu(instance_norm(conv2d(h1, options.df_dim*4, name='d_h2_conv'), 'd_bn2'))
            # h2 is (32x 32 x self.df_dim*4)
            h3 = lrelu(instance_norm(conv2d(h2, options.df_dim*8, s=1, name='d_h3_conv'), 'd_bn3'))
            # h3 is (32 x 32 x self.df_dim*8)
            h4 = conv2d(h3, 1, s=1, name='d_h3_pred')
            # h4 is (32 x 32 x 1)
            return h4

    def fusion_func(self, net_input1, net_input2, name, reuse=False):
        with tf.variable_scope(name):
            conv = self.conv
            if reuse:
                tf.get_variable_scope().reuse_variables()
            input_fusion = tf.concat([net_input1, net_input2], 3)
            conv_fusion = conv(input_fusion, 1, 1, 1, tf.nn.relu)
            return conv_fusion

    def build(self):
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose], activation_fn=tf.nn.elu):
            with tf.variable_scope('model', reuse=self.reuse_variables):

                self.left_pyramid  = self.scale_pyramid(self.left,  4)
                self.right_pyramid = self.scale_pyramid(self.right, 4)
                #right 2 left - forward cycle
                self.skip1_b2a, self.skip2_b2a, self.skip3_b2a, self.skip4_b2a, self.skip5_b2a, self.conv5_b2a = self.encoder(self.right, direction='right', name='enc', reuse=False)
                self.skip1_a2a, self.skip2_a2a, self.skip3_a2a, self.skip4_a2a, self.skip5_a2a, self.conv5_a2a = self.encoder(self.left,  direction='left_implicit', name='enc', reuse=True)

                self.disp4_b2a, self.disp3_b2a, self.disp2_b2a, self.disp1_b2a = self.decoder(self.skip1_b2a, self.skip2_b2a, self.skip3_b2a, self.skip4_b2a, self.skip5_b2a, self.conv5_b2a,  direction='right', name='decb2a', reuse=False)
                self.disp4_a2a, self.disp3_a2a, self.disp2_a2a, self.disp1_a2a = self.decoder(self.skip1_a2a, self.skip2_a2a, self.skip3_a2a, self.skip4_a2a, self.skip5_a2a, self.conv5_a2a,  direction='left_implicit', name='deca2a', reuse=False)

            with tf.variable_scope('disparities'):
                self.disp1_l = self.fusion(self.disp1_b2a, self.disp1_a2a, name='fusion2a', reuse=False)
                self.disp2_l = self.fusion(self.disp2_b2a, self.disp2_a2a, name='fusion2a', reuse=True)
                self.disp3_l = self.fusion(self.disp3_b2a, self.disp3_a2a, name='fusion2a', reuse=True)
                self.disp4_l = self.fusion(self.disp4_b2a, self.disp4_a2a, name='fusion2a', reuse=True)
                self.disp_left_est  = [self.disp1_l, self.disp2_l, self.disp3_l, self.disp4_l]

            # GENERATE IMAGES
            with tf.variable_scope('images'):
                self.left_est = [self.generate_image_left(self.right_pyramid[i], self.disp_left_est[i]) for i in range(4)]
                self.left_est[0].set_shape([self.batch_size, self.size_H, self.size_W, 3])

            with tf.variable_scope('model'):
                #left 2 right - backward cycle
                self.skip1_a2b, self.skip2_a2b, self.skip3_a2b, self.skip4_a2b, self.skip5_a2b, self.conv5_a2b = self.encoder(self.left_est[0],  direction='left', name='enc', reuse=True)
                self.skip1_b2b, self.skip2_b2b, self.skip3_b2b, self.skip4_b2b, self.skip5_b2b, self.conv5_b2b = self.encoder(self.right,  direction='right_implicit', name='enc', reuse=True)

                self.disp4_a2b, self.disp3_a2b, self.disp2_a2b, self.disp1_a2b = self.decoder(self.skip1_a2b, self.skip2_a2b, self.skip3_a2b, self.skip4_a2b, self.skip5_a2b, self.conv5_a2b,  direction='right', name='deca2b', reuse=False)
                self.disp4_b2b, self.disp3_b2b, self.disp2_b2b, self.disp1_b2b = self.decoder(self.skip1_b2b, self.skip2_b2b, self.skip3_b2b, self.skip4_b2b, self.skip5_b2b, self.conv5_b2b,  direction='left_implicit', name='decb2b', reuse=False)

            with tf.variable_scope('disparities'):
                self.disp1_r = self.fusion(self.disp1_a2b, self.disp1_b2b, name='fusion2b', reuse=False)
                self.disp2_r = self.fusion(self.disp2_a2b, self.disp2_b2b, name='fusion2b', reuse=True)
                self.disp3_r = self.fusion(self.disp3_a2b, self.disp3_b2b, name='fusion2b', reuse=True)
                self.disp4_r = self.fusion(self.disp4_a2b, self.disp4_b2b, name='fusion2b', reuse=True)
                self.disp_right_est  = [self.disp1_r, self.disp2_r, self.disp3_r, self.disp4_r]

            with tf.variable_scope('images'):
                self.right_est = [self.generate_image_right(self.left_pyramid[i], self.disp_right_est[i]) for i in range(4)]
                self.right_est[0].set_shape([self.batch_size, self.size_H, self.size_W, 3])

            with tf.variable_scope('model'):
                self.D_left_real = self.discriminator(self.left_pyramid[0], self.options, reuse=False, name="discriminatorB")
                self.D_left_fake = self.discriminator(self.left_est[0], self.options, reuse=True, name="discriminatorB")

                self.D_right_real = self.discriminator(self.right_pyramid[0], self.options, reuse=False, name="discriminatorA")
                self.D_right_fake = self.discriminator(self.right_est[0], self.options, reuse=True, name="discriminatorA")

        # LR CONSISTENCY
        with tf.variable_scope('left-right'):
            self.right_to_left_disp = [self.generate_image_left(self.disp_right_est[i], self.disp_left_est[i])  for i in range(4)]
            self.left_to_right_disp = [self.generate_image_right(self.disp_left_est[i], self.disp_right_est[i]) for i in range(4)]

        t_vars = tf.trainable_variables()
        self.discrA_vars = [var for var in t_vars if 'discriminatorA' in var.name]
        self.discrB_vars = [var for var in t_vars if 'discriminatorB' in var.name]
        self.encoder_vars = [var for var in t_vars if 'enc' in var.name]
        self.gen2b_vars = [var for var in t_vars if '2b' in var.name]
        self.gen2a_vars = [var for var in t_vars if '2a' in var.name]

    def disp_output(self):
        disp_out = tf.reduce_mean(tf.concat([self.disp_left_est[0], self.right_to_left_disp[0]], 3), 3, keep_dims=True)
        return disp_out

    def build_losses(self):
        with tf.variable_scope('losses', reuse=self.reuse_variables):
            # IMAGE RECONSTRUCTION
            # L1 (identity)
            self.l1_left = [tf.abs(self.left_est[i] - self.left_pyramid[i]) for i in range(4)]
            self.l1_reconstruction_loss_left  = [tf.reduce_mean(l) for l in self.l1_left]
            self.l1_right = [tf.abs(self.right_est[i] - self.right_pyramid[i]) for i in range(4)]
            self.l1_reconstruction_loss_right = [tf.reduce_mean(l) for l in self.l1_right]

            # L1 (cycle)
            self.l1_cycle_b = [tf.abs(self.right_est[i] - self.right_pyramid[i]) for i in range(4)]
            self.l1_cycle_backward = [tf.reduce_mean(l) for l in self.l1_cycle_b]
            if self.branch == 'a2b':
                self.cycle_loss = tf.add_n(self.l1_cycle_backward)
            elif self.branch == 'b2a':
                self.cycle_loss = 0
            else:
                self.cycle_loss = tf.add_n(self.l1_cycle_backward)

            #SUM
            self.image_loss_right = [self.l1_reconstruction_loss_right[i] for i in range(4)]
            self.image_loss_left  = [self.l1_reconstruction_loss_left[i]  for i in range(4)]
            #self.image_loss = tf.add_n(self.image_loss_left + self.image_loss_right)
            if self.branch == 'b2a':
                self.image_loss = tf.add_n(self.image_loss_left)
            elif self.branch == 'a2b':
                self.image_loss = tf.add_n(self.image_loss_right)
            else:
                self.image_loss = tf.add_n(self.image_loss_left + self.image_loss_right)

            # DISCRIMINATOR
            # left
            self.d_loss_left_real = self.criterionGAN(self.D_left_real, tf.ones_like(self.D_left_real))
            self.d_loss_left_fake = self.criterionGAN(self.D_left_fake, tf.zeros_like(self.D_left_fake))
            self.d_loss_left = (self.d_loss_left_real + self.d_loss_left_fake) / 2
            # right
            self.d_loss_right_real = self.criterionGAN(self.D_right_real, tf.ones_like(self.D_right_real))
            self.d_loss_right_fake = self.criterionGAN(self.D_right_fake, tf.zeros_like(self.D_right_fake))
            self.d_loss_right = (self.d_loss_right_real + self.d_loss_right_fake) / 2
            if self.branch == 'b2a':
                self.d_loss = self.d_loss_left
                self.d_loss_fake = self.d_loss_left_fake
            elif self.branch == 'a2b':
                self.d_loss = self.d_loss_right
                self.d_loss_fake = self.d_loss_right_fake
            else:
                self.d_loss = self.d_loss_left + self.d_loss_right
                self.d_loss_fake = self.d_loss_left_fake + self.d_loss_right_fake

            # LR CONSISTENCY
            self.lr_left_loss  = [tf.reduce_mean(tf.abs(self.right_to_left_disp[i] - self.disp_left_est[i]))  for i in range(4)]
            self.lr_right_loss = [tf.reduce_mean(tf.abs(self.left_to_right_disp[i] - self.disp_left_est[i])) for i in range(4)]
            if self.branch == 'b2a':
                self.lr_loss = 0
            elif self.branch == 'a2b':
                self.lr_loss = tf.add_n(self.lr_right_loss)
            else:
                self.lr_loss = tf.add_n(self.lr_right_loss +self.lr_left_loss)

            # # TOTAL LOSS
            self.total_loss = self.image_loss + 0.1 * self.cycle_loss - 0.0001 * self.d_loss_fake + 0.1 * self.lr_loss # + self.params.disp_gradient_loss_weight * self.disp_gradient_loss + self.params.lr_loss_weight * self.lr_loss
            self.discr_loss = 0.0001 * self.d_loss

    def build_summaries(self):
        # SUMMARIES
        with tf.device('/cpu:0'):

            for i in range(4):
                tf.summary.scalar('l1_loss_' + str(i), self.l1_reconstruction_loss_left[i] + self.l1_reconstruction_loss_right[i], collections=self.model_collection)
                tf.summary.scalar('image_loss_' + str(i), self.image_loss_left[i] + self.image_loss_right[i], collections=self.model_collection)
                tf.summary.image('disp_left_est_' + str(i), self.disp_left_est[i] , max_outputs=4, collections=self.model_collection)
                tf.summary.image('disp_right_est' + str(i), self.disp_right_est[i], max_outputs=4, collections=self.model_collection)
                tf.summary.histogram('disp_left_est_' + str(i), self.disp_left_est[i], collections=self.model_collection)
                tf.summary.histogram('disp_right_est_' + str(i), self.disp_right_est[i], collections=self.model_collection)
                if self.params.full_summary:
                    tf.summary.image('left_est_' + str(i), self.left_est[i], max_outputs=4, collections=self.model_collection)
                    tf.summary.image('right_est_' + str(i), self.right_est[i], max_outputs=4, collections=self.model_collection)

            tf.summary.scalar('d_loss', self.d_loss, collections=self.model_collection)
            if self.params.full_summary:
                tf.summary.image('left',  self.left,   max_outputs=4, collections=self.model_collection)
                tf.summary.image('right', self.right,  max_outputs=4, collections=self.model_collection)
