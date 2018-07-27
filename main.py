# Code for
# Unsupervised Adversarial Depth Estimation using Cycled Generative Networks
# Andrea Pilzer, Dan Xu, Mihai Puscas, Elisa Ricci, Nicu Sebe
#
# 3DV 2018 Conference, Verona, Italy
#
# parts of the code from https://github.com/mrharicot/monodepth
#

from __future__ import absolute_import, division, print_function

# only keep warnings and errors
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='1'

import numpy as np
import argparse
import re
import time
import tensorflow as tf
import tensorflow.contrib.slim as slim
import matplotlib.pyplot as plt

from monodepth_dataloader import *
from average_gradients import *
from model_stereo_depthGAN import *


parser = argparse.ArgumentParser(description='Monodepth TensorFlow implementation.')

parser.add_argument('--mode',                      type=str,   help='train or test', default='train')
parser.add_argument('--model_name',                type=str,   help='model name', default='stereo_depthGAN')
parser.add_argument('--encoder',                   type=str,   help='type of encoder, resnet50', default='resnet50')
parser.add_argument('--dataset',                   type=str,   help='dataset to train on, kitti, or cityscapes', default='kitti')
parser.add_argument('--data_path',                 type=str,   help='path to the data', required=True)
parser.add_argument('--filenames_file',            type=str,   help='path to the filenames text file', required=True)
parser.add_argument('--input_height',              type=int,   help='input height', default=256)
parser.add_argument('--input_width',               type=int,   help='input width', default=512)
parser.add_argument('--batch_size',                type=int,   help='batch size', default=8)
parser.add_argument('--num_epochs',                type=int,   help='number of epochs', default=50)
parser.add_argument('--learning_rate',             type=float, help='initial learning rate', default=1e-5)
parser.add_argument('--lr_loss_weight',            type=float, help='left-right consistency weight', default=1.0)
parser.add_argument('--do_stereo',                             help='if set, will train the stereo model', action='store_true')
parser.add_argument('--wrap_mode',                 type=str,   help='bilinear sampler wrap mode, edge or border', default='border')
parser.add_argument('--use_deconv',                            help='if set, will use transposed convolutions', action='store_true')
parser.add_argument('--num_gpus',                  type=int,   help='number of GPUs to use for training', default=1)
parser.add_argument('--num_threads',               type=int,   help='number of threads to use for data loading', default=8)
parser.add_argument('--output_directory',          type=str,   help='output directory for test disparities, if empty outputs to checkpoint folder', default='')
parser.add_argument('--log_directory',             type=str,   help='directory to save checkpoints and summaries', default='/data1/pilzer/monodepth-GAN/')
parser.add_argument('--checkpoint_path',           type=str,   help='path to a specific checkpoint to load', default='')
parser.add_argument('--retrain',                               help='if used with checkpoint_path, will restart training from step zero', action='store_true')
parser.add_argument('--full_summary',                          help='if set, will keep more data for each summary. Warning: the file can become very large', action='store_true')
parser.add_argument('--train_branch',             type=str,      help='which branch to train iteratively', default='full')

args = parser.parse_args()

def count_text_lines(file_path):
    f = open(file_path, 'r')
    lines = f.readlines()
    f.close()
    return len(lines)

def train(params):
    """Training loop."""

    with tf.Graph().as_default(), tf.device('/cpu:0'):

        global_step = tf.Variable(0, trainable=False)

        # OPTIMIZER
        num_training_samples = count_text_lines(args.filenames_file)

        steps_per_epoch = np.ceil(num_training_samples / params.batch_size).astype(np.int32)
        num_total_steps = params.num_epochs * steps_per_epoch
        start_learning_rate = args.learning_rate

        boundaries = [np.int32((3/5) * num_total_steps), np.int32((4/5) * num_total_steps)]
        values = [args.learning_rate, args.learning_rate / 2, args.learning_rate / 4]
        learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)

        opt_step = tf.train.AdamOptimizer(learning_rate)

        print("total number of samples: {}".format(num_training_samples))
        print("total number of steps: {}".format(num_total_steps))

        dataloader = MonodepthDataloader(args.data_path, args.filenames_file, params, args.dataset, args.mode)
        left  = dataloader.left_image_batch
        right = dataloader.right_image_batch

        # split for each gpu
        left_splits  = tf.split(left,  args.num_gpus, 0)
        right_splits = tf.split(right, args.num_gpus, 0)

        tower_grads  = []
        tower_losses = []
        reuse_variables = None
        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(args.num_gpus):
                print(i)
                with tf.device('/gpu:%d' % i):
                    model = stereo_depthGAN_Model(params, args.train_branch, args.mode, left_splits[i], right_splits[i], reuse_variables, i)

                    loss = model.total_loss
                    tower_losses.append(loss)

                    reuse_variables = True

                    if args.train_branch == 'a2b':
                        grads_discrA = opt_step.compute_gradients(loss, model.discrA_vars)
                        new_grads_discrA = [(g/10, v) for g, v in grads_discrA]
                        grads_gen2b = opt_step.compute_gradients(loss, model.gen2b_vars)
                        grads_encoder = opt_step.compute_gradients(loss, model.encoder_vars)

                        grads = grads_gen2b + grads_encoder + new_grads_discrA
                    elif args.train_branch == 'b2a':
                        grads_discrB = opt_step.compute_gradients(loss, model.discrB_vars)
                        new_grads_discrB = [(g/10, v) for g, v in grads_discrB]
                        grads_gen2a = opt_step.compute_gradients(loss, model.gen2a_vars)
                        grads_encoder = opt_step.compute_gradients(loss, model.encoder_vars)

                        grads = grads_gen2a + grads_encoder + new_grads_discrB
                    else:
                        grads_discrA = opt_step.compute_gradients(loss, model.discrA_vars)
                        new_grads_discrA = [(g/10, v) for g, v in grads_discrA]
                        grads_gen2b = opt_step.compute_gradients(loss, model.gen2b_vars)
                        grads_discrB = opt_step.compute_gradients(loss, model.discrB_vars)
                        new_grads_discrB = [(g/10, v) for g, v in grads_discrB]
                        grads_gen2a = opt_step.compute_gradients(loss, model.gen2a_vars)
                        grads_encoder = opt_step.compute_gradients(loss, model.encoder_vars)

                        grads = grads_gen2b + new_grads_discrA + grads_gen2a + new_grads_discrB + grads_encoder

                    tower_grads.append(grads)

        grads = average_gradients(tower_grads)

        apply_gradient_op = opt_step.apply_gradients(grads, global_step=global_step)

        total_loss = tf.reduce_mean(tower_losses)

        tf.summary.scalar('learning_rate', learning_rate, ['model_0'])
        tf.summary.scalar('total_loss', total_loss, ['model_0'])
        summary_op = tf.summary.merge_all('model_0')


        # SESSION
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

        # SAVER
        summary_writer = tf.summary.FileWriter(args.log_directory + '/' + args.model_name, sess.graph)
        #train_saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=0)
        train_saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=0)

        # COUNT PARAMS
        total_num_parameters = 0
        for variable in tf.trainable_variables():
            print(variable)
            total_num_parameters += np.array(variable.get_shape().as_list()).prod()
        print("number of trainable parameters: {}".format(total_num_parameters))

        # INIT
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        coordinator = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)

        # LOAD CHECKPOINT IF SET
        if args.checkpoint_path != '':
            if args.train_branch == 'a2b':
                vlist = []
                temp_saver = tf.train.Saver(var_list=vlist)
                temp_saver.restore(sess, args.checkpoint_path)

                if args.retrain:
                    sess.run(global_step.assign(0))
            elif args.train_branch == 'b2a':
                vars_in_checkpoint = tf.contrib.framework.list_variables(args.checkpoint_path)
                all_variables = tf.get_collection_ref(tf.GraphKeys.GLOBAL_VARIABLES)
                vlist = []

                for v in all_variables:
                    if '2b' in v.name:
                        if 'Adam' not in v.name:
                            vlist.append(v)
                    if 'enc' in v.name:
                        if 'Adam' not in v.name:
                            vlist.append(v)
                    if 'discriminatorB' in v.name:
                        if 'Adam' not in v.name:
                            vlist.append(v)
                temp_saver = tf.train.Saver(var_list=vlist)
                temp_saver.restore(sess, args.checkpoint_path)

                if args.retrain:
                    sess.run(global_step.assign(0))
            else:
                vars_in_checkpoint = tf.contrib.framework.list_variables(args.checkpoint_path)
                all_variables = tf.get_collection_ref(tf.GraphKeys.GLOBAL_VARIABLES)
                vlist = []

                for v in all_variables:
                    if '2a' in v.name:
                        if 'Adam' not in v.name:
                            vlist.append(v)
                    if '2b' in v.name:
                        if 'Adam' not in v.name:
                            vlist.append(v)
                    if 'enc' in v.name:
                        if 'Adam' not in v.name:
                            vlist.append(v)
                    if 'discriminator' in v.name:
                        if 'Adam' not in v.name:
                            vlist.append(v)
                temp_saver = tf.train.Saver(var_list=vlist)
                temp_saver.restore(sess, args.checkpoint_path)

                if args.retrain:
                    sess.run(global_step.assign(0))

        # GO!
        start_step = global_step.eval(session=sess)
        start_time = time.time()
        for step in range(start_step, num_total_steps):
            before_op_time = time.time()
            _, loss_value = sess.run([apply_gradient_op, total_loss])
            duration = time.time() - before_op_time
            if step and step % 200 == 0:
                examples_per_sec = params.batch_size / duration
                time_sofar = (time.time() - start_time) / 3600
                training_time_left = (num_total_steps / step - 1.0) * time_sofar
                print_string = 'batch {:>6} | examples/s: {:4.2f} | loss: {:.5f} | time elapsed: {:.2f}h | time left: {:.2f}h'
                print(print_string.format(step, examples_per_sec, loss_value, time_sofar, training_time_left))
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, global_step=step)
            if step and step % 2500 == 0:
                train_saver.save(sess, args.log_directory + '/' + args.model_name + '/model', global_step=step)

        train_saver.save(sess, args.log_directory + '/' + args.model_name + '/model', global_step=num_total_steps)

def test(params):
    """Test function."""
    global_step = tf.Variable(0, trainable=False)

    dataloader = MonodepthDataloader(args.data_path, args.filenames_file, params, args.dataset, args.mode)
    left  = dataloader.left_image_batch
    right = dataloader.right_image_batch

    # split for each gpu
    #left_splits  = tf.split(left,  args.num_gpus, 0)
    #right_splits = tf.split(right, args.num_gpus, 0)

    reuse_variables = None
    with tf.variable_scope(tf.get_variable_scope()):
        for i in range(args.num_gpus):
            with tf.device('/gpu:%d' % i):
                model = stereo_depthGAN_Model(params, args.train_branch, args.mode, left, right, reuse_variables, i)

                reuse_variables = True

    # SESSION
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # SAVER
    train_saver = tf.train.Saver(tf.trainable_variables())
    #train_saver = tf.train.Saver()

    # INIT
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    coordinator = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)

    # RESTORE
    if args.checkpoint_path == '':
        restore_path = tf.train.latest_checkpoint(args.log_directory + '/' + args.model_name)
    else:
        restore_path = args.checkpoint_path
    train_saver.restore(sess, restore_path)

    num_test_samples = count_text_lines(args.filenames_file)

    print('now testing {} files'.format(num_test_samples))
    disparities    = np.zeros((num_test_samples, params.height, params.width), dtype=np.float32)
    os.mkdir(args.log_directory + args.model_name)
    cnt = 0
    while cnt < num_test_samples:

        disp = sess.run(model.disp_out)
        #disp = sess.run(model.disp_right_est[0])
        for d in disp:
            if cnt < num_test_samples:
                disparities[cnt] = d.squeeze()
                cmap = plt.cm.plasma
                norm = plt.Normalize(vmin=disparities[cnt].min(), vmax=disparities[cnt].max())
                images = cmap(norm(disparities[cnt]))
                #plt.imshow(images)
                #plt.show()
                plt.imsave(args.log_directory + args.model_name + '/img_' + str(cnt) + '.png' ,images)
                cnt += 1
    print('done.')

    print('writing disparities.')
    if args.output_directory == '':
        #os.mkdir(args.checkpoint_path)
        output_directory = args.checkpoint_path
    else:
        output_directory = args.output_directory
    np.save(output_directory + '/disparities.npy',    disparities)
    print('done.')

def main(_):

    params = monodepth_parameters(
        encoder=args.encoder,
        height=args.input_height,
        width=args.input_width,
        batch_size=args.batch_size,
        num_threads=args.num_threads,
        num_epochs=args.num_epochs,
        do_stereo=args.do_stereo,
        wrap_mode=args.wrap_mode,
        use_deconv=args.use_deconv,
        lr_loss_weight=args.lr_loss_weight,
        full_summary=args.full_summary,
        num_gpus=args.num_gpus)

    if args.mode == 'train':
        train(params)
    elif args.mode == 'test':
        test(params)

if __name__ == '__main__':
    tf.app.run()
