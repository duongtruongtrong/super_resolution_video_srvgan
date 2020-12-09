#!/usr/bin/env python
# coding: utf-8

from data_loader import DataLoader

from model_building import Model

from model_training import Train

import os

import tensorflow as tf
from tensorflow import keras
import random

# When running the model with conv2d
# UnknownError:  Failed to get convolution algorithm. This is probably because cuDNN failed to initialize, so try looking to see if a warning log message was printed above.
# it is because the cnDNN version you installed is not compatible with the cuDNN version that compiled in tensorflow. -> Let conda or pip automatically choose the right version of tensorflow and cudnn.
# or run out of graphics card RAM -> must set limit for GPU RAM. Splitting into 2 logical GPU with different RAM limit. By default, Tensorflow will use on the logical GPU: 0, the GPU: 1 will be used for training generator and discriminator models.

# https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth
# https://leimao.github.io/blog/TensorFlow-cuDNN-Failure/
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    # Currently, memory growth needs to be the same across GPUs
#     for gpu in gpus:
#       tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [
         tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*0.15),
         tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*5.45) # for Training
#          tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*5) # for Testing
        ])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)

# # 1. Get Image Paths

train_30fps_dir = os.path.join(*['data', 'REDS_VTSR', 'train', 'train_30fps'])
# print(train_30fps_dir)

val_30fps_dir = os.path.join(*['data', 'REDS_VTSR', 'val', 'val_30fps'])
# print(val_30fps_dir)

test_15fps_dir = os.path.join(*['data', 'REDS_VTSR', 'test', 'test_15fps'])
# print(test_15fps_dir)

train_30fps_dir = [os.path.join(train_30fps_dir, p) for p in os.listdir(train_30fps_dir)]
# print('Train 30fps', train_30fps_dir[:2])

val_30fps_dir = [os.path.join(val_30fps_dir, p) for p in os.listdir(val_30fps_dir)]
# print('Val 30fps', val_30fps_dir[:2])

test_15fps_dir = [os.path.join(test_15fps_dir, p) for p in os.listdir(test_15fps_dir)]
# print('Val 15fps', test_15fps_dir[:2])

# random choose 180 videos from train_30fps_dir to limit the overfitting to that dataset.
train_30fps_dir = train_30fps_dir + val_30fps_dir + test_15fps_dir

# # 2. Shapes

hr_height = 360 // 2
hr_width = 640 // 2
scale = 2

lr_height = hr_height // scale
lr_width = hr_width // scale

hr_shape = (hr_height, hr_width, 3)
lr_shape = (lr_height, lr_width, 3)

# # 3. Models

# light model not really real time!! 
# Tested:
# filters_num=8, residual_block_num=8
# filters_num=16, residual_block_num=8

models = Model(hr_shape, lr_shape, filters_num=32, residual_block_num=5)

gen_model = models.build_generator()
disc_model = models.build_discriminator()

logs_folder_name = 'upscale_2_times_logs'

gen_model_save_path = 'models/generator_upscale_2_times.h5'
disc_model_save_path = 'models/discriminator_upscale_2_times.h5'

# # 4. Train Dataset Pipeline

train_30fps_dir = random.sample(train_30fps_dir, 10)

random.shuffle(train_30fps_dir)
train_image_30fps_paths = []
for video_path in train_30fps_dir:
    for x in os.listdir(video_path):
        train_image_30fps_paths.append(os.path.join(video_path, x))

batch_size = 9

data_loader = DataLoader(hr_height, hr_width, lr_height, lr_width, batch_size)

train_dataset = data_loader.train_dataset(train_image_30fps_paths)

# # 4. Training

training = Train(gen_model, disc_model, hr_shape, learning_rate=1e-3, gen_model_save_path=gen_model_save_path, disc_model_save_path=disc_model_save_path)

with tf.device('/device:GPU:1'):

    # Define the directory for saving pretrainig loss tensorboard summary.
    pretrain_summary_writer = tf.summary.create_file_writer(f'{logs_folder_name}/pretrain')
    
    # Run pre-training.

    training.pretrain_generator(train_dataset, pretrain_summary_writer, log_iter=200)
    gen_model.save(gen_model_save_path)
    
    # Define the directory for saving the SRGAN training tensorbaord summary.
    train_summary_writer = tf.summary.create_file_writer(f'{logs_folder_name}/train')

    epochs = 10 # Turn on Turbo mode
    # speed: 14 min/epoch

    # training history: 
    # 10 epochs (first): 2 hours
    
# Run training.
for _ in range(epochs):
    print('===================')
    print(f'Epoch: {_}\n')

    # shuffle video directories
    # [train_30fps_dir, train_60fps_dir, val_30fps_dir, val_60fps_dir]
    random.shuffle(train_30fps_dir) # make the training dataset random

    train_image_30fps_paths = []
    for video_path in train_30fps_dir:
        for x in os.listdir(video_path):
            train_image_30fps_paths.append(os.path.join(video_path, x))

    # recreate dataset every epoch to lightly augment the frames. ".repeat()" in dataset pipeline function does not help.
    train_dataset = data_loader.train_dataset(train_image_30fps_paths)
    # sample_train_dataset = data_loader.train_dataset(train_image_30fps_paths[:180])

    with tf.device('/device:GPU:1'):
        training.train(train_dataset, train_summary_writer, log_iter=200)
        
# import os
import time
print('Shutting Down!!!')
time.sleep(10)
os.system('shutdown /p /f')