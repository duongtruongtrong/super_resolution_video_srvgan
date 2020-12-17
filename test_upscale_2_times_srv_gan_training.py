
# # 1. Set up Directories

# In[1]:

import os
import pathlib
import cv2
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
from tensorflow import keras

import random

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


# In[2]:

# In[2]:

train_high_dir = 'E:/CoderSchool_Final_Project/super_resolution_video/data/extra_frames/high_res'
# train_low_dir = 'E:/CoderSchool_Final_Project/super_resolution_video/data/extra_frames/low_res'

# In[3]:

train_high_dir = [os.path.join(train_high_dir, p) for p in os.listdir(train_high_dir)]

# train_low_dir = [os.path.join(train_low_dir, p) for p in os.listdir(train_low_dir)]

# # ## 1.3. Get Image Paths

# # In[9]:

# train_image_30fps_paths = []
# for video_path in train_30fps_dir:
#     for x in os.listdir(video_path):
#         train_image_30fps_paths.append(os.path.join(video_path, x))

# output format: [image1.png, image2.png,...]

# In[10]:

# # 2. Loading Data

# ## 2.1. Train Dataset Pipeline


# In[13]:


hr_height = 180
hr_width = 320
scale = 2

lr_height = hr_height // scale
lr_width = hr_width // scale

max_start_height = 180-lr_height
max_start_width = 320-lr_width

# In[14]:


# 1 image as 1 element

# can't do reverse for 2 list at the same time
# def reverse(ds):
#     """
#     Function that randomly reverse frames sequence in 1 video.
#     Args:
#         ds: A tf dataset.
#     Returns:
#         ds: A tf dataset with reversed frames sequence.
#     """ 
# #     reverse squence randomly
#     method_list = ['reverse', None]
#     reverse_method = random.choice(method_list)
    
#     image_list = list(ds.as_numpy_iterator())

#     if reverse_method == 'reverse':
#         image_list.reverse()

#     return tf.data.Dataset.from_tensor_slices(image_list)

def parse_image(low_res_path, high_res_path):
    """
    Function that loads the images given the path.
    Args:
        low_res_path: The paths to frames in low res video.
        high_res_path: The paths to frames in high res video.
    Returns:
        low_res: The tf tensor of loaded low res image.
        high_res: The tf tensor of loaded high res image.
    """
    low_res = tf.io.read_file(low_res_path)
    low_res = tf.image.decode_jpeg(low_res, channels=3)
    low_res = tf.image.convert_image_dtype(low_res, tf.float32)

    high_res = tf.io.read_file(high_res_path)
    high_res = tf.image.decode_jpeg(high_res, channels=3)
    high_res = tf.image.convert_image_dtype(high_res, tf.float32)

    return low_res, high_res

def random_crop(ds):
    """
    Function that randomly crop image to desired resolution to produce high_res image.
    Args:
        ds: A tf dataset.
    Returns:
        ds: A tf dataset with cropped frames.
    """

    rand_start_lr_height = tf.random.uniform(shape=[], maxval=max_start_height, dtype=tf.int32)
    rand_start_lr_width = tf.random.uniform(shape=[], maxval=max_start_width, dtype=tf.int32)

    rand_start_hr_height = rand_start_lr_height*scale
    rand_start_hr_width = rand_start_lr_width*scale

    def crop(low_res, high_res):
        """
        Function that randomly crop image to desired resolution to produce high_res image.
        Args:
            low_res: The tf tensor of the low res image.
            high_res: The tf tensor of the high res image.
        Returns:
            low_res: The tf tensor of cropped low res image.
            high_res: The tf tensor of cropped high res image.
        """

        low_res = tf.slice(low_res, [rand_start_lr_height, rand_start_lr_width, 0], [lr_height, lr_width, 3])
        high_res = tf.slice(high_res, [rand_start_hr_height, rand_start_hr_width, 0], [hr_height, hr_width, 3])

        return low_res, high_res

    ds = ds.map(crop, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    return ds

def flip(ds):
    """
    Function that flip horizontally/vertically all images in 1 dataset.
    Args:
        ds: A tf dataset.
    Returns:
        ds: A tf dataset with flipped images.
    """ 
#     flip the image randomly
    method_list = ['horizontal', 'vertical', None]
    flip_method = random.choice(method_list)
    
    def flip_left_right(low_res, high_res):
        low_res = tf.image.flip_left_right(low_res)
        high_res = tf.image.flip_left_right(high_res)
        return low_res, high_res
    
    def flip_up_down(low_res, high_res):
        low_res = tf.image.flip_up_down(low_res)
        high_res = tf.image.flip_up_down(high_res)
        return low_res, high_res
    
    if flip_method == 'horizontal':
        ds = ds.map(flip_up_down, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    elif flip_method == 'vertical':
        ds = ds.map(flip_left_right, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        
    return ds


def rescale(low_res, high_res):
    """
    Function that rescales the pixel values of high_res to the -1 to 1 range.
    For use with the generator output tanh function.
    Args:
        low_res: The tf tensor of the low res image.
        high_res: The tf tensor of the high res image.
    Returns:
        low_res: The tf tensor of the low res image, rescaled.
        high_res: The tf tensor of the high res image, rescaled.
    """
    high_res = high_res * 2.0 - 1.0

    return low_res, high_res

def dataset(low_res_path, high_res_path, batch_size=2):
    """
    Returns a tf dataset object with specified mappings. No shuffle and No repeat.
    No shuffle because it will screw up the frame sequence.
    No repeat because training model will use a manual for loop.
    Args:
        low_res_path: Str, Path to low resolution images.
        high_res_path: Str, Path to high resolution images.
        batch_size: Int, The number of elements in a batch returned by the dataset.
    Returns:
        dataset: A tf dataset object.
    """
    
    # Generate tf dataset from low and high res video paths.
    dataset = tf.data.Dataset.from_tensor_slices((low_res_path, high_res_path))

    # Prefetch the data for optimal GPU utilization.
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    # apply: Applies a transformation function to the whole dataset as once. Good for functions with the same random arg.

    # # randomly reverse frames sequence in 1 video
    # dataset = dataset.apply(reverse)

    # image paths to tensor
    dataset = dataset.map(parse_image, num_parallel_calls=AUTOTUNE)

    # randomly crop frame
    dataset = dataset.apply(random_crop)

    # randomly flip all frames in 1 video
    dataset = dataset.apply(flip)

    # Rescale the values in the input
    dataset = dataset.map(rescale, num_parallel_calls=AUTOTUNE)

    # Batch the input, drop remainder to get a defined batch size.
    dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(AUTOTUNE)

    return dataset

# In[16]:

# # 3. Models

# ## 3.1. VGG Model and Feature Loss

# In[20]:


hr_shape = (hr_height, hr_width, 3)
# lr_shape = (lr_height, lr_width, 3)


# In[21]:


# We use a pre-trained VGG19 model to extract image features from the high resolution
# and the generated high resolution images and minimize the mse between them
# Get the vgg network. Extract features from Block 5, last convolution, exclude layer block5_pool (MaxPooling2D)
vgg = tf.keras.applications.VGG19(weights="imagenet", input_shape=hr_shape, include_top=False)
vgg.trainable = False

# Create model and compile
vgg_model = tf.keras.models.Model(inputs=vgg.input, outputs=vgg.get_layer("block5_conv4").output)
# vgg_model.summary()


# In[22]:

@tf.function
def feature_loss(hr, sr):
    """
    Returns Mean Square Error of VGG19 feature extracted original image (y) and VGG19 feature extracted generated image (y_hat).
    Args:
        hr: A tf tensor of original image (y)
        sr: A tf tensor of generated image (y_hat)
    Returns:
        mse: Mean Square Error.
    """
    sr = tf.keras.applications.vgg19.preprocess_input(((sr + 1.0) * 255) / 2.0)
    hr = tf.keras.applications.vgg19.preprocess_input(((hr + 1.0) * 255) / 2.0)
    sr_features = vgg_model(sr) / 12.75 # why?
    hr_features = vgg_model(hr) / 12.75 # why?
    mse = tf.keras.losses.MeanSquaredError()(hr_features, sr_features)
    return mse

# not helping because it makes adversial loss increase and discriminator loss decrease
# @tf.function
# def content_loss(hr, sr):
#     """
#     Returns Mean Square Error of original image (y) and generated image (y_hat).
#     Args:
#         hr: A tf tensor of original image (y)
#         sr: A tf tensor of generated image (y_hat)
#     Returns:
#         mse: Mean Square Error.
#     """
#     sr = 255.0 * (sr + 1.0) / 2.0
#     hr = 255.0 * (hr + 1.0) / 2.0
#     mse = tf.keras.losses.MeanAbsoluteError()(sr, hr)
#     return mse

# ## 3.2. Optimizers

# In[28]:


# Define a learning rate decay schedule.
lr = 1e-4
# * 0.95 ** ((75 * 1200) // 100000)
# print(lr)

gen_schedule = keras.optimizers.schedules.ExponentialDecay(
    lr,
    decay_steps=100000,
    decay_rate=0.95, # 95%
    staircase=True
)

disc_schedule = keras.optimizers.schedules.ExponentialDecay(
    lr, # * 5  # TTUR - Two Time Scale Updates
    decay_steps=100000,
    decay_rate=0.95, # 95%
    staircase=True
)

gen_optimizer = keras.optimizers.Adam(learning_rate=gen_schedule)
disc_optimizer = keras.optimizers.Adam(learning_rate=disc_schedule)

# # 4. Training

# In[29]:


# Recreate the exact same model, including its weights and the optimizer
disc_model = tf.keras.models.load_model('models/discriminator_upscale_2_times.h5')

for layer in disc_model.layers:
    disc_model_output_shape = layer.output_shape
    # (None, 12, 20, 1)

# 23, 40, 1
height_patch = disc_model_output_shape[1]
# int(hr_height / 2 ** 4)

width_patch = disc_model_output_shape[2]
# int(hr_width / 2 ** 4)

disc_patch = (height_patch, width_patch, 1)
# disc_patch

pretrain_iteration = 1
train_iteration = 1

# In[30]:

@tf.function
def pretrain_step(gen_model, x, y):
    """
    Single step of generator pre-training.
    Args:
        gen_model: A compiled generator model.
        x: The low resolution image tensor.
        y: The high resolution image tensor.
    """
    with tf.GradientTape() as tape:
        fake_hr = gen_model(x)
        loss_mse = tf.keras.losses.MeanSquaredError()(y, fake_hr)

    grads = tape.gradient(loss_mse, gen_model.trainable_variables)
    gen_optimizer.apply_gradients(zip(grads, gen_model.trainable_variables))

    return loss_mse


def pretrain_generator(gen_model, dataset, writer, log_iter=200):
    """Function that pretrains the generator slightly, to avoid local minima.
    Args:
        gen_model: A compiled generator model.
        dataset: A tf dataset object of low and high res images to pretrain over.
        writer: A summary writer object.
    Returns:
        None
    """
    global pretrain_iteration

    with writer.as_default():
        for _ in range(1):
            for x, y in dataset:
                loss = pretrain_step(gen_model, x, y)
                if pretrain_iteration % log_iter == 0:
                    print(f'Pretrain Step: {pretrain_iteration}, Pretrain MSE Loss: {loss}')
                    tf.summary.scalar('MSE Loss', loss, step=tf.cast(pretrain_iteration, tf.int64))
                    writer.flush()
                pretrain_iteration += 1

@tf.function
def train_step(gen_model, disc_model, x, y):
    """Single train step function for the SRGAN.
    Args:
        gen_model: A compiled generator model.
        disc_model: A compiled discriminator model.
        x: The low resolution input image.
        y: The desired high resolution output image.
    Returns:
        disc_loss: The mean loss of the discriminator.
        adv_loss: The Binary Crossentropy loss between real label and predicted label.
        cont_loss: The Mean Square Error of VGG19 feature extracted original image (y) and VGG19 feature extractedgenerated image (y_hat).
        mse_loss: The Mean Square Error of original image (y) and generated image (y_hat).
    """
    # Label smoothing for better gradient flow
    valid = tf.ones((x.shape[0],) + disc_patch)
    fake = tf.zeros((x.shape[0],) + disc_patch)
#     print('label')
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # From low res. image generate high res. version
        fake_hr = gen_model(x)
#         print('gen_model')

        # Train the discriminators (original images = real / generated = Fake)
        valid_prediction = disc_model(y)
        fake_prediction = disc_model(fake_hr)
#         print('disc_model')
        # Generator loss
        feat_loss = feature_loss(y, fake_hr)

        # not helping because it makes adversial loss increase and discriminator loss decrease
        # cont_loss = content_loss(y, fake_hr)

        # Adversarial Loss need to be decreased. Smallen the number to make it decrease faster
        adv_loss = 1e-3 * tf.keras.losses.BinaryCrossentropy()(valid, fake_prediction)
        mse_loss = tf.keras.losses.MeanSquaredError()(y, fake_hr)
        perceptual_loss = feat_loss + adv_loss + mse_loss

        # Discriminator loss
        valid_loss = tf.keras.losses.BinaryCrossentropy()(valid, valid_prediction)
        fake_loss = tf.keras.losses.BinaryCrossentropy()(fake, fake_prediction)
        disc_loss = tf.add(valid_loss, fake_loss)

#         print('finish gradient')
        
    # Backprop on Generator
    gen_grads = gen_tape.gradient(perceptual_loss, gen_model.trainable_variables)
    gen_optimizer.apply_gradients(zip(gen_grads, gen_model.trainable_variables))

    # Backprop on Discriminator
    disc_grads = disc_tape.gradient(disc_loss, disc_model.trainable_variables)
    disc_optimizer.apply_gradients(zip(disc_grads, disc_model.trainable_variables))
#     print('optimizer')
    
    return disc_loss, adv_loss, feat_loss, mse_loss


def train(gen_model, disc_model, dataset, writer, log_iter=200):
    """
    Function that defines a single training step for the SR-GAN.
    Args:
        gen_model: A compiled generator model.
        disc_model: A compiled discriminator model.
        dataset: A tf data object that contains low and high res images.
        log_iter: Number of iterations after which to add logs in 
                  tensorboard.
        writer: Summary writer
    """
    global train_iteration

    with writer.as_default():
        # Iterate over dataset
        for x, y in dataset:
            disc_loss, adv_loss, feat_loss, mse_loss = train_step(gen_model, disc_model, x, y)
#             print(train_iteration)
            # Log tensorboard summaries if log iteration is reached.
            if train_iteration % log_iter == 0:
                print(f'Train Step: {train_iteration}, Adversarial Loss: {adv_loss}, Feature Loss: {feat_loss}, MSE Loss: {mse_loss}, Discriminator Loss: {disc_loss}')
                
                tf.summary.scalar('Adversarial Loss', adv_loss, step=train_iteration)
                tf.summary.scalar('Feature Loss', feat_loss, step=train_iteration)
                # tf.summary.scalar('Content Loss', cont_loss, step=train_iteration)
                tf.summary.scalar('MSE Loss', mse_loss, step=train_iteration)
                tf.summary.scalar('Discriminator Loss', disc_loss, step=train_iteration)

                if train_iteration % (log_iter*10) == 0:
                    tf.summary.image('Low Res', tf.cast(255 * x, tf.uint8), step=train_iteration)
                    tf.summary.image('High Res', tf.cast(255 * (y + 1.0) / 2.0, tf.uint8), step=train_iteration)
                    tf.summary.image('Generated', tf.cast(255 * (gen_model.predict(x) + 1.0) / 2.0, tf.uint8), step=train_iteration)

                gen_model.save('models/generator_upscale_2_times.h5')
                disc_model.save('models/discriminator_upscale_2_times.h5')
                writer.flush()
            train_iteration += 1

# ============================================================
# Load pretrain models (generator.h5 and disc_model)

# Recreate the exact same model, including its weights and the optimizer
gen_model = tf.keras.models.load_model('models/generator_upscale_2_times.h5')

# Define the directory for saving the SRGAN training tensorbaord summary.
train_summary_writer = tf.summary.create_file_writer('upscale_2_times_logs/train')

epochs = 15 # Turn on Turbo mode
# speed: 14 min/epoch
# 32k frames speed: 44 min/epoch
# 18k frames speed: 30 min/epoch

# training history: total 261, 56.5 hours
# 10 epochs (first): 2 hours
# 10 epochs: 2 hours
# 7 epochs: 1.5 hours
# 8 epochs: 1.5 hours
# 30 epochs: 6.5 hours
# 10 epochs: 2 hours
# 40 epochs: 8 hours
# 8 epochs (with new data): 1.7 hours
# 2 epochs: 0.3 hours
# 30 epochs: 6.5 hours
# 8 epochs: 1.5 hours
# 50 epochs (120 videos only: 30 val + 30 test + 60 train): 5.7 hours ~ 25 epochs with full dataset (240 videos)
# 18 epochs: 2 hours ~ 9 epochs with full dataset (240 videos)
# 60 epochs: 6.8 hours ~ 30 epcochs with full dataset (240 videos)
# Stop training

# extra videos
# 3 epochs (32232 frames: REDS OpenCV resize + REDS Sony Vegas Pro 17.0 resize + Pexels): 2.5 hours ~ 9 epochs with original REDS
# 15 epochs (18732 frames: REDS Sony Vegas Pro 17.0 resize + Pexels): 6 hours ~ 20 epochs with original REDS

batch_size = 9

# Run training.
for _ in range(epochs):
    print('===================')
    print(f'Epoch: {_}\n')

    # shuffle video directories
    random.shuffle(train_high_dir) # make the training dataset random

    train_low_dir = [p.replace("high_res", "low_res") for p in train_high_dir]

    train_high_image_paths = []
    for video_path in train_high_dir:
        for x in os.listdir(video_path):
            train_high_image_paths.append(os.path.join(video_path, x))
    
    train_low_image_paths = []
    for video_path in train_low_dir:
        for x in os.listdir(video_path):
            train_low_image_paths.append(os.path.join(video_path, x))

    # total 18732 frames
    
    # recreate dataset every epoch to lightly augment the frames. ".repeat()" in dataset pipeline function does not help.
    train_dataset = dataset(train_low_image_paths, train_high_image_paths, batch_size=batch_size)
    # sample_train_dataset = dataset(train_image_30fps_paths[:180], batch_size=batch_size)

    with tf.device('/device:GPU:1'):
        train(gen_model, disc_model, train_dataset, train_summary_writer, log_iter=208)

# import os
print('Shutting Down!!!')
import time
time.sleep(10)
os.system('shutdown /p /f')