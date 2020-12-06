
# # 1. Set up Directories

# In[1]:

import os
import pathlib
import cv2
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
from tensorflow import keras

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

train_30fps_dir = os.path.join(*['data', 'REDS_VTSR', 'train', 'train_30fps'])
# print(train_30fps_dir)


# In[3]:

train_30fps_dir = [os.path.join(train_30fps_dir, p) for p in os.listdir(train_30fps_dir)]
# print('Train 30fps', train_30fps_dir[:2])

# ## 1.2. Randomize Videos Paths

# In[7]:

import random
random.shuffle(train_30fps_dir) # make the training dataset random
# random.shuffle(train_60fps_dir) # make the training dataset random

# ## 1.3. Get Image Paths

# In[9]:

train_image_30fps_paths = []
for video_path in train_30fps_dir:
    for x in os.listdir(video_path):
        train_image_30fps_paths.append(os.path.join(video_path, x))

# output format: [image1.png, image2.png,...]

# In[10]:

# # 2. Loading Data

# ## 2.1. Train Dataset Pipeline


# In[13]:


hr_height = 360 // 2
hr_width = 640 // 2
scale = 2

lr_height = hr_height // scale
lr_width = hr_width // scale


# In[14]:


# 1 image as 1 element

def reverse(ds):
    """
    Function that randomly reverse frames sequence in 1 video.
    Args:
        ds: A tf dataset.
    Returns:
        ds: A tf dataset with reversed frames sequence.
    """ 
#     reverse squence randomly
    method_list = ['reverse', None]
    reverse_method = random.choice(method_list)
    
    image_list = list(ds.as_numpy_iterator())

    if reverse_method == 'reverse':
        image_list.reverse()

    return tf.data.Dataset.from_tensor_slices(image_list)

def parse_image(image_path):
    """
    Function that loads the images given the path.
    Args:
        image_path: The paths to frames in the video.
    Returns:
        image: A tf tensor of the loaded frames.
    """
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)

    return image

def random_crop_resize(ds):
    """
    Function that randomly crop or resize image to desired resolution to produce high_res image.
    Args:
        ds: A tf dataset.
    Returns:
        ds: A tf dataset with cropped or resized frames.
    """
    method_list = ['crop', 'resize']
    crop_resize_method = random.choice(method_list)

    method_list = ['bilinear', 'gaussian', 'nearest', 'area']
    downsampling_method = random.choice(method_list)

    def random_crop(image):
        """
        Function that randomly crop image to desired resolution to produce high_res image.
        Args:
            image: A tf tensor of the loaded frames.
        Returns:
            image: A tf tensor of cropped frames.
        """

        # resolution under 360 is too smal that cropping out too much details of the image.
        # if hr_height < 360:
        image = tf.image.random_crop(image, [hr_height*2, hr_width*2, 3])
        image = tf.image.resize(image, 
                                [hr_height, hr_width],
                                preserve_aspect_ratio=True,
                                method=downsampling_method)
        # else:
            # image = tf.image.random_crop(image, [hr_height, hr_width, 3])

        return image

    def downsampling(image):
        """
        Function that resize image to desired resolution.
        Downsampling methods: ['bilinear', 'gaussian', 'nearest', 'area']
        Args:
            image: A tf tensor of the loaded frames.
        Returns:
            image: A tf tensor of resized frames.
        """
#         print(tf.shape(high_res)[0])
        image = tf.image.resize(image, 
                                [hr_height, hr_width],
                                preserve_aspect_ratio=True,
                                method=downsampling_method)

        return image

    if crop_resize_method == 'crop':
        # randomly crop frame
        ds = ds.map(random_crop, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    else:
        # downsampling frame
        ds = ds.map(downsampling, num_parallel_calls=tf.data.experimental.AUTOTUNE)

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
    
    def flip_left_right(image):
        image = tf.image.flip_left_right(image)
        return image
    
    def flip_up_down(image):
        image = tf.image.flip_up_down(image)
        return image
    
    if flip_method == 'horizontal':
        ds = ds.map(flip_up_down, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    elif flip_method == 'vertical':
        ds = ds.map(flip_left_right, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        
    return ds

def high_low_res_pairs(ds):
    """
    Function that generates a low resolution image given the high resolution image with random methods.
    Downsampling methods: ['bilinear', 'gaussian', 'nearest', 'area']
    Args:
        ds: A tf dataset.
    Returns:
        ds: A tf dataset with low and high res images.
    """
    method_list = ['bilinear', 'gaussian', 'nearest', 'area']
    downsampling_method = random.choice(method_list)

    def downsampling(high_res):
        """
        Function that generates a low resolution image given the high resolution image.
        Args:
            high_res: A tf tensor of the high res image.
        Returns:
            low_res: A tf tensor of the low res image.
            high_res: A tf tensor of the high res image.
        """
#         print(tf.shape(high_res)[0])
        low_res = tf.image.resize(high_res, 
                                  [lr_height, lr_width],
                                  preserve_aspect_ratio=True,
                                  method=downsampling_method)

        return low_res, high_res
    
    ds = ds.map(downsampling, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
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

def dataset(image_paths, batch_size=2):
    """
    Returns a tf dataset object with specified mappings. No shuffle and No repeat.
    No shuffle because it will screw up the frame sequence.
    No repeat because training model will use a manual for loop.
    Args:
        image_paths: Str, Path to images.
        batch_size: Int, The number of elements in a batch returned by the dataset.
    Returns:
        dataset: A tf dataset object.
    """
    
    # Generate tf dataset from high res video paths.
    dataset = tf.data.Dataset.from_tensor_slices(image_paths)

    # Prefetch the data for optimal GPU utilization.
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    # apply: Applies a transformation function to the whole dataset as once. Good for functions with the same random arg.

    # randomly reverse frames sequence in 1 video
    dataset = dataset.apply(reverse)

    # image paths to tensor
    dataset = dataset.map(parse_image, num_parallel_calls=AUTOTUNE)

    # randomly crop frame
    dataset = dataset.apply(random_crop_resize)

    # randomly flip all frames in 1 video
    dataset = dataset.apply(flip)

    # Generate low resolution by downsampling.
    dataset = dataset.apply(high_low_res_pairs)

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
lr_shape = (lr_height, lr_width, 3)


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
    sr_features = vgg_model(sr) / 12.75
    hr_features = vgg_model(hr) / 12.75
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
lr = 1e-3
#  - (1e-2 * ((40 * 1200) // 20000))

gen_schedule = keras.optimizers.schedules.ExponentialDecay(
    lr,
    decay_steps=20000,
    decay_rate=1e-2,
    staircase=True
)

disc_schedule = keras.optimizers.schedules.ExponentialDecay(
    lr * 5,  # TTUR - Two Time Scale Updates
    decay_steps=20000,
    decay_rate=1e-2,
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

        # Adversarial Loss need to be decreased. Why smallen it?
        # 1e-3 * 
        adv_loss = tf.keras.losses.BinaryCrossentropy()(valid, fake_prediction)
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

epochs = 18
# speed: 14 min/epoch

# training history: 
# 5 epochs (first): 1 hours
# 40 epochs: 9 hours
# 18 epochs:

batch_size = 9

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
    train_dataset = dataset(train_image_30fps_paths, batch_size=batch_size)
    # sample_train_dataset = dataset(train_image_30fps_paths[:180], batch_size=batch_size)

    with tf.device('/device:GPU:1'):
        train(gen_model, disc_model, train_dataset, train_summary_writer, log_iter=200)

# import os
import time
time.sleep(10)
os.system('shutdown /p /f')