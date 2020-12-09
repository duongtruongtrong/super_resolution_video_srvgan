from flask import Flask, render_template, render_template, request, jsonify, Response
 
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
    try:
        # Currently, memory growth needs to be the same across GPUs
    #    for gpu in gpus:
    #       tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [
            tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*0.15),
            #  tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*5.45) # for Training
            tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*4) # for Testing
            ])
        # logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        # print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)

import numpy as np
import re
import os
import base64
import uuid
from cv2 import cv2

import time

#Initialize the Flask app
app = Flask(__name__)

# https://www.pyimagesearch.com/2019/09/02/opencv-stream-video-to-web-browser-html-page/

# https://towardsdatascience.com/video-streaming-in-web-browsers-with-opencv-flask-93a38846fe00 
camera = cv2.VideoCapture(0)

# resize capture resulution
# https://www.kurokesu.com/main/2020/07/12/pulling-full-resolution-from-a-webcam-with-opencv-windows/

width = 640
height = 480

codec = 0x47504A4D  # MJPG
# camera.set(cv2.CAP_PROP_FPS, 30.0)
camera.set(cv2.CAP_PROP_FOURCC, codec)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

resize_width, resize_height = (160, 120)

model = tf.keras.models.load_model("models/generator_upscale_2_times.h5")

# Define arbitrary spatial dims, and 3 channels.
inputs = tf.keras.Input((None, None, 3))

# Trace out the graph using the input:
outputs = model(inputs)

model = tf.keras.models.Model(inputs, outputs)

# How many time to use model.predict, each model.predict upscale by 2 times, logarith of 2.
# upscale_times = int(np.log2(640 / resize_width))

# each loop use 1 model due to this error: # Invalid argument:  Conv2DSlowBackpropInput: Size of out_backprop doesn't match computed:
# model_list = []

# Override the model with new inputs and outputs shapes.
# for i in range(upscale_times):
#     model_list.append(tf.keras.models.Model(inputs, outputs))

def webcam_low_res():
    frame_num = 1
    start = time.time()
    # loop over frames from the video stream
    while True:
        success, frame = camera.read()  # read the camera frame

        if not success:
            break
        else:

            frame = cv2.resize(frame, (resize_width, resize_height))
            
            end = time.time()
            duration = end - start
            fps = round(frame_num/duration, 1) # 27 fps

            frame = cv2.putText(frame, str(fps), (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            frame_num += 1

            _, buffer = cv2.imencode('.jpg', frame)

            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

def webcam_high_res():
    frame_num = 1
    start = time.time()
    # loop over frames from the video stream
    while True:
        
        success, frame = camera.read()  # read the camera frame

        if not success:
            break
        else:
            
            end = time.time()
            duration = end - start
            fps = round(frame_num/duration, 1) # 27 fps

            frame = cv2.putText(frame, str(fps), (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)
            frame_num += 1

            _, buffer = cv2.imencode('.jpg', frame)

            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

# use batch failed -> even slower without batch
def upscale_frame():

    frame_num = 1
    start = time.time()

    # loop over frames from the video stream
    while True:
        _, frame = camera.read()  # read the camera frame
        # print('Resolution: ' + str(frame.shape[0]) + ' x ' + str(frame.shape[1]))

        # if not success:
        #     break
        # else:
        start_pred = time.time()
        with tf.device('/device:GPU:1'):
            frame = cv2.resize(frame, (resize_width, resize_height))
            
            # frame = cv2.resize(frame, (width // 2, height // 2))
            # opencv image to tensorflow image
            tensor_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            tensor_image = tf.image.convert_image_dtype(tensor_image, tf.float32)

            tensor_image_gen = tf.expand_dims(tensor_image, axis=0)

        
            tensor_image_gen = model.predict(tensor_image_gen)

            tensor_image_gen = tf.cast(255 * (tensor_image_gen + 1.0) / 2.0, tf.uint8)

            tensor_image_gen = tf.image.convert_image_dtype(tensor_image_gen, tf.float32)
            
            tensor_image_gen = model.predict(tensor_image_gen)

            tensor_image_gen = tf.cast(255 * (tensor_image_gen + 1.0) / 2.0, tf.uint8)

            # for i, model in zip(range(upscale_times), model_list):
            #     with tf.device('/device:GPU:1'):
            #         tensor_image_gen = model.predict(tensor_image_gen)

            #         tensor_image_gen = tf.cast(255 * (tensor_image_gen + 1.0) / 2.0, tf.uint8)

            #         if i != upscale_times-1:
            #             tensor_image_gen = tf.image.convert_image_dtype(tensor_image_gen, tf.float32)

            # tensor to opencv image
            tensor_image_gen = cv2.cvtColor(tensor_image_gen[0].numpy(), cv2.COLOR_RGB2BGR)
            
            # FPS
            end = time.time()
            duration = end - start
            fps = round(frame_num/duration, 2) 
            # 1.8 fps, 16 times upscale, 3 for-loop
            # 2 fps, 8 times upscale, 2 for-loop
            # 3.2 fps, 2 times upscale, 1 for-loop

            tensor_image_gen = cv2.putText(tensor_image_gen, str(fps), (10, tensor_image_gen.shape[0] - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)

            # predict_time 
            # 0.35 second/frame, 16 times upscale, 3 for-loop
            # 0.3 second/frame, 8 times upscale, 2 for-loop
            # 0.18 second/frame, 2 times upscale, 1 for-loop
            duration_pred = round(end - start_pred, 3)
            tensor_image_gen = cv2.putText(tensor_image_gen, str(duration_pred), (30, 100),
                                            cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)

            frame_num += 1

            _, buffer = cv2.imencode('.jpg', tensor_image_gen)

            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

def test_upscale_frame():
    frame_num = 1
    start = time.time()
    # loop over frames from the video stream
    while True:
        success, frame = camera.read()  # read the camera frame

        if not success:
            break
        else:

            frame = cv2.resize(frame, (resize_width, resize_height))
            frame = cv2.resize(frame, (width, height))
            
            end = time.time()
            duration = end - start
            fps = round(frame_num/duration, 1) # 27 fps

            frame = cv2.putText(frame, str(fps), (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)
            frame_num += 1

            _, buffer = cv2.imencode('.jpg', frame)

            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

@app.route('/')
def index():
    return render_template('home_page.html')

@app.route('/comparision')
def comparision():
    return render_template('comparision_page.html') 

@app.route('/video_feed')
def video_feed():
    return Response(webcam_low_res(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_upscale')
def upscale():
    return Response(upscale_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/high_res')
def high_res():
    return Response(webcam_high_res(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/test_upscale')
def test_upscale():
    return Response(test_upscale_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)