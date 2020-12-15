from flask import Flask, render_template, request, jsonify, Response, redirect, url_for
 
import tensorflow as tf

from werkzeug.utils import secure_filename

import os

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
from cv2 import cv2

import time

from threading import Thread

#Initialize the Flask app
app = Flask(__name__, static_folder='static')

# https://www.pyimagesearch.com/2019/09/02/opencv-stream-video-to-web-browser-html-page/

# https://towardsdatascience.com/video-streaming-in-web-browsers-with-opencv-flask-93a38846fe00 
camera = cv2.VideoCapture(0)

# resize capture resulution
# https://www.kurokesu.com/main/2020/07/12/pulling-full-resolution-from-a-webcam-with-opencv-windows/

# width = 640
# height = 480

width = 1280
height = 720

target_width = 640
target_height = 360

codec = 0x47504A4D  # MJPG
# camera.set(cv2.CAP_PROP_FPS, 30.0)
camera.set(cv2.CAP_PROP_FOURCC, codec)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

resize_width, resize_height = (320, 180)

model = tf.keras.models.load_model("models/generator_upscale_2_times.h5")

# Define arbitrary spatial dims, and 3 channels.
inputs = tf.keras.Input((None, None, 3))

# Trace out the graph using the input:
outputs = model(inputs)

model = tf.keras.models.Model(inputs, outputs)

model_4x = tf.keras.models.load_model("models/generator_upscale_4_times-FAST-GAN.h5")

# Define arbitrary spatial dims, and 3 channels.
inputs = tf.keras.Input((None, None, 3))

# Trace out the graph using the input:
outputs = model_4x(inputs)

model_4x = tf.keras.models.Model(inputs, outputs)

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
        _, frame = camera.read()  # read the camera frame

        frame = cv2.resize(frame, (resize_width, resize_height), interpolation=cv2.INTER_AREA)
        
        end = time.time()
        duration = end - start
        fps = round(frame_num/duration, 1) # 27 fps

        frame = cv2.putText(frame, 'FPS: ' + str(fps), (10, frame.shape[0] - 10),
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
        
        _, frame = camera.read()  # read the camera frame

        if target_width != width:
            frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
        
        end = time.time()
        duration = end - start
        fps = round(frame_num/duration, 1) # 27 fps

        frame = cv2.putText(frame, 'FPS: ' + str(fps), (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
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

        # start_pred = time.time()
        with tf.device('/device:GPU:1'):
            frame = cv2.resize(frame, (resize_width, resize_height), interpolation=cv2.INTER_AREA)
            
            # opencv image to tensorflow image
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            frame = tf.image.convert_image_dtype(frame, tf.float32)

            frame = tf.expand_dims(frame, axis=0)

        
            frame = model.predict(frame)

            frame = tf.cast(255 * (frame + 1.0) / 2.0, tf.uint8)

            # frame = tf.image.convert_image_dtype(frame, tf.float32)

            # frame = model.predict(frame)

            # frame = tf.cast(255 * (frame + 1.0) / 2.0, tf.uint8)


            # tensor to opencv image
            frame = cv2.cvtColor(frame[0].numpy(), cv2.COLOR_RGB2BGR)
            
            # FPS
            end = time.time()
            duration = end - start
            fps = round(frame_num/duration, 2) 
            # 7 fps, 2 times upscale: 320x180 -> 640x360

            frame = cv2.putText(frame, 'FPS: ' + str(fps), (10, frame.shape[0] - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

            # # predict_time 
            # # 0.08 second/frame, 2 times upscale: 320x180 -> 640x360
            # duration_pred = round(end - start_pred, 3)
            # frame = cv2.putText(frame, 'SPF: ' + str(duration_pred), (10, 30),
            #                                 cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

            frame_num += 1

            _, buffer = cv2.imencode('.jpg', frame)

            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

def upscale_4x_frame():

    frame_num = 1
    start = time.time()

    # loop over frames from the video stream
    while True:
        _, frame = camera.read()  # read the camera frame
        
        # start_pred = time.time()
        with tf.device('/device:GPU:1'):
            frame = cv2.resize(frame, (resize_width, resize_height), interpolation=cv2.INTER_AREA)
            
            # frame = cv2.resize(frame, (width // 2, height // 2))
            # opencv image to tensorflow image
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            frame = tf.image.convert_image_dtype(frame, tf.float32)

            frame = tf.expand_dims(frame, axis=0)

            # frame = model.predict(frame)

            # frame = tf.cast(255 * (frame + 1.0) / 2.0, tf.uint8)

            # frame = tf.image.convert_image_dtype(frame, tf.float32)

            # model_4x
            frame = model_4x.predict(frame)
            # frame = model.predict(frame)

            frame = tf.cast(255 * (frame + 1.0) / 2.0, tf.uint8)

            # tensor to opencv image
            frame = cv2.cvtColor(frame[0].numpy(), cv2.COLOR_RGB2BGR)
            
            # FPS
            end = time.time()
            duration = end - start
            fps = round(frame_num/duration, 2) 
            # 5 fps, 4 times upscale: 320x180 -> 1280x720

            frame = cv2.putText(frame, 'FPS: ' + str(fps), (10, frame.shape[0] - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 3)

            # predict_time 
            # 0.11 second/frame, 4 times upscale: 320x180 -> 1280x720
            # duration_pred = round(end - start_pred, 3)
            # frame = cv2.putText(frame, 'SPF: ' + str(duration_pred), (10, 30),
            #                                 cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

            frame_num += 1

            _, buffer = cv2.imencode('.jpg', frame)

            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

def test_upscale_frame():
    frame_num = 1
    start = time.time()
    # loop over frames from the video stream
    while True:
        _, frame = camera.read()  # read the camera frame

        frame = cv2.resize(frame, (resize_width, resize_height), interpolation=cv2.INTER_AREA)
        frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
        
        end = time.time()
        duration = end - start
        fps = round(frame_num/duration, 1) # 27 fps

        frame = cv2.putText(frame, 'FPS: ' + str(fps), (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        frame_num += 1

        _, buffer = cv2.imencode('.jpg', frame)

        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

@app.route('/')
def index():
    return render_template('home_page.html')

# https://flask.palletsprojects.com/en/1.1.x/patterns/fileuploads/
# https://www.tutorialspoint.com/flask/flask_file_uploading.htm

# video must be in static folder
UPLOAD_FOLDER = 'E:/CoderSchool_Final_Project/super_resolution_video/flask_app/static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

filename = None
video = None
fps = 0
upload_width = 0
upload_height = 0

@app.route('/upload', methods=['POST'])
def upload():
    global filename, video, fps, upload_width, upload_height

    file = request.files['file']

    filename = secure_filename(file.filename)
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(path)

    video = cv2.VideoCapture(path)
    fps = int(video.get(cv2.CAP_PROP_FPS))
    upload_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    upload_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    return render_template('home_page.html', video_name=filename, video_width=upload_width, video_height=upload_height, video_fps=fps)

@app.route('/upscale')
# @app.route('/output_upscale_video')
def output_upscale_video():
    global filename, video, fps, upload_width, upload_height
    start = time.time()
    with tf.device('/device:GPU:1'):
        # Define arbitrary spatial dims, and 3 channels.
        inputs = tf.keras.Input((upload_height, upload_width, 3))

        # Trace out the graph using the input:
        outputs = model(inputs)
        output_width = outputs.shape[2]
        output_height = outputs.shape[1]

        gen_model = tf.keras.models.Model(inputs, outputs)

        output_filename = filename.split('.')[0] + '_upscaled.mp4'
        path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)

        # to play video on html from opencv output
        # https://stackoverflow.com/questions/49530857/python-opencv-video-format-play-in-browser
        # *'H264'
        # *'DIVX'
        # cv2.VideoWriter_fourcc(*'MPEG-4')
        # 0x00000021
        
        output_video = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'H264'), fps, (output_width, output_height))
        
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        record = True
        
        while record:
            # read the next frame from the file
            record, frame = video.read()

            if record:

                # opencv image to tensorflow image
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                frame = tf.image.convert_image_dtype(frame, tf.float32)

                frame = tf.expand_dims(frame, axis=0)

                frame = gen_model.predict(frame)
                frame = tf.cast(255 * (frame + 1.0) / 2.0, tf.uint8)

                # tensor to opencv image
                frame = cv2.cvtColor(frame[0].numpy(), cv2.COLOR_RGB2BGR)

                # save to video
                output_video.write(frame)

    video.release()
    output_video.release()

    duration = round(time.time() - start, 1)

    speed = round(duration/frame_count, 4)

    return render_template('home_page.html', video_name=filename, video_width=upload_width, video_height=upload_height, video_fps=fps, upscale_video_name=output_filename, upscale_video_height=output_height, upscale_video_width=output_width, duration=duration, upscale_speed=speed, frame_count=frame_count)

@app.route('/webcam')
def upscale_2x_webcam():
    return render_template('webcam.html', original_width=resize_width, original_height=resize_height, target_width=target_width, target_height=target_height)

@app.route('/comparision')
def comparision():
    return render_template('comparision_page.html', original_width=resize_width, original_height=resize_height, target_width=target_width, target_height=target_height) 

@app.route('/upscale_4x_page')
def upscale_4x_webcam():
    return render_template('upscale_4x.html', original_width=resize_width, original_height=resize_height, target_width=width, target_height=height)


@app.route('/video_feed')
def video_feed():
    return Response(webcam_low_res(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_upscale')
def upscale_2x():
    return Response(upscale_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_upscale_4x')
def upscale_4x():
    return Response(upscale_4x_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/high_res')
def high_res():
    return Response(webcam_high_res(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/test_upscale')
def test_upscale():
    return Response(test_upscale_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)