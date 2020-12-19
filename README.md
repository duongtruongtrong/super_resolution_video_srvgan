# Super Resolution Video - Video Upscale - 2x SRGAN

# Summary: [Slides](https://docs.google.com/presentation/d/1gQwtfkFHy1mLXXEUTjmJuWvw7kQeu_houVPvBrYJYT4/edit#slide=id.gcb9a0b074_1_0)
Upscale video from low resolution to high resolution to reduce the effect of low bandwidth internet on video chat (meeting) or to reduce streaming video size without losing video quality.

## Problem to Solve:
![image](https://user-images.githubusercontent.com/71629218/99867213-16cff080-2bea-11eb-9d85-0ed480ec6aac.png)

**Eliminate pixelated videos** in streaming, video call/chat/conference due to poor Internet.

## Solution:
![image](https://user-images.githubusercontent.com/71629218/102676146-bb842480-41ce-11eb-8f3c-73ec7f084f4a.png)

## Requirements:
**Speed**: Real time video rendering. Target: 0.04 second/frame ~ 25 FPS (frame per second) video.

**Image Quality**: Acceptable.

## Result:
### Image Quality:
Comparison is between:

- Low resolution input.
- Sony Vegas Pro 17.0 (a software for video editing, similar to Adobe A.I.).
- OpenCV (a python library for processing images).
- My model.
- The real high resolution image.

![comparision](https://user-images.githubusercontent.com/71629218/102676496-677a3f80-41d0-11eb-88fd-0f4a92f31a0d.png)

Image quality is acceptable.

### Speed:
Not yet reach real time rendering speed.

![speed](https://user-images.githubusercontent.com/71629218/102676556-baec8d80-41d0-11eb-81e0-1a076e77a52f.png)

# Project Details:
## Datasets:
### 1. REDS_VTSR:
**RE**alistic and **D**ynamic **S**cenes dataset for **V**ideo **T**emporal **S**uper-**R**esolution (frame interpolation) ([REDS_VTSR](https://seungjunnah.github.io/Datasets/reds_vtsr)) dataset includes **~70 GB** data:

- 13,500 frames.
- Shaking camera movements.
- Outside environment and acivities.

-> Suitable for live streaming videos.

[REDS_VTSR sample video](https://drive.google.com/file/d/1G0JDEubonHLVBaFFzYpaw-i_W7vas3w8/view?usp=sharing)

### 2. Pexels:
- 5,332 frames
- Steady camera movements.
- In-door environment and acivities.
- Capture a lot of human faces.

-> Suitable for video call.

[Pexels sample video](https://drive.google.com/file/d/1VfPxkXx9auWXS9ACDDL6qvHHakQhvfo6/view?usp=sharing)

## Training:
### GAN model:
**Model reference**: [Fast-SRGAN](https://github.com/HasnainRaz/Fast-SRGAN)

**Input**: 160x90p videos.

**Output**: 320x180p videos.

2x upscale.

![image](https://user-images.githubusercontent.com/71629218/102676343-c8edde80-41cf-11eb-94d9-fc8c4cd44cfa.png)

## Production - Demo:
**Pipeline**:

1. Use OpenCV to turn input video frames into images.
2. Input those images to Generator Model to generate high resolution images.
3. Then, use OpenCV to turn high resolution images to a video for display.

![image](https://user-images.githubusercontent.com/71629218/102676363-e28f2600-41cf-11eb-89a8-5e26013ae649.png)

**Flask app**:

- Upscale a whole video via upload.
- Upscale real time video recording from webcam.

![image](https://user-images.githubusercontent.com/71629218/102676757-a78df200-41d1-11eb-9ad1-baf2d336a9c7.png)
