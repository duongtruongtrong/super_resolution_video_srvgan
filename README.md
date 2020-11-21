# Super Resolution Video - Video Upscale
Upscale video from low resolution to high resolution to reduce the effect of low bandwidth internet on video chat (meeting) or to reduce streaming video size without losing video quality.

# Problem to Solve:
![image](https://user-images.githubusercontent.com/71629218/99867213-16cff080-2bea-11eb-9d85-0ed480ec6aac.png)

**Eliminate pixelated videos** in streaming, video call/chat/conference.

# Idea Origin:
**NVIDIA DLSS**

![image](https://user-images.githubusercontent.com/71629218/99867318-2c91e580-2beb-11eb-92f2-076c187e10e3.png)

Upscaling video games rendered at low resolution (e.g.: 720p) to high resolution (e.g.: 1080p) with high frame per second (FPS) and still preserve details in the image.

# Project Details:
## Dataset:
REalistic and Dynamic Scenes dataset for Video Temporal Super-Resolution (frame interpolation) ([REDS_VTSR](https://seungjunnah.github.io/Datasets/reds.html)) dataset includes **~70 GB** of high resolution **15-frame, 30-frame, 60-frame** videos in form of **1280 x 720** (720p) images (1 frame = 1 image).

## Training:
### GAN model:
**Model reference**: [Fast-SRGAN](https://github.com/HasnainRaz/Fast-SRGAN)

**Input**: 180p videos.

**Output**: 720p videos.

4 times upscale.

## Production:
Upscale real time video recording.

**Video frames to images**:
https://theailearner.com/2018/10/15/extracting-and-saving-video-frames-using-opencv-python/
```
import cv2
 
# Opens the Video file
cap = cv2.VideoCapture(0) # Extracting frames from camera
i=0
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        break
    cv2.imwrite('kang'+str(i)+'.jpg',frame)
    i+=1
 
cap.release()
cv2.destroyAllWindows()
```

**Images to video**:
https://theailearner.com/2018/10/15/creating-video-from-images-using-opencv-python/
```
import cv2
 
# Opens the Video file
import cv2
import numpy as np
import glob

img_array = []
for filename in glob.glob('C:/New folder/Images/*.jpg'):
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)


out = cv2.VideoWriter('project.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()
```
