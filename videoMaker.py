import os
import lib.extraction as video
import torch
import cv2
import warnings
import numpy as np
warnings.filterwarnings('ignore')

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

## set folder & files
root_folder = os.getcwd()
result_folder = os.path.join(root_folder, "res")
model_folder = os.path.join(root_folder, "model_sprite")
work_folder = os.path.join(root_folder, "test")
yolo_folder = os.path.join(root_folder, "yolov5")
result_images = os.path.join(root_folder, "result_images")
main_video = f"{root_folder}/main.mp4"

## 1. video frame extraction
# video.getFrame(main_video, result_folder)

## 2. images detect & create x1y1x2y2
#### 2.1. model load
model = torch.hub.load(f'{yolo_folder}',
                       'custom',
                       path=f'{model_folder}/best.pt',
                       source='local',
                       force_reload=True)  # local repo
model.conf = 0.88

#### 2.2. image read & sort
ext = "jpg"
files = os.listdir(result_folder)
files = [int(file.split(".")[0]) for file in files]
files.sort()

logo = cv2.imread(f"{root_folder}/logo.png", cv2.IMREAD_UNCHANGED)
#### 2.3. create labels & creat result
for file in files:
    rimg = cv2.imread(f"{result_folder}/{file}.{ext}")
    oimg = cv2.imread(f"{result_folder}/{file}.{ext}")
    results = model(rimg)
    resultLabel = []
    for d in results.xyxy[0].tolist():
        x1, y1, x2, y2, _, __, = d
        resultLabel.append([int(x1), int(y1), int(x2), int(y2)])
    print(resultLabel)
    if (len(resultLabel) == 0):
        continue
    mask = video.mask(resultLabel, rimg.shape)
    print(oimg.shape, mask.shape)
    ## inpainting
    dst = cv2.inpaint(oimg, mask, 3, cv2.INPAINT_NS)
    ## add new logo
    for rl in resultLabel:
        x1, y1, x2, y2 = rl
        width = x2 - x1
        height = y2 - y1
        res = cv2.resize(logo,
                         dsize=(width, height),
                         interpolation=cv2.INTER_CUBIC)
        dst = video.logoOverlay(dst, res, x=x1, y=y1)
        cv2.imwrite(f'{result_images}/{file}.jpg', dst)
## END: video create
files = os.listdir(result_images)
frame = []
files = [int(file.split(".")[0]) for file in files]
files.sort()

for file in files:
    img = cv2.imread(f"{result_images}/{file}.{ext}")
    height, width, layers = img.shape
    size = (width, height)
    frame.append(img)

out = cv2.VideoWriter('project.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 20.0,
                      size)
for i in range(len(frame)):
    out.write(frame[i])
out.release()
