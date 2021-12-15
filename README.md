# BrandLogoChanger

BLC(Brand Logo Changer) is change brand logo in image or video

![그림1](https://user-images.githubusercontent.com/36920367/146116321-a5f2c59b-08f4-4687-ab26-c5590c3fb7de.png)

## Sample Video

- [sample_video_1](sample/project.mp4)
- [sample_video_2](sample/project2.mp4)
- [sample_video_3](sample/project3.mp4)
- [sample_video_4](sample/project4.mp4)

## Requirement Software

```python
conda create -n blc python=3.7
conda activate blc

## you will be modify cudatoolkit version prefer in your system
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge -y

## if use with only cpu
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y

## flask with cv2
conda install -c conda-forge opencv -y
```

```
git clone https://github.com/ultralytics/yolov5
cd yolov5
pip install -r requirements.txt
```

## How to use it?

if you use this program you have to install `Requirement Software`

## 1. Logo Detection

### 1.1. starbucks logo detection

```
python yolov5/detect.py --weights model_starbucks/best.pt --source https://www.youtube.com/watch?v=EJF919p_hb0 --conf-thres 0.87
```

### 1.2. sprite logo detection

```
python yolov5/detect.py --weights model_sprite/best.pt --source https://www.youtube.com/watch?v=2Ti83X7-370 --conf-thres 0.87
```

## 2. Logo Change

if you want change video & logo, modify videoMaker.py code or main.mp4, main_logo.png file change

```
python videoMaker.py
```

## 3. Sample Video

- Starbucks Logo to Ediya Logo

- Sprite Logo Korean Cidar Logo

## Autohrs

- Yonghoon Jung, @dydgns2017
- Haneul Lim, @sky81219
- Byongmo Kang, @-

## Reference

- https://towardsdatascience.com/image-inpainting-with-a-single-line-of-code-c0eef715dfe2 (inpainting)
- https://github.com/ultralytics/yolov5 (Yolov5)
