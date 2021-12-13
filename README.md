# BrandLogoChanger

BLC(Brand Logo Changer) is change brand logo in image or video

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
conda install -c anaconda flask -y
```

```
git clone https://github.com/ultralytics/yolov5
cd yolov5
pip install -r requirements.txt
```

## How to use it?

if you use this program you have to install `Requirement Software`

## Sample 1 : starbucks to buggerking

video url : https://www.youtube.com/watch?v=EJF919p_hb0

## Sample 2 : sprite to buggerking

video url : https://www.youtube.com/watch?v=2Ti83X7-370
