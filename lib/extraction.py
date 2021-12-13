import cv2
import os
import numpy as np


def getFrame(videoName, res):
    if (os.path.exists(f"{res}/1.jpg")):
        print("images already exists")
        return
    vidcap = cv2.VideoCapture(videoName)
    success, image = vidcap.read()
    count = 1
    success = True
    while success:
        success, image = vidcap.read()
        if (success):
            cv2.imwrite(f"{res}/{count}.jpg", image)  # save frame as JPEG file
        if cv2.waitKey(10) == 27:  # exit if Escape is hit
            break
        count += 1


def readLabel(data, imsz):
    ## 좌표 값 및 크기 계산 후 return
    labels = []
    dh, dw, _ = imsz
    for d in data:
        _, x, y, w, h = list(map(float, d.split(" ")))
        l = int((x - w / 2) * dw)
        r = int((x + w / 2) * dw)
        t = int((y - h / 2) * dh)
        b = int((y + h / 2) * dh)

        if l < 0:
            l = 0
        if r > dw - 1:
            r = dw - 1
        if t < 0:
            t = 0
        if b > dh - 1:
            b = dh - 1
        labels.append([l, r, t, b])
    return labels


def logoOverlay(image, logo, alpha=1.0, x=0, y=0, scale=1.0):
    (h, w) = image.shape[:2]
    image = np.dstack([image, np.ones((h, w), dtype="uint8") * 255])
    overlay = cv2.resize(logo, None, fx=scale, fy=scale)
    (wH, wW) = overlay.shape[:2]
    output = image.copy()
    # blend the two images together using transparent overlays
    try:
        if x < 0: x = w + x
        if y < 0: y = h + y
        if x + wW > w: wW = w - x
        if y + wH > h: wH = h - y
        print(x, y, wW, wH)
        overlay = cv2.addWeighted(output[y:y + wH, x:x + wW], alpha,
                                  overlay[:wH, :wW], 1.0, 0)
        output[y:y + wH, x:x + wW] = overlay
    except Exception as e:
        print("Error: Logo position is overshooting image!")
        print(e)
    output = output[:, :, :3]
    return output


def mask(param, imsz):
    dh, dw, _ = imsz
    rectangle = np.zeros((dh, dw), dtype="uint8")
    for i in param:
        # x,y 좌표, 크기
        x, y, xs, ys = i
        print(x, y)
        print(ys)
        cv2.rectangle(rectangle, (x, y), (xs, ys), 255, -1)
    return rectangle