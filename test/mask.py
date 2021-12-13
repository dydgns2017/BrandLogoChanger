import cv2
import numpy as np
import matplotlib.pyplot as plt
import os, torch


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


def test(param, imsz):
    dh, dw, _ = imsz
    rectangle = np.zeros((dw, dh), dtype="uint8")
    for i in param:
        # x,y 좌표, 크기
        x, y, xs, ys = i
        print(x, y)
        print(ys)
        cv2.rectangle(rectangle, (x, y), (xs, ys), 255, -1)
    return rectangle


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


if __name__ == "__main__":
    ## set foloder
    rootfolder = os.getcwd()
    model_folder = os.path.join(rootfolder, "model_sprite")
    work_folder = os.path.join(rootfolder, "test")
    yolo_folder = os.path.join(rootfolder, "yolov5")
    ## model load
    model = torch.hub.load(f'{yolo_folder}',
                           'custom',
                           path=f'{model_folder}/best.pt',
                           source='local',
                           force_reload=True)  # local repo
    ## image read
    img = cv2.imread(f"{work_folder}/test.jpg")
    test_img = cv2.imread(f"{work_folder}/test.jpg")
    ## logo process
    logo = cv2.imread(f"{work_folder}/testlogo.png", cv2.IMREAD_UNCHANGED)

    results = model(img)
    print(results.xyxy[0].tolist())
    resultLabel = []
    for d in results.xyxy[0].tolist():
        x1, y1, x2, y2, _, __, = d
        resultLabel.append([int(x1), int(y1), int(x2), int(y2)])
    mask = test(resultLabel, img.shape)
    #results.show()
    cv2.imshow('test', test_img)
    cv2.imshow('mask', mask)
    dst = cv2.inpaint(test_img, mask, 3, cv2.INPAINT_NS)

    for rl in resultLabel:
        x1, y1, x2, y2 = rl
        width = x2 - x1
        height = y2 - y1
        res = cv2.resize(logo,
                         dsize=(width, height),
                         interpolation=cv2.INTER_CUBIC)
        dst = logoOverlay(dst, res, x=x1, y=y1)
    cv2.imshow("out", dst)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
