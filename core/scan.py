
import yaml
from core.util import *
from core.general import *
from core.image_handler import ImagePack

import pandas as pd


# pt 모델 설정 세팅
def model_setting(model, half, imgz):
    if half:
        model.half()
    stride = int(model.stride.max())
    img_size = check_img_size(imgz, s=stride)

    names = model.module.names if hasattr(model, 'module') else model.names

    return model, stride, img_size, names


# pt 검출
def detecting(model, img, im0s, device, img_size, half, option, ciou=20):
    confidence, iou = option
    if device.type != 'cpu':
        model(torch.zeros(1, 3, img_size, img_size).to(device).type_as(next(model.parameters())))

    # 이미지 정규화
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # 추론 & NMS 적용
    prediction = model(img, augment=False)[0]
    prediction = non_max_suppression(prediction, confidence, iou, classes=None, agnostic=False)

    detect = None
    for _, det in enumerate(prediction):
        obj, det[:, :4] = {}, scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()
        detect = det.cpu()

    # 중복 상자 제거
    detList = []
    for *rect, conf, cls in detect:
        detList.append((rect, conf, cls))

    detect = unsorted_remove_intersect_box_det(detList, ciou)

    return detect


def detection(det, names):
    det.sort(key=lambda x: x[2])
    label_list = []
    rect_list = []
    for *rect, conf, cls in det:
        rects = rect[0][0]
        label_list.append(names[int(cls)])
        rect_list.append([int(rects[0]), int(rects[2]), int(rects[1]), int(rects[3])])

    return rect_list, label_list


def pt_detect(path, device, models, ciou, reader, gray=False, byteMode=False):
    driver_weights = models

    half = device.type != 'cpu'
    # config 로드
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    img_size, confidence, iou = config['detection-size'], config['detection-confidence'], config['detection-iou']
    detection_option = (img_size, confidence, iou)
    f.close()

    model, stride, img_size, names = model_setting(driver_weights, half, detection_option[0])
    image_pack = ImagePack(path, img_size, stride, byteMode=byteMode, gray=gray)
    img, im0s = image_pack.getImg()
    det = detecting(model, img, im0s, device, img_size, half, detection_option[1:], ciou)
    rect_list, label_list = detection(det, names)

    result = reader.recogss(im0s, rect_list)

    result_line, i = [], 0
    print(f'----------- {path} -----------')
    for r in result:
        line = r[1]
        print(f'{label_list[i]} : {line}')
        result_line.append(line)
        i += 1
    print('-------------------------------')