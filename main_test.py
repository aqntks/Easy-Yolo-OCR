import cv2
import torch
import numpy as np
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.plots import plot_one_box
from utils.torch_utils import select_device, time_synchronized

import argparse
from pytesseract import *


def main(opt):
    weights, images, img_size, confidence, iou, device = \
        opt.weights, opt.img, opt.img_size, opt.conf, opt.iou, opt.device

    # 디바이스 세팅
    device = select_device(device)  # 첫번째 gpu 사용
    half = device.type != 'cpu'  # gpu + cpu 섞어서 사용

    # 모델 로드
    model = attempt_load(weights, map_location=device)
    stride = int(model.stride.max())
    img_size = check_img_size(img_size, s=stride)
    if half:
        model.half()

    # 데이터 세팅
    dataset = LoadImages(images, img_size=img_size, stride=stride)
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # 추론 실행
    if device.type != 'cpu':
        model(torch.zeros(1, 3, img_size, img_size).to(device).type_as(next(model.parameters())))
    for path, img, im0s, vid_cap in dataset:

        print(type(img))
        print(np.shape(img))
        print(type(im0s))
        print(np.shape(im0s))

        startT = time_synchronized()
        # 이미지 정규화
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # 추론 & NMS 적용
        prediction = model(img, augment=False)[0]
        prediction = non_max_suppression(prediction, confidence, iou, classes=None, agnostic=False)

        # 검출 값 처리
        for i, det in enumerate(prediction):
            if len(det):
                obj, det[:, :4] = {}, scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()

                # 이미지 출력
                # showImg(det, names, im0s, colors)

                # 영역 검출 항목 저장
                for *rect, conf, cls in reversed(det):
                    cls = names[int(cls)]
                    if cls.split('_')[0] == 'title' or cls == 'kor':
                        obj['title'] = (cls, rect)
                    elif cls == 'mrz':
                        obj['title'] = (cls, rect)
                        obj[cls] = (cls, rect)
                    elif cls.split('_')[0] == 'local':
                        obj['local'] = (cls, rect)
                    else:
                        obj[cls] = (cls, rect)

                # 분류 여부
                titleDetectCheck = obj['title'][0] if 'title' in obj else None
                if titleDetectCheck is None:
                    print("분류 실패")
                    continue

                    # 결과 저장
                ids = Id(det, names, nonCheck('title', obj), nonCheck('name', obj), nonCheck('regnum', obj),
                         nonCheck('issuedate', obj))
                if obj['title'][0] == 'title_jumin':
                    detectId = Jumin(ids)
                elif obj['title'][0] == 'title_driver':
                    detectId = Driver(ids, nonCheck('local', obj), nonCheck('licensenum', obj),
                                      nonCheck('condition', obj), nonCheck('encnum', obj))
                elif obj['title'][0] == 'title_welfare':
                    detectId = Welfare(ids, nonCheck('gradetype', obj), nonCheck('expire', obj))
                elif obj['title'][0] == 'kor':
                    detectId = Alien(ids, nonCheck('nationality', obj), nonCheck('visatype', obj))
                else:
                    detectId = Passport(det, names, nonCheck('mrz', obj))

                # 결과 출력
                detectId.resultPrint(im0s)

        print("\n검출 속도: " + str(time_synchronized() - startT) + '\n')
        cv2.waitKey(0)


# 이미지 크롭
def crop(rect, im0s):
    x1, y1, x2, y2 = int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3])
    img_crop = im0s[y1:y2, x1:x2]
    return img_crop


# 테서렉트
def tesseract(img, lang='eng', style='color'):
    if style == 'gray':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    result = image_to_string(img, lang=lang, config='--psm 8')
    return result.replace(' ', '').replace('\n', '').split('(')[0].replace('', '')


# 검출 여부 확인
def nonCheck(item, obj):
    return obj[item] if item in obj else ('0', 0)


# 이미지 출력 (openCV)
def showImg(det, names, im0s, colors):
    realImg, drawImg = im0s.copy(), im0s.copy()
    for *rect, conf, cls in reversed(det):
        label = f'{names[int(cls)]} {conf:.2f}'
        plot_one_box(rect, drawImg, label=label, color=colors[int(cls)], line_thickness=1)

    appendImg = np.append(realImg, drawImg, axis=1)
    cv2.imshow("result", cv2.resize(appendImg, (1616, 504)))


# 이미지 출력 (colab)
def colabShow(det, names, im0s, colors):
    realImg, drawImg = im0s.copy(), im0s.copy()
    for *rect, conf, cls in reversed(det):
        label = f'{names[int(cls)]} {conf:.2f}'
        plot_one_box(rect, drawImg, label=label, color=colors[int(cls)], line_thickness=2)

    appendImg = np.append(realImg, drawImg, axis=1)
    cv2_imshow(cv2.resize(appendImg, (1616, 504)))


# Jumin, Driver, Welfare, Alien -> Id 상속
class Id:
    def __init__(self, det, names, title, name, regnum, issuedate):
        self.det = det
        self.names = names
        self.title = title
        self.name = name
        self.regnum = regnum
        self.issuedate = issuedate
        self.nameValue = self.setValue(name)
        self.regnumValue = self.setValue(regnum)
        self.issuedateValue = self.setValue(issuedate)

    # 검출 값 저장
    def setValue(self, item):
        if item[0] == '0': return ''
        result, resultDic, item_rect = '', {}, item[1]
        for *rect, conf, cls in reversed(self.det):
            if len(self.names[int(cls)]) != 1: continue
            if (rect[0] > item_rect[0]) and (rect[1] > item_rect[1]) and (rect[2] < item_rect[2]) and (
                    rect[3] < item_rect[3]):
                resultDic[rect[0]] = self.names[int(cls)]
        for d in sorted(resultDic):
            result += resultDic[d]
        return result

    # 결과 출력
    def resultPrint(self, im0s):
        name = tesseract(crop(self.name[1], im0s), 'kor', 'gray') if self.name[0] in 'name' else ''
        print('\ntitle: ' + self.title[0])
        if self.title[0] == 'kor':
            print('name: ' + self.nameValue)
        else:
            print('name: ' + name)
        print('regum: ' + self.regnumValue)
        print('issuedate: ' + self.issuedateValue)


class Jumin(Id):
    def __init__(self, ids):
        super().__init__(ids.det, ids.names, ids.title, ids.name, ids.regnum, ids.issuedate)


class Driver(Id):
    def __init__(self, ids, local, licensenum, condition, encnum):
        super().__init__(ids.det, ids.names, ids.title, ids.name, ids.regnum, ids.issuedate)
        self.local = local
        self.licensenum = licensenum
        self.condition = condition
        self.encnum = encnum
        self.licensenumValue = self.setValue(licensenum)
        self.conditionValue = self.setValue(condition)
        self.encnumValue = self.setValue(encnum)

    def resultPrint(self, im0s):
        super().resultPrint(im0s)
        print('local: ' + self.local[0])
        print('licensenum: ' + self.licensenumValue)
        print('condition: ' + self.conditionValue)
        print('encnum: ' + self.encnumValue)


class Welfare(Id):
    def __init__(self, ids, gradetype, expire):
        super().__init__(ids.det, ids.names, ids.title, ids.name, ids.regnum, ids.issuedate)
        self.gradetype = gradetype
        self.expire = expire
        self.gradetypeValue = self.setValue(gradetype)
        self.expireValue = self.setValue(expire)

    def resultPrint(self, im0s):
        super().resultPrint(im0s)
        print('gradetype: ' + self.gradetypeValue)
        print('expire: ' + self.expireValue)


class Alien(Id):
    def __init__(self, ids, nationality, visatype):
        super().__init__(ids.det, ids.names, ids.title, ids.name, ids.regnum, ids.issuedate)
        self.nationality = nationality
        self.visatype = visatype
        self.nationalityValue = self.setValue(nationality)
        self.visatypeValue = self.setValue(visatype)

    def resultPrint(self, im0s):
        super().resultPrint(im0s)
        print('nationality: ' + self.nationalityValue)
        print('visatype: ' + self.visatypeValue)


# Passport 클래스
class Passport:
    def __init__(self, det, names, mrz):
        self.det = det
        self.names = names
        self.mrz = mrz
        self.mrzValue = self.sortMrz(mrz)

    # MRZ 정렬
    def sortMrz(self, mrz):
        if mrz[0] == '0': return ''
        result, mrzStr, mrz_rect = '', [], mrz[1]
        for *rect, conf, cls in reversed(self.det):
            if (rect[0] > mrz_rect[0]) and (rect[1] > mrz_rect[1]) and (rect[2] < mrz_rect[2]) and (
                    rect[3] < mrz_rect[3]):
                mrzStr.append((rect[0], rect[1], self.names[int(cls)]))
        mrzStr.sort(key=lambda x: x[1])
        mrzFirst, mrzSecond = mrzStr[0:45], mrzStr[44:]
        mrzFirst.sort(key=lambda x: x[0])
        mrzSecond.sort(key=lambda x: x[0])

        for x, y, cls in mrzFirst:
            result += cls
        result += '\n'
        for x, y, cls in mrzSecond:
            result += cls

        return result

    def resultPrint(self, im0s):
        print("mrz: \n" + self.mrzValue)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='weights/id_m_last.pt')
    parser.add_argument('--img', type=str, default='data/images')
    parser.add_argument('--img-size', type=int, default=1280)
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--iou', type=float, default=0.45)
    parser.add_argument('--device', type=str, default='cpu')
    option = parser.parse_args()
    main(opt=option)
