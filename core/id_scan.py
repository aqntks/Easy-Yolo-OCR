import re
import yaml
import ctypes as c
from core.util import *
from core.id_card import Driver, Jumin
from core.general import *
from core.correction import *
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
        detect = det

    # 중복 상자 제거
    detList = []
    for *rect, conf, cls in detect:
        detList.append((rect, conf, cls))

    detect = unsorted_remove_intersect_box_det(detList, ciou)

    return detect


# 주민등록증 검출
def juminScan(det, names):
    name_conf, regnum_conf, issueDate_conf, bracket_conf = 0, 0, 0, 0
    nameRect, regnumRect, issueDateRect, bracketRect = None, None, None, None
    regnum, issueDate = "", ""

    for *rect, conf, cls in det:
        if names[int(cls)] == 'name':
            if conf > name_conf:
                name_conf = conf
                nameRect = rect
        if names[int(cls)] == 'regnum':
            if conf > regnum_conf:
                regnum_conf = conf
                regnumRect = rect
        if names[int(cls)] == 'issuedate':
            if conf > issueDate_conf:
                issueDate_conf = conf
                issueDateRect = rect

    if regnumRect is not None:
        regnum = rect_in_value(det, regnumRect, names)
    if issueDateRect is not None:
        issueDate = rect_in_value(det, issueDateRect, names)

    if nameRect:
        for *rect, conf, cls in det:
            if rect[0][0][0] > nameRect[0][0][0] and rect[0][0][1] > nameRect[0][0][1] \
                    and rect[0][0][2] < nameRect[0][0][2] and rect[0][0][3] < nameRect[0][0][3]:
                if names[int(cls)] == '(':
                    if conf > bracket_conf:
                        bracket_conf = conf
                        bracketRect = rect

    if bracketRect:
        nameRect[0][0][2] = bracketRect[0][0][0]

    return Jumin(nameRect, regnum, issueDate)


# 운전면허증 검출
def driverScan(det, names):
    name_conf, regnum_conf, issueDate_conf, local_conf, licensenum_conf, encnum_conf = 0, 0, 0, 0, 0, 0
    nameRect, regnumRect, issueDateRect, licensenumRect, encnumRect = None, None, None, None, None
    regnum, issueDate, licensenum, encnum, local = "", "", "", "", ""

    for *rect, conf, cls in det:
        if names[int(cls)] == 'name':
            if conf > name_conf:
                name_conf = conf
                nameRect = rect
        if names[int(cls)] == 'regnum':
            if conf > regnum_conf:
                regnum_conf = conf
                regnumRect = rect
        if names[int(cls)] == 'issuedate':
            if conf > issueDate_conf:
                issueDate_conf = conf
                issueDateRect = rect
        if names[int(cls)] == 'licensenum':
            if conf > licensenum_conf:
                licensenum_conf = conf
                licensenumRect = rect
        if names[int(cls)] == 'encnum':
            if conf > encnum_conf:
                encnum_conf = conf
                encnumRect = rect
        if names[int(cls)].split('_')[0] == 'local':
            if conf > local_conf:
                local_conf = conf
                local = names[int(cls)]

    if regnumRect is not None:
        regnum = rect_in_value(det, regnumRect, names)
    if issueDateRect is not None:
        issueDate = rect_in_value(det, issueDateRect, names)
    if licensenumRect is not None:
        licensenum = rect_in_value(det, licensenumRect, names)
    if encnumRect is not None:
        encnum = rect_in_value(det, encnumRect, names)

    return Driver(nameRect, regnum, issueDate, local, licensenum, encnum, encnumRect)


# 운전면허증 검출
def easyScan(det, names):
    title_conf, name_conf, regnum_conf, issueDate_conf, local_conf, licensenum_conf, encnum_conf = 0, 0, 0, 0, 0, 0, 0
    rect_list = []
    title = 'jumin'

    det.sort(key=lambda x : x[2])

    for *rect, conf, cls in det:
        rects = rect[0][0]
        if names[int(cls)].split('_')[0] == 'title':
            if conf > title_conf:
                title_conf = conf
                title = names[int(cls)].split('_')[-1]
        if names[int(cls)] == 'name':
            if conf > name_conf:
                name_conf = conf
                rect_list.append([int(rects[0]), int(rects[2]), int(rects[1]), int(rects[3])])
        if names[int(cls)] == 'regnum':
            if conf > regnum_conf:
                regnum_conf = conf
                rect_list.append([int(rects[0]), int(rects[2]), int(rects[1]), int(rects[3])])
        if names[int(cls)] == 'issuedate':
            if conf > issueDate_conf:
                issueDate_conf = conf
                rect_list.append([int(rects[0]), int(rects[2]), int(rects[1]), int(rects[3])])
        if names[int(cls)] == 'licensenum':
            if conf > licensenum_conf:
                licensenum_conf = conf
                rect_list.append([int(rects[0]), int(rects[2]), int(rects[1]), int(rects[3])])
        if names[int(cls)] == 'encnum':
            if conf > encnum_conf:
                encnum_conf = conf
                rect_list.append([int(rects[0]), int(rects[2]), int(rects[1]), int(rects[3])])

    # if title == 'jumin':
    #     name_conf = 0
    #     for *rect, conf, cls in det:
    #         rects = rect[0][0]
    #         if names[int(cls)] == 'name':
    #             if conf > name_conf:
    #                 name_conf = conf
    #                 w = rects[2] - rects[0]
    #                 rect_list[1] = [int(rects[0]), int(rects[2] - w/2), int(rects[1]), int(rects[3])]

    return rect_list


# ID 카드 분류
def id_classification(det):
    bestConf = 0
    bestId = None
    id_list = ['jumin', 'driver', 'welfare', 'alien']
    for *rect, conf, cls in det:
        if conf > bestConf:
            bestConf = conf
            bestId = cls

    if bestId is None:
        return None
    else:
        return id_list[int(bestId)]


# 여권 검출 여부 체크
def passport_classification(det, names):
    for *rect, conf, cls in det:
        if names[int(cls)] == 'mrz':
            return 'passport', (rect[0][0][0], rect[0][0][1], rect[0][0][2], rect[0][0][3])
    return None, None

# 한글 검출
def hangulScan(det, names):
    obj, name = [], ''
    for *rect, conf, cls in det:
        obj.append((rect, conf, names[int(cls)]))
        obj.sort(key=lambda x: x[0][0])
    for s, conf, cls in obj:
        name += cls

    return name


def pt_detect(path, device, models, ciou, reader, gray=False, byteMode=False, perspect=False):
    driver_weights = models

    half = device.type != 'cpu'
    # config 로드
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    img_size, confidence, iou = config['cls-img_size'], config['cls-confidence'], config['cls-iou']
    id_cls_option = (img_size, confidence, iou)
    img_size, confidence, iou = config['jumin-img_size'], config['jumin-confidence'], config['jumin-iou']
    jumin_option = (img_size, confidence, iou)
    img_size, confidence, iou = config['driver-img_size'], config['driver-confidence'], config['driver-iou']
    driver_option = (img_size, confidence, iou)
    img_size, confidence, iou = config['passport-img_size'], config['passport-confidence'], config['passport-iou']
    passport_option = (img_size, confidence, iou)
    img_size, confidence, iou = config['welfare-img_size'], config['welfare-confidence'], config['welfare-iou']
    welfare_option = (img_size, confidence, iou)
    img_size, confidence, iou = config['alien-img_size'], config['alien-confidence'], config['alien-iou']
    alien_option = (img_size, confidence, iou)
    img_size, confidence, iou = config['hangul-img_size'], config['hangul-confidence'], config['hangul-iou']
    hangul_option = (img_size, confidence, iou)
    img_size, confidence, iou = config['encnum-img_size'], config['encnum-confidence'], config['encnum-iou']
    encnum_option = (img_size, confidence, iou)
    f.close()

    model, stride, img_size, names = model_setting(driver_weights, half, driver_option[0])
    image_pack = ImagePack(path, img_size, stride, byteMode=byteMode, gray=gray, perspect=perspect)
    img, im0s = image_pack.getImg()
    det = detecting(model, img, im0s, device, img_size, half, driver_option[1:])
    rect_list = easyScan(det, names)

    # ################# 회색크롭
    # best = 0
    # for rect in rect_list:
    #     w = rect[1] - rect[0]
    #     best = w if w > best else best
    #
    # new_image = np.zeros((1, best, 3))
    # new_rect_list = []
    # nowH = 0
    #
    # for rect in rect_list:
    #     x1, x2, y1, y2 = rect[0], rect[1], rect[2], rect[3]
    #     w = x2 - x1
    #     h = y2 - y1
    #     img_crop = im0s[y1:y2, x1:x2]
    #     gray = np.zeros((h, best - w, 3))
    #     gray[:, :, :] = 80
    #     img_crop = np.concatenate([img_crop, gray], axis=1)
    #     new_image = np.concatenate([new_image, img_crop], axis=0)
    #     new_rect_list.append([0, w, nowH, nowH + h])
    #     nowH = nowH + h

    # ################# 640 대비 크롭
    # size = 640
    # best = 0
    # for rect in rect_list:
    #     w = rect[1] - rect[0]
    #     best = w if w > best else best
    #
    # if best < size:
    #     best = size
    #
    # new_image = np.zeros((1, best, 3))
    # new_rect_list = []
    # nowH = 0
    #
    # for rect in rect_list:
    #     x1, x2, y1, y2 = rect[0], rect[1], rect[2], rect[3]
    #     w = x2 - x1
    #     h = y2 - y1
    #
    #     img_crop = im0s[y1:y2, x1:x2]
    #
    #     if w < size:
    #         ratio = size / w
    #         h = int(h * ratio)
    #         img_crop = cv2.resize(img_crop, (size, h))
    #         w = size
    #
    #     gray = np.zeros((h, best - w, 3))
    #     gray[:, :, :] = 80
    #     img_crop = np.concatenate([img_crop, gray], axis=1)
    #     new_image = np.concatenate([new_image, img_crop], axis=0)
    #     new_rect_list.append([0, w, nowH, nowH + h])
    #     nowH = nowH + half


    # # 강제 rsize 버전입니다@@@@@@@@@@@@@@@@@@@@
    # size = 700
    # new_image = np.zeros((1, size, 3))
    # new_rect_list = []
    # nowH = 0
    #
    # for rect in rect_list:
    #     x1, x2, y1, y2 = rect[0], rect[1], rect[2], rect[3]
    #     w = x2 - x1
    #     h = y2 - y1
    #     ratio = size / w
    #     reH = int(h * ratio)
    #     img_crop = im0s[y1:y2, x1:x2]
    #     img_crop = cv2.resize(img_crop, (size, reH))
    #     new_image = np.concatenate([new_image, img_crop], axis=0)
    #     new_rect_list.append([0, size, nowH, nowH + reH])
    #     nowH = nowH + reH

    # cv2.imwrite('test.jpg', new_image)
    result = reader.recogss(im0s, rect_list)

    result_line = []
    print(f'----------- {path} -----------')
    for r in result:
        # line = r[1].replace(' ', '').replace('성명:', '').replace('\'', '').replace('\"', '') \
        # .replace(',', '').replace('`', '').split('(')[0].upper()
        line = r[1].replace(' ', '')
        print(line)
        result_line.append(line)
    print('-------------------------------')

    fileName = path.split('/')[-1]
    if len(result_line) == 5:
        jumin = result_line[0]
        name = result_line[1]
        licenseNum = result_line[2]
        encnum = result_line[3]
        issue = result_line[4]
    elif len(result_line) == 2:
        jumin = result_line[0]
        name = '-'
        licenseNum = '-'
        encnum = '-'
        issue = result_line[1]
    else:
        jumin = result_line[0]
        name = result_line[1]
        licenseNum = '-'
        encnum = '-'
        issue = result_line[2]

    if len(jumin) == 13 and ('-' in jumin) is False:
        jumin = jumin[:7] + '-' + jumin[7:]

    name = name.split('(')[0].replace('-', '').replace('.', '').replace('(', '')
    if len(name) > 3 and '성명' in name:
        name = name.replace('성명', '')

    jumin = jumin.replace('.', '').replace('(', '').replace('L', '1').replace('O', '0').replace('Q', '0')\
        .replace('U', '0').replace('D', '0').replace('I', '1').replace('Z', '2').replace('B', '3')\
        .replace('A', '4').replace('S', '5').replace('T', '1').replace('-', '')
    if len(jumin) == 13:
        jumin = jumin[0:6] + '-' + jumin[6:]

    if licenseNum != '-':
        licenseNum = licenseNum.replace('.', '').replace('(', '').replace('-', '').replace('L', '1').replace('O', '0').replace('Q', '0')\
        .replace('U', '0').replace('D', '0').replace('I', '1').replace('Z', '2').replace('B', '3')\
        .replace('A', '4').replace('S', '5').replace('T', '1')
        if len(licenseNum) == 12:
            licenseNum = licenseNum[0:2] + '-' + licenseNum[2:4] + '-' + licenseNum[4:10] + '-' + licenseNum[10:]

    if encnum != '-':
        en_dg_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S',
                      'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        encnum = encnum.replace('-', '').replace('.', '').replace('(', '')
        for e in encnum:
            if (e in en_dg_list) is False:
                encnum = encnum.replace(e, '')

    issue = issue.replace('-', '').replace('(', '').replace('L', '1').replace('O', '0').replace('Q', '0')\
        .replace('U', '0').replace('D', '0').replace('I', '1').replace('Z', '2').replace('B', '3')\
        .replace('A', '4').replace('S', '5').replace('T', '1')

    df = pd.DataFrame({"file": [fileName], "jumin": [jumin], "name": [name],
                       "license": [licenseNum], "encnum": [encnum], "issue": [issue]})

    return df


