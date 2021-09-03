import os
import cv2
import math
import numpy as np


# 검출 박스 상자의 겹친 비율
def compute_intersect_ratio(rect1, rect2):
    x1, y1, x2, y2 = rect1[0], rect1[1], rect1[2], rect1[3]
    x3, y3, x4, y4 = rect2[0], rect2[1], rect2[2], rect2[3]

    if x2 < x3: return 0
    if x1 > x4: return 0
    if y2 < y3: return 0
    if y1 > y4: return 0

    left_up_x = max(x1, x3)
    left_up_y = max(y1, y3)
    right_down_x = min(x2, x4)
    right_down_y = min(y2, y4)

    width = right_down_x - left_up_x
    height = right_down_y - left_up_y

    original = max((y2 - y1) * (x2 - x1), (y4 - y3) * (x4 - x3))
    intersect = width * height

    ratio = int(intersect / original * 100)

    return ratio


# 겹친 상자 제거 (30% 이상) mrz 용
def remove_intersect_box_mrz(mrzLine):
    i, line = 0, mrzLine.copy()
    while True:
        if i > len(line) - 2: break
        if compute_intersect_ratio(line[i][0], line[i+1][0]) > 30:
            lose = i if line[i][2] < line[i+1][2] else i+1
            del line[lose]
        else: i += 1

    return line


# 겹친 상자 제거 (30% 이상)
def remove_intersect_box(det):
    result = det.copy()
    for i in range(0, len(result)):
        if i > len(result) - 2: break
        for y in range(i + 1, len(result)):
            if y > len(result) - 1: break
            if compute_intersect_ratio((result[i]['x1'], result[i]['y1'], result[i]['x2'], result[i]['y2']),
                                       (result[y]['x1'], result[y]['y1'], result[y]['x2'],
                                        result[y]['y2'])) > 30:
                del result[y]
    return result


# 겹친 상자 제거 (30% 이상) - 정렬 하기 힘든 경우
def unsorted_remove_intersect_box(lists):
    for i in range(0, len(lists)-1):
        if i > len(lists)-2: break
        for y in range(i+1, len(lists)-1):
            if y > len(lists)-1: break
            if compute_intersect_ratio(lists[i][0], lists[y][0]) > 30:
                if lists[i][1] > lists[y][1]:
                    del lists[y]
                    y -= 1
                else:
                    del lists[i]
                    i -= 1

    result = []
    for l in lists:
        result.append(l[0])
    return result


# 겹친 상자 제거 (50% 이상) - 정렬 하기 힘든 경우 - pytorch의 det인 경우
def unsorted_remove_intersect_box_det(det, ciou):
    for i in range(0, len(det)-1):
        if i > len(det)-2: break
        for y in range(i+1, len(det)):
            if y > len(det)-1: break
            if compute_intersect_ratio(det[i][0], det[y][0]) > ciou:
                if det[i][1] > det[y][1]:
                    del det[y]
                    y -= 1
                else:
                    del det[i]

    result = []
    for *rect, conf, cls in det:
        result.append((rect, conf, cls))
    return result


# 어파인 로테이션
def affine_rotation(src, angle):
    cp = (src.shape[1] / 2, src.shape[0] / 2)
    affine_mat = cv2.getRotationMatrix2D(cp, angle, 1)

    dst = cv2.warpAffine(src, affine_mat, (0, 0))
    return dst


# 각도 검출
def degree_detection(p1_x, p1_y, p2_x, p2_y):
    radian = math.atan2(p2_y - p1_y, p2_x - p1_x)
    return radian * 180 / np.pi


# 라인단위 정렬
def sort_v2(mrzStr):
    mrzStr.sort(key=lambda x: x[0][0])

    firstLine_firstChar = mrzStr[0] if mrzStr[0][0][1] < mrzStr[1][0][1] else mrzStr[1]
    firstLine_lastChar = mrzStr[len(mrzStr) - 1] if mrzStr[len(mrzStr) - 1][0][1] < mrzStr[len(mrzStr) - 2][0][1] else mrzStr[len(mrzStr) - 2]
    standard = firstLine_firstChar if firstLine_firstChar[0][1] > firstLine_lastChar[0][1] else firstLine_lastChar

    mrzFirst, mrzSecond = [], []
    for c in mrzStr:
        if c[0][1] <= standard[0][1]:
            mrzFirst.append(c)
        else:
            mrzSecond.append(c)

    mrzFirst.sort(key=lambda x: x[0][0])
    mrzSecond.sort(key=lambda x: x[0][0])

    return mrzFirst, mrzSecond


# 라인단위 정렬
def line_by_line_sort(mrzStr):
    middleChar, mrzFirst, mrzSecond = mrzStr[0][0][0], [], []
    for c in mrzStr:
        if c[0][0][0][1] < middleChar[0][3]:
            mrzFirst.append(c)
        else:
            mrzSecond.append(c)

    mrzFirst.sort(key=lambda x: x[0][0][0][0])
    mrzSecond.sort(key=lambda x: x[0][0][0][0])

    return mrzFirst, mrzSecond


# 검출 값 한꺼번에 정렬
def all_sort(mrzStr):
    mrzFirst, mrzSecond = mrzStr[0:44], mrzStr[44:]
    mrzFirst.sort(key=lambda x: x[0])
    mrzSecond.sort(key=lambda x: x[0])

    return mrzFirst, mrzSecond


# 이름 Surname, GivenName 분리
def spiltName(name):
    nameCheck, nameBool = 0, False
    surName, givenNames = '', ''
    for s in name:
        if s == '<':
            nameCheck += 1
        else:
            if nameCheck == 1: nameCheck = 0

        if nameCheck == 2 and nameBool is True:
            break
        elif nameCheck == 2:
            nameCheck = 0
            nameBool = True
        elif nameBool is False:
            surName += s
        else:
            givenNames += s

    return surName, givenNames


# 이미지 크롭
def crop(rect, im0s):
    x1, y1, x2, y2 = int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3])
    img_crop = im0s[y1:y2, x1:x2]
    return img_crop


# 검출 여부 확인
def nonCheck(item, obj):
    return obj[item] if item in obj else ('0', 0)


# 검출 상자 그리기
def plot_one_box(x, im, color=(128, 128, 128), label=None, line_thickness=3):
    # Plots one bounding box on image 'im' using OpenCV
    assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to plot_on_box() input image.'
    tl = line_thickness or round(0.002 * (im.shape[0] + im.shape[1]) / 2) + 1  # line/font thickness
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(im, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(im, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(im, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


# 이미지 출력 (openCV)
def showImg(det, names, im0s, colors, real):
    realImg, drawImg = real.copy(), im0s.copy()
    for *rect, conf, cls in reversed(det):
        label = f'{names[int(cls)]} {conf:.2f}'
        plot_one_box(rect, drawImg, label=label, color=colors[int(cls)], line_thickness=1)

    appendImg = np.append(cv2.resize(realImg, (drawImg.shape[1], drawImg.shape[0])), drawImg, axis=1)
    cv2.imshow("result", cv2.resize(appendImg, (1280, 400)))


# 디렉토리 탐색
def watchDir(path):
    output = os.listdir(path)
    filelist = []

    for i in output:
        if os.path.isdir(path + "/" + i):
            filelist.extend(watchDir(path + "/" + i))
        elif os.path.isfile(path + "/" + i):
            filelist.append(path + "/" + i)

    return filelist


# 딕셔너리 받아서 마스킹
def dict_masking(dict, img):
    masking = img.copy()
    for d in dict:
        masking[d['y1']: d['y2'], d['x1']: d['x2']] = 0
    return masking


# 중심점 찾기
def center_point(rect):
    x1, y1, x2, y2 = rect
    x_center = int(x1 + (x2 - x1) / 2)
    y_center = int(y1 + (y2 - y1) / 2)
    return x_center, y_center


# 두 점의 거리
def distance_two_points(p1, p2):
    p1X, p1Y = p1
    p2X, p2Y = p2
    l1 = p2X - p1X
    l2 = p2Y - p1Y
    return math.sqrt(math.pow(l1, 2) + math.pow(l2, 2))


# 범위 상자 안에 있는 값 검출
def rect_in_value(det, item, names, charMode=True):
    result, resultDic, item_rect = '', {}, item
    for *rect, conf, cls in det:
        if charMode and len(names[int(cls)]) != 1: continue  # 클래스명이 한글자가 아니면 지나감
        if (rect[0][0][0] > item_rect[0][0][0]) and (rect[0][0][1] > item_rect[0][0][1]) and (rect[0][0][2] < item_rect[0][0][2]) and (
                rect[0][0][3] < item_rect[0][0][3]):
            resultDic[rect[0][0][0]] = names[int(cls)]
    for d in sorted(resultDic):
        result += resultDic[d]

    return result


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