from numpy import random

from core.util import *
from core.correction import *
from core.general import *
from core.passport import Passport
from core.image_handler import ImagePack


def detecting(model, img, im0s, device, img_size, half, confidence, iou):
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

    return detect


def detect(path, model, device, opt, mode=''):
    mrz_thres = 50
    imgz, confidence, iou = opt
    fileName = path.split('/')[-1]
    detLenList = []

    half = device.type != 'cpu'
    if half:
        model.half()
    stride = int(model.stride.max())
    img_size = check_img_size(imgz, s=stride)

    # 데이터 세팅
    image_pack = ImagePack(path, img_size, stride)
    real = image_pack.getOImg()

    # 클래스, 색상 셋 로드
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    img, im0s = image_pack.getImg()
    det = detecting(model, img, im0s, device, img_size, half, confidence, iou)

    detLenList.append(len(det))
    mrzRect = False
    # mrz 존재 여부 판단
    for *rect, conf, cls in det:
        if names[int(cls)] == 'mrz':
            mrzRect = rect

    # mrz가 없거나, 검출 항목이 mrz_thres개 이하인경우 rotate (회전된 이미지라고 판단)
    mrz = False
    if mrzRect is False or len(det) < mrz_thres:
        for deg in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]:
            r_im0s = cv2.rotate(real, deg)
            image_pack.setImg(r_im0s)

            img, im0s = image_pack.getImg()

            det = detecting(model, img, im0s, device, img_size, half, confidence, iou)
            detLenList.append(len(det))

            for *rect, conf, cls in det:
                if names[int(cls)] == 'mrz':
                    mrz = True
                    mrzRect = rect

            # mrz가 검출되고 검출 항목이 mrz_thres개 이상이면 정상
            if mrzRect and len(det) > mrz_thres:
                break
            else:
                mrzRect = False

    # 4방향 전부 mrz 검출 실패 또는 검출 항목 mrz_thres개 이하
    if mrzRect is False and mrz is False:
        print(f'{fileName} 검출 실패')
        if mode == 'show':
            cv2.imshow("result", cv2.resize(real, (640, 400)))
            cv2.waitKey(0)
        return None

    # mrz는 나왔는데 검출 항목 mrz_thres개 이하인 경우 최대로 검출된 방향으로 처리
    if mrzRect is False and mrz is True and len(detLenList) == 4:
        index = detLenList.index(max(detLenList))
        rot_list = [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]

        if index != 0:
            r_im0s = cv2.rotate(real, rot_list[index-1])
            image_pack.setImg(r_im0s)
        else:
            image_pack.setImg(real)

        img, im0s = image_pack.getImg()
        det = detecting(model, img, im0s, device, img_size, half, confidence, iou)

        for *rect, conf, cls in det:
            if names[int(cls)] == 'mrz':
                mrzRect = rect

    # 이미지 크롭
    img, im0s = image_pack.passportCrop(mrzRect)
    det = detecting(model, img, im0s, device, img_size, half, confidence, iou)

    # 중복 상자 제거
    det_tmp = []
    for *rect, conf, cls in det:
        if names[int(cls)] != 'mrz':
            det_tmp.append((rect, conf))

    # 중복 상자 제거
    rect_list = unsorted_remove_intersect_box(det_tmp)

    # 기울기 조정
    rect_list.sort(key=lambda x: x[0])
    firstChar = rect_list[0] if rect_list[0][1] < rect_list[1][1] else rect_list[1]
    lastChar = rect_list[len(rect_list) - 1] \
        if rect_list[len(rect_list) - 1][1] < rect_list[len(rect_list) - 2][1] else rect_list[len(rect_list) - 2]
    p1_x, p1_y = firstChar[0], firstChar[1]
    p2_x, p2_y = lastChar[0], lastChar[1]

    degree = degree_detection(p1_x, p1_y, p2_x, p2_y)
    im0s = affine_rotation(im0s, degree)

    image_pack.setImg(im0s)
    img, im0s = image_pack.getImg()
    det = detecting(model, img, im0s, device, img_size, half, confidence, iou)

    # 검출 값 처리
    # mrz 검출
    mrz_rect = None
    for *rect, conf, cls in det:
        if names[int(cls)] == 'mrz':
            mrz_rect = rect
            break

    # mrz 검출 실패
    if mrz_rect is None:
        print(f'{fileName} 검출 실패')
        if mode == 'show':
            showImg(det, names, im0s, colors, real)
            cv2.waitKey(0)
        return None

    # mrz 정렬
    result, mrzStr, = '', []
    for *rect, conf, cls in det:
        if (rect[0] > mrz_rect[0]) and (rect[1] > mrz_rect[1]) and (rect[2] < mrz_rect[2]) and (
                rect[3] < mrz_rect[3]):
            cls_name = names[int(cls)] if names[int(cls)] != 'sign' else '<'
            mrzStr.append((rect, cls_name, conf))

    mrzStr.sort(key=lambda x: x[0][1])

    # 라인단위 정렬 v2
    # mrzFirst, mrzSecond = sort_v2(mrzStr)

    # 라인 단위 정렬
    mrzFirst, mrzSecond = line_by_line_sort(mrzStr)

    # 한번에 정렬
    # mrzFirst, mrzSecond = all_sort(mrzStr)

    # 중복 상자 제거
    mrzFirst, mrzSecond = remove_intersect_box(mrzFirst), remove_intersect_box(mrzSecond)

    # 결과 저장
    firstLine, secondLine = "", ""
    for rect, mrz_cls, conf in mrzFirst:
        firstLine += mrz_cls
    for rect, mrz_cls, conf in mrzSecond:
        secondLine += mrz_cls

    if len(firstLine) < 44:
        for i in range(len(firstLine), 44):
            firstLine += '<'

    if len(secondLine) < 44:
        for i in range(len(secondLine), 44):
            secondLine += '<'

    surName, givenNames = spiltName(firstLine[5:44])
    passportType = typeCorrection(mrzCorrection(firstLine[0:2].replace('<', ''), 'dg2en'))
    issuingCounty = nationCorrection(mrzCorrection(firstLine[2:5], 'dg2en'))
    sur = mrzCorrection(surName.replace('<', ' ').strip(), 'dg2en')
    given = mrzCorrection(givenNames.replace('<', ' ').strip(), 'dg2en')

    passportNo = secondLine[0:9].replace('<', '')
    nationality = nationCorrection(mrzCorrection(secondLine[10:13], 'dg2en'))
    birth = mrzCorrection(secondLine[13:19].replace('<', ''), 'en2dg')
    sex = sexCorrection(mrzCorrection(secondLine[20].replace('<', ''), 'dg2en'))
    expiry = mrzCorrection(secondLine[21:27].replace('<', ''), 'en2dg')
    personalNo = mrzCorrection(secondLine[28:35].replace('<', ''), 'en2dg')

    passport = Passport(passportType, issuingCounty, sur, given, passportNo, nationality, birth, sex, expiry,
                        personalNo)

    if mode == '':
        print(f'{fileName} 검출 성공')
    if mode == 'print':
        printResult(fileName, passport)
    if mode == 'show':
        showImg(det, names, im0s, colors, real)
        printResult(fileName, passport)
        cv2.waitKey(0)
    if mode == 'save':
        drawImg = im0s.copy()
        for *rect, conf, cls in reversed(det):
            label = f'{names[int(cls)]} {conf:.2f}'
            plot_one_box(rect, drawImg, label=label, color=colors[int(cls)], line_thickness=1)
        printResult(fileName, passport)
        cv2.imwrite(f'data/result/ori_{fileName}', real)
        cv2.imwrite(f'data/result/det_{fileName}', drawImg)

    return passport


