import os
import cv2
import numpy as np
from core.point import Point
import math
import ctypes as c


# 행렬 교차점
def parametricIntersect(r1, t1, r2, t2, x, y):
    ct1 = np.cos(t1)  # matrix element a
    st1 = np.sin(t1)  # b
    ct2 = np.cos(t2)  # c
    st2 = np.sin(t2)  # d
    d = ct1 * st2 - st1 * ct2  # determinative (rearranged matrix for inverse)
    if d != 0.0:
        x = int(((st2 * r1 - st1 * r2) / d))
        y = int(((-ct2 * r1 + ct1 * r2) / d))
        return True, x, y
    else:
        return False, x, y  # Lines are parallel and will NEVER intersect!


# 주어진 직사각형 내의 점이 맞는지 확인
def is_point_within_given_rectangle(x, y, x_min, x_max, y_min, y_max):
    if x_min <= x <= x_max and y_min <= y <= y_max: return True
    else: return False


# 교차되는 좌표 검출
def find_intersection_coordinates(image, lines):
    img_border = 110.0
    points = []
    # Define borders. So, given lines should intersect inside these borders
    x_min, x_max, y_min, y_max = -img_border, image.shape[1] + img_border, -img_border, image.shape[0] + img_border

    for line1 in lines:
        rho1, theta1 = line1[0][0], line1[0][1]
        for line2 in lines:
            rho2, theta2 = line2[0][0], line2[0][1]
            inter_x, inter_y = 0.0, 0.0
            pi_result, inter_x, inter_y = parametricIntersect(rho1, theta1, rho2, theta2, inter_x, inter_y)
            if pi_result \
                    and is_point_within_given_rectangle(inter_x, inter_y, x_min, x_max, y_min, y_max):
                check = False
                for p in points:
                    if check: break
                    if p == Point(inter_x, inter_y): check = True
                if check is False: points.append(Point(inter_x, inter_y))
    return points


# 라인 사이의 임계값 정도
def threshold_degree_between_lines(angle, theta1, theta2):
    if angle < 0 or angle > 180:
        print("Wrong angle input !!!")

    angle_minus_180 = 180 - angle
    angle *= np.pi / 180
    angle_minus_180 *= np.pi / 180

    if abs(theta1 - theta2) < angle or angle_minus_180 < abs(theta1 - theta2):
        return False
    else:
        return True


def close_lines(r1, t1, r2, t2):
    if abs(r1 - r2) * abs(t1 - t2) < 8.5: return True
    return False


# 단계 건너뛰기
def update_step(step):
    if step != 1: step -= 1
    return step


# 허프 변환 직선 검출 -> 4개의 꼭지점 검출
def detect_4_points_from_hough_lines(image, min_line_length, rho=1, theta=np.pi / 180, step=10, line_thresh=50):
    img_border = 110.0
    angle_thresh_between_lines = 5.0
    # 중요 라인 추출 Extracted most important lines # Just store all lines ----------> just for experiment
    important_lines, zet_lines = [], []

    # Define borders. So, given lines should intersect inside these borders
    x_min, x_max, y_min, y_max = -img_border, image.shape[1] + img_border, -img_border, image.shape[0] + img_border

    # Loop through lines length and decrease line lenth limit to detect more and more lines. Ex: 150, 140, 130 ...
    for _ in range(min_line_length, line_thresh, -step):
        # 허프 변환 직선 검출
        lines = cv2.HoughLines(image, rho, theta, line_thresh, min_line_length)

        if len(lines) == 0: continue

        for zet in lines:
            zet_lines.append(zet)

        # Assign first line as important line. Start comparing this line with next ones
        if len(important_lines) == 0: important_lines.append(lines[0])

        # if more than 1 lines are detected, decrease step so that smaller step probably detects only 1 line
        if len(lines) > 1: step = update_step(step)

        for new_line in lines:  # Iterate through all detected line
            # Find rho and theta for new line
            rho1, theta1 = new_line[0][0], new_line[0][1]

            for l in important_lines:  # if new lines already exists in important_line, then skip
                if np.equal(l.all(), new_line.all()): continue
            necessary_line = True  # Assume that new line is necessary

            for old_line in important_lines:  # Iterate through all old important lines
                rho2, theta2 = old_line[0][0], old_line[0][1]  # Find rho and theta for old line
                inter_x, inter_y = 0.0, 0.0  # Coordinate for intersection of two lines

                # 두 선의 교차점 좌표 검출
                pi_result, inter_x, inter_y = parametricIntersect(rho1, theta1, rho2, theta2, inter_x, inter_y)

                if pi_result \
                        and is_point_within_given_rectangle(inter_x, inter_y, x_min, x_max, y_min, y_max) \
                        and threshold_degree_between_lines(angle_thresh_between_lines, theta1, theta2) is False:
                    necessary_line = False

                if close_lines(rho1, theta1, rho2, theta2): necessary_line = False

            # Append lines if they satisfy given conditions
            if necessary_line:
                important_lines.append(new_line)
                if len(important_lines) == 4:
                    return important_lines  # if 4 lines are detected, return 4 points and exit the function
    return important_lines


def sort_4_points(image, points):
    points.sort(key=lambda point: point.y)
    p1 = points[0:2]
    p2 = points[2:]

    p1.sort(key=lambda point: point.x)
    p2.sort(key=lambda point: point.x)
    p1.append(p2[1])
    p1.append(p2[0])
    return p1

    # results_points = [points[0], points[0], points[0], points[0]]  # result to be returned
    # initial_distance = image.shape[1] if image.shape[0] < image.shape[1] else image.shape[
    #     0]  # find longest side of an image
    # dist_0 = initial_distance
    # dist_1 = initial_distance
    # dist_2 = initial_distance
    # dist_3 = initial_distance  # all distances assigned to longest side of an image
    #
    # print(initial_distance)
    #
    # for p in points:  # Find point -> top left
    #     if np.linalg.norm(p) < dist_0:
    #         dist_0 = np.linalg.norm(p)
    #         results_points[0] = p
    #
    # for p in points:  # Find point -> top right
    #     if np.linalg.norm((p[0] - image.shape[0], p[1] - 0.0)) < dist_1:
    #         dist_1 = np.linalg.norm((p[0] - image.shape[0], p[1] - 0.0))
    #         results_points[1] = p
    #
    # for p in points:  # Find point -> bottom right
    #     if np.linalg.norm((p[0] - image.shape[0], p[1] - image.shape[1])) < dist_2:
    #         dist_2 = np.linalg.norm((p[0] - image.shape[0], p[1] - image.shape[1]))
    #         results_points[2] = p
    #
    # for p in points:  # Find point -> bottom left
    #     if np.linalg.norm((p[0] - 0.0, p[1] - image.shape[1])) < dist_3:
    #         dist_3 = np.linalg.norm((p[0] - 0.0, p[1] - image.shape[1]))
    #         results_points[3] = p
    #
    # return results_points


def find_triangle_area(p1, p2, p3):
    sides = [Point.norm(p1 - p2), Point.norm(p2 - p3), Point.norm(p3 - p1)]
    s = (sides[0] + sides[1] + sides[2]) / 2
    return np.sqrt(s * (s - sides[0]) * (s - sides[1]) * (s - sides[2]))


def find_rect_area(points):
    triangle1 = find_triangle_area(points[0], points[1], points[2])
    triangle2 = find_triangle_area(points[0], points[3], points[2])
    return triangle1 + triangle2


def ratio_rect(points):
    smallest = Point.norm(points[0] - points[1])
    biggest = smallest
    if Point.norm(points[0] - points[1]) < smallest: smallest = Point.norm(points[0] - points[1])
    if Point.norm(points[1] - points[2]) < smallest: smallest = Point.norm(points[1] - points[2])
    if Point.norm(points[2] - points[3]) < smallest: smallest = Point.norm(points[2] - points[3])
    if Point.norm(points[0] - points[1]) > biggest: biggest = Point.norm(points[0] - points[1])
    if Point.norm(points[1] - points[2]) > biggest: biggest = Point.norm(points[1] - points[2])
    if Point.norm(points[2] - points[3]) > biggest: biggest = Point.norm(points[2] - points[3])
    return smallest / biggest


def perspective(image):
    tmp, img = image.copy(), image.copy()

    # 그레이스케일 Gray Scale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 양방향 필터 Bilateral Filter
    img = cv2.bilateralFilter(img, 9, 75, 75, borderType=cv2.BORDER_DEFAULT)

    # 적응형 이진화 Adaptive Threshold
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 13, 0)

    # Border 생성
    img = cv2.copyMakeBorder(img, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    # 캐니 에지 검출 Canny edge detection 임계값 50
    canny_img = cv2.Canny(img, 50, 100)

    # 모폴로지 팽창 Dilate an image
    kernel = np.ones((3, 3), np.uint8)
    canny_img = cv2.dilate(canny_img, kernel)

    # 외곽선 검출 Find contours
    final_contours = []
    contours, hierarchy = cv2.findContours(canny_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 임계값 100 이하의 외곽선 제거 Remove objects smaller than certain threshold // 100 now
    for i in range(len(contours)):
        if len(contours[i]) > 100:
            final_contours.append(contours[i])

    # 검출 된 외각선 그리기 (사용 안함)
    nee = np.zeros((tmp.shape[0], tmp.shape[1], 3), np.uint8)

    for i in range(len(contours)):
        cv2.drawContours(nee, contours, i, (255, 255, 255), cv2.FILLED, 8)

    # 임계값 제거한 외각선 그리기
    contours = final_contours

    nee_thres = np.zeros((tmp.shape[0], tmp.shape[1], 3), np.uint8)

    for i in range(len(contours)):
        cv2.drawContours(nee_thres, contours, i, (255, 255, 255), cv2.FILLED, 8)

    # 그레이스케일 Grayscale (사용 안함)
    nee_gray = cv2.cvtColor(nee_thres, cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv2.findContours(nee_gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE, offset=(0, 0))

    # 모폴로지 침식 Erode
    image_eroded_with_5x5_kernel = cv2.erode(nee_gray, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))

    # 외각선 검출 Find Contours
    contours, hierarchy = cv2.findContours(image_eroded_with_5x5_kernel.copy(), cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_NONE)  # Find contours

    # 외각선 검출 안되면 원본 return
    if len(contours) == 0:
        return tmp.copy()

    # 가장 큰 면적으로 가장 큰 윤곽 찾기 Find biggest contour with biggest area
    max_contour_id, max_area_val = 0, 0.0

    for i in range(len(contours)):
        if max_area_val < cv2.contourArea(contours[i]):
            max_contour_id = i
            max_area_val = cv2.contourArea(contours[i])

    # 가장 큰 외각선으로 그리기 Get biggest contour
    chelsea = np.zeros((tmp.shape[0], tmp.shape[1]), np.uint8)
    cv2.drawContours(chelsea, contours, max_contour_id, (255, 255, 255), cv2.FILLED, 8, maxLevel=0)

    # 외각선 검출 안되면 원본 return
    if len(contours) == 0:
        return tmp.copy()

    # 캐니 에지 검출 Canny edge detection
    chelsea = cv2.Canny(chelsea, 50, 100)

    # 확률적 허프 변환 직선 검출 Show Hough lines Probability ################ JUST TO SEE
    hough_lines_p_mat = np.zeros((tmp.shape[0], tmp.shape[1], 3), np.uint8)

    linesP = cv2.HoughLinesP(chelsea, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=10)

    # 직선 그리기
    for i in range(len(linesP)):
        line = linesP[i][0]
        cv2.line(hough_lines_p_mat, (line[0], line[1]), (line[2], line[3]), (0, 0, 255), 3, cv2.LINE_AA)

    # Border 100 ndarray 생성 (사용 안함)
    overBorderGap = 100
    bd = np.zeros((tmp.shape[0], tmp.shape[1], 3), np.uint8)
    bd_wide = np.zeros((bd.shape[0] + overBorderGap, bd.shape[1] + overBorderGap, 3), np.uint8)

    # 허프 변환 직선 검출 (사용 안함)
    lines = cv2.HoughLines(chelsea.copy(), 1, np.pi / 180, 160)

    # 전처리 완료 이미지
    cv2.imwrite('processing.jpg', chelsea.copy())

    # 허프 변환 직선 검출 후 4개의 꼭지점 검출
    final_lines = detect_4_points_from_hough_lines(chelsea.copy(), chelsea.shape[1])
    final_points = find_intersection_coordinates(tmp, final_lines)

    # 검출 된 직선이 4개 이하면 return
    if len(final_lines) < 4:
        return tmp.copy()

    # 검출 된 꼭지점이 4개 이하면 return
    if len(final_points) < 4:
        return tmp.copy()

    # 검출 된 직선이 4개가 아니면 return
    if len(final_points) != 4:
        return tmp.copy()

    # 꼭지점 정렬
    finish = sort_4_points(tmp, final_points)
    # 일부 포인트가 둘 이상 존재하는지 확인 Check if some points exist more than one
    for p in finish:
        repeat = 0
        for p1 in finish:
            if p == p1: repeat += 1
        if repeat > 2:
            return tmp.copy()

    dists = [Point.norm(finish[0] - finish[1]),
             Point.norm(finish[1] - finish[2]),
             Point.norm(finish[2] - finish[3]),
             Point.norm(finish[3] - finish[0])]

    width_output = dists[0] if dists[0] > dists[2] else dists[2]
    height_output = dists[1] if dists[1] > dists[3] else dists[3]
    width_output *= 1.1
    height_output *= 1.1
    out_size = (width_output, height_output)

    # Ignore if rectangle is too small
    src_img_area = tmp.shape[1] * tmp.shape[0]
    if (src_img_area / (find_rect_area(finish) * ratio_rect(finish))) > 20:
        return tmp.copy()
    # width_output = width_output.tup()
    # height_output = height_output.tup()
    if len(final_lines) == 4:
        p, q = [], []
        p.append(finish[0].tup())
        q.append((0, 0))

        p.append(finish[1].tup())
        q.append((width_output, 0))

        p.append(finish[2].tup())
        q.append((width_output, height_output))

        p.append(finish[3].tup())
        q.append((0, height_output))

        p = np.float32(p)
        q = np.float32(q)

        rotation = cv2.getPerspectiveTransform(p, q)

        final_result = cv2.warpPerspective(tmp, rotation, (int(out_size[0]), int(out_size[1])))
        print("perspective_transform 완료")
        return final_result

    return tmp.copy()


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)

    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def hough(img):
    lines = cv2.HoughLines(img, 1, math.pi / 180, 250)

    dst = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if lines is not None:
        for i in range(lines.shape[0]):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            cos_t = math.cos(theta)
            sin_t = math.sin(theta)
            x0, y0 = rho * cos_t, rho * sin_t
            alpha = 1000
            pt1 = (int(x0 - alpha * sin_t), int(y0 + alpha * cos_t))
            pt2 = (int(x0 + alpha * sin_t), int(y0 - alpha * cos_t))
            cv2.line(dst, pt1, pt2, (0, 0, 255), 2, cv2.LINE_AA)

    return dst


def hough_segment(img):
    lines = cv2.HoughLinesP(img, 1, math.pi / 180, 160, minLineLength=50, maxLineGap=5)

    dst = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if lines is not None:
        for i in range(lines.shape[0]):
            pt1 = (lines[i][0][0], lines[i][0][1])
            pt2 = (lines[i][0][2], lines[i][0][3])
            cv2.line(dst, pt1, pt2, (0, 0, 255), 2, cv2.LINE_AA)

    return dst


def auto_scan_image(path, name):

    image = cv2.imread(path)
    orig = image.copy()

    r = 800.0 / image.shape[0]
    dim = (int(image.shape[1] * r), 800)
    image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edged = cv2.Canny(gray, 70, 250)
    # cv2.imwrite(f'C:/Users/home/Desktop/py_workspace/id_detect/data/pespective_result/per_{name}.jpg', edged)

    dst = hough_segment(edged)

    cv2.imwrite(f'C:/Users/home/Desktop/py_workspace/id_detect/data/pespective_result/per_{name}.jpg', dst)

    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4:
            screenCnt = approx
            break

    if 'screenCnt' in locals():
        pass
    else:
        print(name + '-------- 실패')
        # cv2.imwrite(f'C:/Users/home/Desktop/py_workspace/id_detect/data/pespective_result/per_{name}_0.jpg', image)
        return

    cv2.drawContours(image, [screenCnt], -1, (0, 0, 255), 2)
    # cv2.imwrite(f'C:/Users/home/Desktop/py_workspace/id_detect/data/pespective_result/per_{name}_0.jpg', image)

    rect = order_points(screenCnt.reshape(4, 2) / r)

    (topLeft, topRight, bottomRight, bottomLeft) = rect

    w1 = abs(bottomRight[0] - bottomLeft[0])
    w2 = abs(topRight[0] - topLeft[0])
    h1 = abs(topRight[1] - bottomRight[1])
    h2 = abs(topLeft[1] - bottomLeft[1])

    maxWidth = max([w1, w2])
    maxHeight = max([h1, h2])

    dst = np.float32([[0, 0], [maxWidth-1, 0], [maxWidth-1, maxHeight-1], [0, maxHeight-1]])

    M = cv2.getPerspectiveTransform(rect, dst)

    warped = cv2.warpPerspective(orig, M, (int(maxWidth), int(maxHeight)))

    #cv2.imwrite(f'C:/Users/home/Desktop/py_workspace/id_detect/data/pespective_result/per_{name}.jpg', warped)
    print(name + '-------- 성공')


def test(path, name):
    image = cv2.imread(path)
    orig = image.copy()

    r = 800.0 / image.shape[0]
    dim = (int(image.shape[1] * r), 800)
    image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # # filter2D 블러
    # kernel = np.ones((5, 5), np.float32) / 25
    # gray = cv2.filter2D(gray, -1, kernel)
    #
    # # 평균 블러
    # gray = cv2.blur(gray,(3,3))

    # 가우시안 블러
    # gray = cv2.GaussianBlur(gray, (7, 7), 0)
    # gray = cv2.GaussianBlur(gray, (0, 0), 1)
    #
    # # 미디안 블러
    # gray = cv2.medianBlur(gray, 9)
    #
    # # bilateralFilter 블러
    # gray = cv2.bilateralFilter(gray, -1, 75, 75)
    # gray = cv2.bilateralFilter(gray, -1, 10, 5)
    # gray = cv2.bilateralFilter(gray, -1, 10, 5)

    # 소벨 에지
    # edged = cv2.Sobel(gray, cv2.CV_32FC1, 1, 0)
    # edged = cv2.Sobel(edged, cv2.CV_32FC1, 0, 1)

    #케니 에지
    # edged = cv2.Canny(gray, 70, 250)
    # edged = cv2.Canny(gray, 50, 100)
    # edged = cv2.Canny(gray, 10, 20)

    gray = cv2.GaussianBlur(gray, (0, 0), 1)

    # 양방향 필터 Bilateral Filter
    # gray = cv2.bilateralFilter(gray, 9, 75, 75, borderType=cv2.BORDER_DEFAULT)
    gray = cv2.bilateralFilter(gray, -1, 10, 5)

    # 적응형 이진화 Adaptive Threshold
    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 13, 0)

    # Border 생성
    # gray = cv2.copyMakeBorder(gray, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    # 캐니 에지 검출 Canny edge detection 임계값 50
    canny_img = cv2.Canny(gray, 10, 20)

    # 모폴로지 팽창 Dilate an image
    kernel = np.ones((3, 3), np.uint8)
    canny_img = cv2.dilate(canny_img, kernel)

    # 외곽선 검출 Find contours
    final_contours = []
    contours, hierarchy = cv2.findContours(canny_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 임계값 100 이하의 외곽선 제거 Remove objects smaller than certain threshold // 100 now
    for i in range(len(contours)):
        if len(contours[i]) > 100:
            final_contours.append(contours[i])

    # 검출 된 외각선 그리기 (사용 안함)
    nee = np.zeros((gray.shape[0], gray.shape[1], 3), np.uint8)

    for i in range(len(contours)):
        cv2.drawContours(nee, contours, i, (255, 255, 255), cv2.FILLED, 8)

    # 임계값 제거한 외각선 그리기
    contours = final_contours

    nee_thres = np.zeros((gray.shape[0], gray.shape[1], 3), np.uint8)

    for i in range(len(contours)):
        cv2.drawContours(nee_thres, contours, i, (255, 255, 255), cv2.FILLED, 8)

    # 그레이스케일 Grayscale (사용 안함)
    nee_gray = cv2.cvtColor(nee_thres, cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv2.findContours(nee_gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE, offset=(0, 0))

    # 모폴로지 침식 Erode
    image_eroded_with_5x5_kernel = cv2.erode(nee_gray, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))

    # 외각선 검출 Find Contours
    contours, hierarchy = cv2.findContours(image_eroded_with_5x5_kernel.copy(), cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_NONE)  # Find contours



    # 가장 큰 면적으로 가장 큰 윤곽 찾기 Find biggest contour with biggest area
    max_contour_id, max_area_val = 0, 0.0

    for i in range(len(contours)):
        if max_area_val < cv2.contourArea(contours[i]):
            max_contour_id = i
            max_area_val = cv2.contourArea(contours[i])

    # 가장 큰 외각선으로 그리기 Get biggest contour
    chelsea = np.zeros((gray.shape[0], gray.shape[1]), np.uint8)
    cv2.drawContours(chelsea, contours, max_contour_id, (255, 255, 255), cv2.FILLED, 8, maxLevel=0)

    edged = cv2.Canny(chelsea, 10, 20)

    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4:
            screenCnt = approx
            break

    if 'screenCnt' in locals():
        pass
    else:
        print(name + '-------- 윤곽선 계산 실패')
        # cv2.imwrite(f'C:/Users/home/Desktop/py_workspace/id_detect/data/pespective_result/per_{name}_0.jpg', image)
        return

    imageSize = edged.shape[0] * edged.shape[1]
    if cv2.contourArea(screenCnt) / imageSize < 0.1:
        print(name + '-------- 잘못된 윤곽선 실패')
        return

    cv2.drawContours(image, [screenCnt], -1, (0, 0, 255), 2)
    # cv2.imwrite(f'C:/Users/home/Desktop/py_workspace/id_detect/data/pespective_result/per_{name}_0.jpg', image)

    rect = order_points(screenCnt.reshape(4, 2) / r)

    (topLeft, topRight, bottomRight, bottomLeft) = rect

    w1 = abs(bottomRight[0] - bottomLeft[0])
    w2 = abs(topRight[0] - topLeft[0])
    h1 = abs(topRight[1] - bottomRight[1])
    h2 = abs(topLeft[1] - bottomLeft[1])

    maxWidth = max([w1, w2])
    maxHeight = max([h1, h2])

    dst = np.float32([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]])

    M = cv2.getPerspectiveTransform(rect, dst)

    warped = cv2.warpPerspective(orig, M, (int(maxWidth), int(maxHeight)))

    cv2.imwrite(f'C:/Users/home/Desktop/py_workspace/id_detect/data/pespective_result/per_{name}.jpg', warped)
    print(name + '-------- 성공')


if __name__ == "__main__":
    input_dir = 'C:/Users/home/Desktop/work/test_img/perspect_test_img/'
    # input_dir = 'C:/Users/home/Desktop/idscan/IDCard/driver/2020-04-02/'

    for path in os.listdir(input_dir):
        test(input_dir + path, path.split('.')[0])

