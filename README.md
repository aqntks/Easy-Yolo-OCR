# Easy Yolo OCR

![issue badge](https://img.shields.io/github/license/aqntks/recog)
![issue badge](https://img.shields.io/badge/build-passing-brightgreen)
![issue badge](https://img.shields.io/badge/%EB%8B%A4%EA%B5%AD%EC%96%B4-%EC%A7%80%EC%9B%90-yellow)
[![LinkedIn Badge](http://img.shields.io/badge/LinkedIn-@InpyoHong-0072b1?style=flat&logo=linkedin&link=https://www.linkedin.com/in/inpyo-hong-886781212/)](https://www.linkedin.com/in/inpyo-hong-886781212/)

원하는 영역만 텍스트 검출을 진행하세요  

이 저장소는 [yolov5](https://github.com/ultralytics/yolov5) 와 [EasyOCR](https://github.com/JaidedAI/EasyOCR) 을 활용한 프로젝트입니다.


## Introduction

기존의 OCR(Optical character recognition) 프로세스는 Text Detection 모델로 문자 영역을 검출한 후 Text Recognition 모델을 통해 문자를 인식하는 방식입니다. 이러한 OCR 모델은 원하는 문서나 이미지 내의 문자 전체를 인식하는 데 효과적입니다.    

하지만 이미지나 문서 내의 특정 영역 문자만 탐지하기 원하는 경우 불필요한 영역까지 검출하여 검출 속도가 오래 걸리며 결과 값을 처리하기 불편합니다.

다양한 이미지에서 특정한 패턴이나 영역에 위치한 문자만 검출하기 원하시는 분들을 위해 Easy Yolo OCR을 제안합니다.

Easy Yolo OCR은 텍스트 영역을 검출하기 위한 Text Detection 모델을 객체 탐지에 사용되는 Object Detection 모델로 변경하였습니다. 자신에게 맞는 커스텀 Detection 모델을 학습하고 원하는 서식의 원하는 영역만 검출하세요.

Object Detection 모델은 Real Time Object Detection 분야에서 활발히 활용되는 [yolov5](https://github.com/ultralytics/yolov5) 를 사용합니다. OCR 프로세스는 [EasyOCR](https://github.com/JaidedAI/EasyOCR) 을 벤치마킹 하였으며 Text Recognition 모델은 Clova AI Research의 [deep-text-recognition-benchmark](https://github.com/clovaai/deep-text-recognition-benchmark) 을 통하여 학습되었습니다.

- 기존 OCR 프로세스

![](res/original.jpg)

- Easy Yolo OCR 프로세스

![](res/easyyoloocr.jpg)


## Installation


``` bash
$ git clone https://github.com/aqntks/Easy-Yolo-OCR
$ cd Easy-Yolo-OCR
$ pip install -r requirements.txt
```

## Prepare Training Data
학습 이미지의 BOX 영역

```bash
# image_label.txt

0 0.6659722222222223 0.11302083333333333 0.4013888888888889 0.06770833333333333
0 0.48333333333333334 0.12552083333333333 0.025 0.036458333333333336
0 0.5145833333333334 0.1265625 0.02638888888888889 0.036458333333333336
0 0.5479166666666667 0.125 0.0375 0.0375
0 0.5798611111111112 0.125 0.029166666666666667 0.03333333333333333
0 0.6145833333333334 0.12447916666666667 0.03194444444444445 0.03854166666666667
0 0.6479166666666667 0.12395833333333334 0.03194444444444445 0.041666666666666664
0 0.68125 0.12447916666666667 0.03194444444444445 0.03229166666666666
0 0.7145833333333333 0.12395833333333334 0.03194444444444445 0.03125
0 0.7465277777777778 0.12552083333333333 0.029166666666666667 0.034375
0 0.78125 0.12239583333333333 0.03194444444444445 0.03229166666666666
0 0.8104166666666667 0.125 0.029166666666666667 0.0375
0 0.8423611111111111 0.12343749999999999 0.034722222222222224 0.036458333333333336
```



custom.yaml 파일 생성 후 아래 내용 작성

```bash
# custom.yaml

path: ./dataset/custom_data
train: ./dataset/custom_train.txt
val:  ./dataset/custom_valid.txt
test:  ./dataset/custom_train_test.txt

nc: 1  # number of classes
names: ['text_area']  # class names
```

## Train Detection Model
``` bash
$ cd yolov5
```

```bash
$ python train.py --data coco.yaml --weights 'yolov5s.pt' --batch-size 64
                                              yolov5m.pt               40
                                              yolov5l.pt               24
                                              yolov5x.pt               16
```

## Setting Config
```bash
# config.yaml

images: image                                # 검출 이미지 폴더

detection: weights/example.pt                # 학습된 detecting model
detection-size: 640                          # 검출 이미지 사이즈
detection-confidence: 0.25                   # detecting confidence
detection-iou: 0.45                          # detecting iou
```


## OCR

```bash
$ python main.py --gpu 0 --lang en ko
$ python main.py --gpu 0 --lang en
$ python main.py --gpu -1 --lang ko         # --gpu -1 : cpu mode
```

