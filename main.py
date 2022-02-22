import time

import yaml
import torch
import argparse

from core.util import watchDir
from core.scan import pt_detect
from easyocr.easyocr import Reader
from utils.torch_utils import time_sync
from models.experimental import attempt_load


def main(arg):
    gpu, gray, ciou, lang = arg.gpu, arg.gray, arg.ciou, arg.lang
    img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo', 'gif']

    # 디바이스 세팅
    if gpu == -1:
        dev = 'cpu'
    else:
        dev = f'cuda:{gpu}'
    device = torch.device(dev)

    # config 로드
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    img_path, detection = config['images'], config['detection']
    f.close()

    # 모델 세팅
    lang_list = lang.split('/')
    reader = Reader(lang_list)

    detection_model = attempt_load(detection, map_location=device)

    print('----- 모델 로드 완료 -----')

    fileList = watchDir(img_path)
    if fileList:
        images = [x for x in fileList if x.split('.')[-1].lower() in img_formats]
        for img in images:

            # pytorch 검출
            try:
                start_time = time_sync()
                pt_detect(img, device, detection_model, ciou, reader, gray=gray, byteMode=False)
                print('detecting time:', time_sync() - start_time)
            except Exception as e:
                print("detecting Fail")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--ciou', type=float, default=20)
    parser.add_argument('--gray', type=bool, default=False)
    parser.add_argument('--lang', type=str, default='en/ko')
    opt = parser.parse_args()
    main(opt)
