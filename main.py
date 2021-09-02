import os
import time
import yaml
import torch
import argparse
import pandas as pd
from easyocr.easyocr import Reader

from core.util import watchDir
from core.id_card import *
from core.id_scan import pt_detect
from models.experimental import attempt_load


def main(arg):
    gpu, gray, ciou = arg.gpu, arg.gray, arg.ciou
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
    img_path, cls_weight, jumin_weight, driver_weight, passport_weight, welfare_weight, alien_weight, hangul_weight, encnum_weight = \
        config['images'], config['cls-weights'], config['jumin-weights'], config['driver-weights'], \
        config['passport-weights'], config['welfare-weights'], config['alien-weights'], config['hangul-weights'], config['encnum-weights']
    f.close()

    # 모델 세팅
    reader = Reader(['en', 'ko'])


    driver_model = attempt_load(driver_weight, map_location=device)

    models = driver_model

    print('----- 모델 로드 완료 -----')

    result_csv = pd.DataFrame()
    fileList = watchDir(img_path)
    if fileList:
        images = [x for x in fileList if x.split('.')[-1].lower() in img_formats]
        for img in images:

            # pytorch 검출
            df = pt_detect(img, device, models, ciou, reader, gray=gray, byteMode=False, perspect=False)

            result_csv = pd.concat([result_csv, df])

    result_csv.to_csv('csv/result.csv', index=False, encoding='utf-8-sig')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--ciou', type=float, default=20)
    parser.add_argument('--gray', type=bool, default=False)
    opt = parser.parse_args()
    main(opt)
