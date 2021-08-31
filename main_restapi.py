
import io
import cv2
import pandas as pd
import yaml
import torch
import argparse
import numpy as np
import json

from collections import OrderedDict

from core.id_scan import pt_detect
from models.experimental import attempt_load

import pprint
from PIL import Image
from flask import Flask, request
from waitress import serve

app = Flask(__name__)

DETECTION_URL = "/id-scan"


@app.route(DETECTION_URL, methods=["POST"])
def predict():
    if not request.method == "POST":
        return

    if request.files.get("file_param_1"):
        image_file = request.files["file_param_1"]
        # face_file = request.files["file_param_face"]
        image_bytes = image_file.read()

        img = Image.open(io.BytesIO(image_bytes))

        result = pt_detect(img, device, models, gray=False, byteMode=True, perspect=False)

        if result is None:
            result_json = pd.DataFrame().to_json(orient="columns")
            print('검출 실패', '\n---------------------------------------')
        else:
            # df = result.mkDataFrame()
            # df.columns = ['ocr_result']
            # df['err_code'] = 10
            # # print(df, '\n---------------------------------------')
            # result_json = df.to_json(orient="columns")
            result_json = json.dumps(result.mkDataFrameDict(), ensure_ascii=False)
            pprint.pprint(result_json)

        return result_json


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=5000, type=int, help="port number")
    parser.add_argument('--gpu', type=int, default=-1)
    args = parser.parse_args()

    gpu = args.gpu
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
    img_path, cls_weight, jumin_weight, driver_weight, passport_weight, welfare_weight, alien_weight, hangul_weight = \
        config['images'], config['cls-weights'], config['jumin-weights'], config['driver-weights'], \
        config['passport-weights'], config['welfare-weights'], config['alien-weights'], config['hangul-weights']
    f.close()

    # 모델 세팅
    cls_model = attempt_load(cls_weight, map_location=device)
    jumin_model = attempt_load(jumin_weight, map_location=device)
    driver_model = attempt_load(driver_weight, map_location=device)
    passport_model = attempt_load(passport_weight, map_location=device)
    welfare_model = attempt_load(welfare_weight, map_location=device)
    alien_model = attempt_load(alien_weight, map_location=device)
    hangul_model = attempt_load(hangul_weight, map_location=device)
    models = (cls_model, jumin_model, driver_model, passport_model, welfare_model, alien_model, hangul_model)
    print('----- 모델 로드 완료 -----')

    # app.run(host="0.0.0.0", port=args.port)
    serve(app, host='0.0.0.0', port=args.port)
